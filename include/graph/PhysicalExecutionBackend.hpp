#pragma once

#include <vector>

#include "graph/BarrierTranslator.hpp"
#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	class DynamicRenderingWrapper {
	public:
		static bool Begin(VkCommandBuffer cmd, const PhysicalResourceRegistry& registry, const Recipe& recipe) {
			if (cmd == VK_NULL_HANDLE || recipe.domain != ExecutionDomain::Graphics) {
				return false;
			}

#if __has_include(<vulkan/vulkan.h>) || __has_include("glad/vulkan.h")
			std::vector<VkRenderingAttachmentInfo> colorAttachments;
			VkRenderingAttachmentInfo              depthAttachment{};
			bool                                   hasDepth = false;
			uint32_t                               renderWidth = 0;
			uint32_t                               renderHeight = 0;

			for (const auto& r : recipe.realizations) {
				if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
					continue;
				}

				const auto* tex = registry.GetTexture(r.key);
				if (!tex || tex->view == VK_NULL_HANDLE) {
					continue;
				}

				if (r.desc.width > renderWidth)
					renderWidth = r.desc.width;
				if (r.desc.height > renderHeight)
					renderHeight = r.desc.height;

				VkFormat format = static_cast<VkFormat>(tex->desc.formatCode);
				bool     isDepth =
					(format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
					 format == VK_FORMAT_D16_UNORM);

				if (isDepth) {
					depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
					depthAttachment.imageView = tex->view;
					depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
					depthAttachment.loadOp = (r.access == AccessKind::Write) ? VK_ATTACHMENT_LOAD_OP_CLEAR
																			 : VK_ATTACHMENT_LOAD_OP_LOAD;
					depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
					depthAttachment.clearValue.depthStencil = {1.0f, 0};
					hasDepth = true;
				} else if (r.access == AccessKind::Write || r.access == AccessKind::ReadWrite) {
					VkRenderingAttachmentInfo colorAtt{};
					colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
					colorAtt.imageView = tex->view;
					colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
					colorAtt.loadOp = (r.access == AccessKind::Write) ? VK_ATTACHMENT_LOAD_OP_CLEAR
																	  : VK_ATTACHMENT_LOAD_OP_LOAD;
					colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
					colorAtt.clearValue.color.float32[0] = 0.0f;
					colorAtt.clearValue.color.float32[1] = 0.0f;
					colorAtt.clearValue.color.float32[2] = 0.0f;
					colorAtt.clearValue.color.float32[3] = 1.0f;

					colorAttachments.push_back(colorAtt);
				}
			}

			if (colorAttachments.empty() && !hasDepth) {
				return false;
			}

	#if defined(VK_VERSION_1_3) || defined(VK_KHR_dynamic_rendering)
			VkRenderingInfo renderingInfo{};
			renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderingInfo.renderArea.offset = {0, 0};
			renderingInfo.renderArea.extent.width = renderWidth ? renderWidth : 1;
			renderingInfo.renderArea.extent.height = renderHeight ? renderHeight : 1;
			renderingInfo.layerCount = 1;
			renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
			renderingInfo.pColorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data();
			renderingInfo.pDepthAttachment = hasDepth ? &depthAttachment : nullptr;

			PFN_vkCmdBeginRendering pfnCmdBeginRendering = reinterpret_cast<PFN_vkCmdBeginRendering>(
				vkGetDeviceProcAddr(nullptr, "vkCmdBeginRendering")
			);
			if (pfnCmdBeginRendering) {
				pfnCmdBeginRendering(cmd, &renderingInfo);
				return true;
			}
	#endif
#else
			(void)registry;
			(void)recipe;
#endif
			return false;
		}

		static void End(VkCommandBuffer cmd) {
			if (cmd == VK_NULL_HANDLE)
				return;
#if (defined(VK_VERSION_1_3) || defined(VK_KHR_dynamic_rendering)) && (__has_include(<vulkan/vulkan.h>) || __has_include("glad/vulkan.h"))
			PFN_vkCmdEndRendering pfnCmdEndRendering = reinterpret_cast<PFN_vkCmdEndRendering>(
				vkGetDeviceProcAddr(nullptr, "vkCmdEndRendering")
			);
			if (pfnCmdEndRendering) {
				pfnCmdEndRendering(cmd);
			}
#endif
		}
	};

	class PhysicalExecutionBackend {
	public:
		PhysicalExecutionBackend(PhysicalResourceRegistry& registry): m_registry(registry) {}

		void Execute(Graph& graph, const FrameContext& ctx, CommandBuffer& cmd, bool enableAliasing = true) {
			// 1. Setup recipes for the frame
			graph.Setup(ctx);

			// 2. Compile: levels nodes into dependency-respecting stages and synthesizes the
			// barriers -- including cross-domain transfers -- that sit at each stage boundary.
			if (auto compileRes = graph.Compile(); !compileRes) {
				throw std::runtime_error("Graph compilation failed: " + compileRes.error().message);
			}

			const Schedule& schedule = graph.GetSchedule();
			const auto      recipes = graph.Recipes();

			// 3. Provision physical resources matching concrete recipes
			m_registry.Provision(schedule, recipes, enableAliasing);

			// 4. Extract raw VkCommandBuffer handle if available
			VkCommandBuffer vkCmd = static_cast<VkCommandBuffer>(cmd.vkCmd);

			// 5. Staged execution: flush a stage's preBarriers once (covering every node in the
			// stage, not once per node -- that's the point of batching them), run the stage's
			// nodes, then flush postBarriers (releases for any transfer this stage originated)
			// before moving on. Nodes within a stage are provably independent of each other, so
			// nothing here decides how they're actually parallelized across queues; that's a
			// scheduling decision Compile() already made, not this loop's job.
			for (const auto& stage : schedule.stages) {
				BarrierTranslator::TranslateAndDispatch(vkCmd, m_registry, stage.preBarriers);

				for (std::size_t nodeIndex : stage.nodes) {
					const auto& recipe = recipes[nodeIndex];

					bool activeRendering = DynamicRenderingWrapper::Begin(vkCmd, m_registry, recipe);
					graph.ExecuteNode(nodeIndex, cmd);
					if (activeRendering) {
						DynamicRenderingWrapper::End(vkCmd);
					}
				}

				BarrierTranslator::TranslateAndDispatch(vkCmd, m_registry, stage.postBarriers);
			}
		}

	private:
		PhysicalResourceRegistry& m_registry;
	};

} // namespace brassica::graph
