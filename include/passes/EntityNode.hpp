#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/EntityPrepareNode.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/EntityRenderComponent.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	// S4: no kPhase -- TerrainNode's sole Create<> of each GBuffer key this node Modify<>s
	// already orders it after TerrainNode via Create-before-accessor, and every reader of those
	// keys (WaterNode, GTAONode, HiZDownsampleNode, DeferredNode, the Composite-phase cluster)
	// gets a real default edge from this node's writes regardless of phase -- confirmed none of
	// them Modify<> the same keys (WaterNode only Reads them, despite depth-testing against
	// them), so there's no writer-pairwise conflict left to resolve beyond FoliageNode's own
	// AnyOrder() pairing below.
	struct EntityNode: render::NodeRegistrar<EntityNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		const DispatchLoaderDynamic* dls = nullptr;

		std::uint64_t m_overrideBufferAddress{0};
		std::uint32_t m_overrideTotalInstances{0};

		void SetEntityData(std::uint64_t addr, std::uint32_t count) {
			m_overrideBufferAddress = addr;
			m_overrideTotalInstances = count;
		}

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/entity.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/entity.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/entity.frag")) {
				spdlog::critical("EntityNode shader compilation failed.");
				throw std::runtime_error("EntityNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<GBufferPosition>();
				catalog.Get<GBufferNormal>();
				catalog.Get<GBufferAlbedo>();
				catalog.Get<GBufferMaterial>();
				catalog.Get<GBufferDepth>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		// Deliberately no Read<EntityInstanceBuffer/EntityIndirectBuffer>() here, matching this
		// node's own pre-conversion Recipe exactly (it declared both at compile time but never
		// realized either). Neither key is catalog-converted yet, so an empty-desc realization
		// here would win via lesson #10 -- same reasoning as GTAONode.hpp. Ordering against
		// EntityPrepareNode (whose GetBufferDeviceAddress()/GetTotalInstances() this node reads
		// via a direct EngineNodeRegistry pointer lookup in Execute(), and whose
		// EntityIndirectBuffer this node reads via ctx.resources directly -- neither going
		// through the resource graph at all) now comes from this explicit After<>() (S4),
		// instead of being an accident of EntityPrepareNode's Prepare phase sorting earlier than
		// this node's old GBuffer phase.
		//
		// AnyOrder() on Position/Normal/Albedo/Depth (not Material, which FoliageNode doesn't
		// touch at all): both this node and FoliageNode Modify the same four keys with nothing
		// else ordering them, which the relational compiler (S3) treats as a hard ambiguity
		// error unless both sides opt in. Safe specifically because both are depth-tested opaque
		// writes to disjoint screen regions -- the visual result doesn't depend on draw order --
		// but AnyOrder() still gets a real, deterministically-directed edge (and the barrier that
		// comes with it): two same-stage Graphics nodes with *no* edge at all between them,
		// writing the same attachment-backed images, would be a genuine WAW hazard regardless of
		// how commutative the actual pixels are.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferMaterial>();
			r.Modify<GBufferDepth>().AnyOrder();
			r.After<EntityPrepareNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::Buffer    indirectBuf{nullptr};
			std::uint64_t indirectOffset = 0;

			std::uint64_t bufferAddress = m_overrideBufferAddress;
			std::uint32_t totalInstances = m_overrideTotalInstances;

			if (bufferAddress == 0 || totalInstances == 0) {
				if (auto* prepNode = render::EngineNodeRegistry::Instance().GetNode<EntityPrepareNode>()) {
					bufferAddress = prepNode->GetBufferDeviceAddress();
					totalInstances = prepNode->GetTotalInstances();
				}
			}

			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<EntityIndirectBuffer>()) {
						indirectBuf = physBuf->GetBuffer();
						indirectOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
					}
				}
			}

			if (totalInstances == 0) {
				return;
			}

			EntityPushConstants push{
				.entityBufferAddress = bufferAddress,
				.totalInstances = totalInstances,
				.baseInstanceIndex = 0
			};

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
				vk::Format::eR32G32B32A32Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Srgb,
				vk::Format::eR8G8B8A8Unorm
			};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(EntityPushConstants)
			}};
			render::GraphicsPipelineRequest      request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = vk::Format::eD32Sfloat,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, boundSets, nullptr);
			}

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(EntityPushConstants),
				&push
			);

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					indirectOffset,
					1,
					sizeof(MeshTasksIndirectCommand)
				);
			} else if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(totalInstances, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(EntityNode);

} // namespace brassica
