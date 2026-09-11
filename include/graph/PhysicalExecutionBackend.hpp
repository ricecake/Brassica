#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "graph/BarrierTranslator.hpp"
#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceState.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	// Begins/ends a dynamic-rendering pass for a Graphics-domain node's color/depth
	// realizations. Direct vk::CommandBuffer calls -- no proc-addr resolution needed. Vulkan
	// 1.3 core (already the engine's baseline: see Engine.cpp's VkPhysicalDeviceVulkan13Features
	// dynamicRendering/synchronization2) links vkCmdBeginRendering/vkCmdEndRendering directly,
	// the same way src/passes/RenderPass.cpp already calls cmd.beginRendering()/endRendering().
	class DynamicRenderingWrapper {
	public:
		// registry is const: Begin only ever reads tracked resource state (the barrier-synthesis
		// seam in PhysicalExecutionBackend::Execute is what transitions a resource before Begin
		// runs, not Begin itself) -- see PhysicalRegistry.hpp's non-const GetTexture/GetBuffer
		// overloads for the other half of that invariant.
		static bool Begin(vk::CommandBuffer cmd, const PhysicalResourceRegistry& registry, const Recipe& recipe) {
			if (!cmd || recipe.domain != ExecutionDomain::Graphics) {
				return false;
			}

			std::vector<vk::RenderingAttachmentInfo> colorAttachments;
			vk::RenderingAttachmentInfo              depthAttachment{};
			bool                                     hasDepth = false;
			vk::RenderingFragmentShadingRateAttachmentInfoKHR shadingRateAttachmentInfo{};
			bool                                     hasShadingRateAttachment = false;
			std::uint32_t                            renderWidth = 0;
			std::uint32_t                            renderHeight = 0;

			for (const auto& r : recipe.realizations) {
				if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
					continue;
				}

				auto usage = static_cast<vk::ImageUsageFlags>(r.desc.usageMask);
				if (usage & vk::ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR) {
					auto tex = registry.GetTexture(r.key);
					if (tex && tex->GetView()) {
						shadingRateAttachmentInfo.imageView = tex->GetView();
						shadingRateAttachmentInfo.imageLayout = tex->GetCurrentLayout();
						shadingRateAttachmentInfo.shadingRateAttachmentTexelSize = vk::Extent2D{16, 16};
						hasShadingRateAttachment = true;
					}
					continue;
				}

				if (r.access != AccessKind::Write && r.access != AccessKind::ReadWrite) {
					// A Read realization is sampled through a descriptor, not attached. Attaching
					// it here would ask dynamic rendering to write into a resource the barrier
					// seam has (correctly) put in a read-only layout -- invalid. This mirrors the
					// color branch below, which always required Write/ReadWrite; the depth
					// branch used to attach on any access, including Read, which the hardcoded
					// eDepthStencilAttachmentOptimal below masked until tracked-layout replaced it.
					continue;
				}

				auto tex = registry.GetTexture(r.key);
				if (!tex || !tex->GetView()) {
					continue;
				}

				renderWidth = std::max(renderWidth, r.desc.width);
				renderHeight = std::max(renderHeight, r.desc.height);

				// loadOp: eLoad only when this resource already has meaningful content *and*
				// this access isn't a plain Write (a Write means "I'm about to produce all of
				// it," so discarding is correct and cheaper). A ReadWrite realization on a
				// never-written resource still has nothing to load, which is why this checks
				// HasDefinedContents() explicitly rather than trusting AccessKind alone.
				const bool preserveExisting = tex->HasDefinedContents() && r.access != AccessKind::Write;
				const auto loadOp = preserveExisting ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;
				const auto format = static_cast<vk::Format>(tex->GetDesc().formatCode);

				// imageLayout comes from tracked state, not a hardcoded constant: by the time
				// Begin runs, PhysicalExecutionBackend::Execute has already flushed this stage's
				// Acquire barrier, which transitioned this exact realization to the layout
				// DeriveImageState(r.access, ...) computed -- the same function this file no
				// longer duplicates its own copy of.
				if (IsDepthFormat(format)) {
					depthAttachment.imageView = tex->GetView();
					depthAttachment.imageLayout = tex->GetCurrentLayout();
					depthAttachment.loadOp = loadOp;
					depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
					depthAttachment.clearValue = vk::ClearDepthStencilValue{1.0f, 0};
					hasDepth = true;
				} else {
					vk::RenderingAttachmentInfo colorAtt{};
					colorAtt.imageView = tex->GetView();
					colorAtt.imageLayout = tex->GetCurrentLayout();
					colorAtt.loadOp = loadOp;
					colorAtt.storeOp = vk::AttachmentStoreOp::eStore;
					// Per-realization, not a blanket default: see ResourceRealization::clearColor's
					// comment (Execution.hpp) -- the G-buffer's alpha channel is a data sentinel
					// deferred.frag reads, not just opacity, so it needs {0,0,0,0} while everything
					// else wants opaque black.
					colorAtt.clearValue =
						vk::ClearColorValue(r.clearColor[0], r.clearColor[1], r.clearColor[2], r.clearColor[3]);
					colorAttachments.push_back(colorAtt);
				}
			}

			if (colorAttachments.empty() && !hasDepth) {
				return false;
			}

			vk::RenderingInfo renderingInfo{
				{},
				vk::Rect2D{{0, 0}, {renderWidth ? renderWidth : 1, renderHeight ? renderHeight : 1}},
				1,
				0,
				static_cast<std::uint32_t>(colorAttachments.size()),
				colorAttachments.empty() ? nullptr : colorAttachments.data(),
				hasDepth ? &depthAttachment : nullptr,
			};
			if (hasShadingRateAttachment) {
				renderingInfo.pNext = &shadingRateAttachmentInfo;
			}
			cmd.beginRendering(renderingInfo);
			return true;
		}

		static void End(vk::CommandBuffer cmd) {
			if (cmd) {
				cmd.endRendering();
			}
		}
	};

	class PhysicalExecutionBackend {
	public:
		PhysicalExecutionBackend(PhysicalResourceRegistry& registry): m_registry(registry) {}

		void Execute(Graph& graph, const FrameContext& ctx, CommandBuffer& cmd, bool enableAliasing = true) {
			// 1. Setup recipes for the frame
			graph.Setup(ctx);

			// 2. Compile: levels nodes into dependency-respecting stages and synthesizes the
			// (resource, access, domain) metadata -- including cross-domain transfers -- that
			// sits at each stage boundary. Layouts/stages/access masks are deliberately not this
			// layer's job; see ResourceState.hpp's header comment for why.
			if (auto compileRes = graph.Compile(); !compileRes) {
				throw std::runtime_error("Graph compilation failed: " + compileRes.error().message);
			}

			const Schedule& schedule = graph.GetSchedule();
			const auto      recipes = graph.Recipes();

			// 3. Provision physical resources matching concrete recipes
			m_registry.Provision(schedule, recipes, enableAliasing);

			// 4. The graph's opaque CommandBuffer handle, viewed as real Vulkan at this boundary
			// only -- Execution.hpp's CommandBuffer stays untouched and Vulkan-free.
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			// 5. Staged execution. Nodes within a stage are provably independent of each other
			// (ScheduleStage's documented invariant), so per-node barriers between them would be
			// wrong even if they happened to be correct today -- everything a stage's nodes need
			// transitioned is therefore synthesized once, up front, into one local batch.
			for (const auto& stage : schedule.stages) {
				// Acquire = stage.preBarriers (real cross-node edges, synthesized by
				// Graph::SynthesizeBarrier) unioned with a self-entry for every realization of
				// every node in this stage. The self-entries are what close the "first use has
				// no incoming edge" gap: a node that only creates or self-modifies a resource
				// (Create<K>, Modify<K>) has no edge pointing at it, so without this it would
				// never get transitioned out of eUndefined. Built as a local copy -- schedule is
				// held by const reference, and stage.preBarriers must stay exactly what
				// Graph::Compile() produced (tests/graph/test_graph.cpp asserts stage 0's is
				// empty).
				BarrierBatch acquireBatch;
				for (const auto& mb : stage.preBarriers.Items()) {
					acquireBatch.Add(mb);
				}
				for (std::size_t nodeIndex : stage.nodes) {
					const auto& recipe = recipes[nodeIndex];
					for (const auto& r : recipe.realizations) {
						// srcDomain == dstDomain == recipe.domain: this is a node's own use of
						// its own resource, not a cross-node transfer. access comes from the
						// realization, not Graph::AccessOf's declaration-derived answer -- the
						// realization is the concrete per-frame truth Provision() already keys
						// off (PhysicalRegistry.hpp), and it's the only value in scope here.
						acquireBatch.Add(
							MemoryBarrier{
								.resource = r.key,
								.access = r.access,
								.srcDomain = recipe.domain,
								.dstDomain = recipe.domain,
							}
						);
					}
				}
				BarrierTranslator::TranslateAndDispatch(vkCmd, m_registry, acquireBatch, BarrierPhase::Acquire);

				for (std::size_t nodeIndex : stage.nodes) {
					const auto& recipe = recipes[nodeIndex];

					bool activeRendering = DynamicRenderingWrapper::Begin(vkCmd, m_registry, recipe);
					graph.ExecuteNode(nodeIndex, cmd);
					if (activeRendering) {
						DynamicRenderingWrapper::End(vkCmd);
					}

					// Mark contents defined only now, after the node has actually recorded its
					// commands -- this is what lets Begin (run before ExecuteNode, on the *next*
					// stage that touches this resource) still see "never written" and choose
					// eClear correctly for this node's own first write.
					for (const auto& r : recipe.realizations) {
						if (r.access != AccessKind::Write && r.access != AccessKind::ReadWrite) {
							continue;
						}
						if (auto tex = m_registry.GetTexture(r.key)) {
							tex->SetHasDefinedContents(true);
						} else if (auto buf = m_registry.GetBuffer(r.key)) {
							buf->SetHasDefinedContents(true);
						}
					}
				}

				// Release: the producer side of any cross-domain edge this stage originated.
				// Used as-is, straight from the schedule -- unlike Acquire, there is no
				// first-use case to synthesize here (see BarrierTranslator::DispatchRelease for
				// why this phase carries no per-resource layout at all).
				BarrierTranslator::TranslateAndDispatch(vkCmd, m_registry, stage.postBarriers, BarrierPhase::Release);
			}
		}

	private:
		PhysicalResourceRegistry& m_registry;
	};

} // namespace brassica::graph
