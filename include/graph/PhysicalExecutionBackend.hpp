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

	// The attachment shape a Graphics-domain recipe implies, derived once from the registry's
	// tracked descs -- formats only, no live view/layout state. This is what a pipeline needs to
	// know at creation time (VkPipelineRenderingCreateInfo), before any command buffer exists,
	// and it is the single source of truth DynamicRenderingWrapper::Begin also reads from below:
	// previously, each pass's constructor took its own {colorFormats, depthFormat} args that had
	// to be hand-kept in sync with whatever Setup()'s realizations actually produced, with nothing
	// checking the two agreed. Reuses AttachmentRoleFor (ResourceState.hpp) so a new attachment
	// kind (like ShadingRate) only ever needs teaching in one place.
	struct DerivedAttachments {
		std::vector<vk::Format> colorFormats; // in attachment order
		vk::Format              depthFormat = vk::Format::eUndefined;
		bool                    hasShadingRate = false;
		std::uint32_t           width = 0;
		std::uint32_t           height = 0;
	};

	inline DerivedAttachments DeriveAttachments(const PhysicalResourceRegistry& registry, const Recipe& recipe) {
		DerivedAttachments out;
		if (recipe.domain != ExecutionDomain::Graphics) {
			return out;
		}

		for (const auto& r : recipe.realizations) {
			if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
				continue;
			}

			auto tex = registry.GetTexture(r.key);
			if (!tex || !tex->GetView()) {
				continue;
			}
			const auto           format = static_cast<vk::Format>(tex->GetDesc().formatCode);
			const auto           usage = tex->GetDesc().usageMask ? vk::ImageUsageFlags(tex->GetDesc().usageMask)
																  : vk::ImageUsageFlags{};
			const AttachmentRole role = AttachmentRoleFor(usage, format);

			// A shading-rate map is Read, not Write/ReadWrite (it's computed by an earlier
			// compute node), and contributes no color/depth format at all -- it rides pNext, not
			// pColorAttachments/pDepthAttachment.
			if (role == AttachmentRole::ShadingRate) {
				out.hasShadingRate = true;
				continue;
			}

			if (r.access != AccessKind::Write && r.access != AccessKind::ReadWrite) {
				// A Read realization is sampled through a descriptor, not attached -- see
				// DynamicRenderingWrapper::Begin's identical check for why this must exclude it
				// from color/depth just as much as from the real vk::RenderingAttachmentInfo list.
				continue;
			}

			out.width = std::max(out.width, r.desc.width);
			out.height = std::max(out.height, r.desc.height);

			if (role == AttachmentRole::Depth) {
				out.depthFormat = format;
			} else {
				out.colorFormats.push_back(format);
			}
		}
		return out;
	}

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

			// width/height/hasShadingRate come from here rather than being re-derived below --
			// this is the one thing Begin and pipeline-request construction (PipelineLibrary)
			// both need, and now share.
			const DerivedAttachments derived = DeriveAttachments(registry, recipe);

			std::vector<vk::RenderingAttachmentInfo>          colorAttachments;
			vk::RenderingAttachmentInfo                       depthAttachment{};
			bool                                              hasDepth = false;
			vk::RenderingFragmentShadingRateAttachmentInfoKHR shadingRateAttachment{};

			for (const auto& r : recipe.realizations) {
				if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
					continue;
				}

				auto tex = registry.GetTexture(r.key);
				if (!tex || !tex->GetView()) {
					continue;
				}
				const auto           format = static_cast<vk::Format>(tex->GetDesc().formatCode);
				const auto           usage = tex->GetDesc().usageMask ? vk::ImageUsageFlags(tex->GetDesc().usageMask)
																	  : vk::ImageUsageFlags{};
				const AttachmentRole role = AttachmentRoleFor(usage, format);

				// Its extent is the render extent divided by its texel size, not the render
				// extent itself -- so unlike color/depth, it never contributes to renderWidth/
				// renderHeight, and it needs the real view DeriveAttachments doesn't carry.
				if (role == AttachmentRole::ShadingRate) {
					shadingRateAttachment.imageView = tex->GetView();
					shadingRateAttachment.imageLayout = tex->GetCurrentLayout();
					shadingRateAttachment.shadingRateAttachmentTexelSize = vk::Extent2D{
						r.shadingRateTexelSize[0],
						r.shadingRateTexelSize[1]
					};
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

				// loadOp: eLoad only when this resource already has meaningful content *and*
				// this access isn't a plain Write (a Write means "I'm about to produce all of
				// it," so discarding is correct and cheaper). A ReadWrite realization on a
				// never-written resource still has nothing to load, which is why this checks
				// HasDefinedContents() explicitly rather than trusting AccessKind alone.
				const bool preserveExisting = tex->HasDefinedContents() && r.access != AccessKind::Write;
				const auto loadOp = preserveExisting ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;

				// imageLayout comes from tracked state, not a hardcoded constant: by the time
				// Begin runs, PhysicalExecutionBackend::Execute has already flushed this stage's
				// Acquire barrier, which transitioned this exact realization to the layout
				// DeriveImageState(r.access, ...) computed -- the same function this file no
				// longer duplicates its own copy of.
				if (role == AttachmentRole::Depth) {
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
				vk::Rect2D{{0, 0}, {derived.width ? derived.width : 1, derived.height ? derived.height : 1}},
				1,
				0,
				static_cast<std::uint32_t>(colorAttachments.size()),
				colorAttachments.empty() ? nullptr : colorAttachments.data(),
				hasDepth ? &depthAttachment : nullptr,
			};
			if (derived.hasShadingRate) {
				renderingInfo.pNext = &shadingRateAttachment;
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

	// Selects the ClearColorValue union member matching a format's numeric interpretation --
	// zero is zero in every representation, but which member of the union holds it is not opaque
	// to the driver. Covers the integer formats this codebase actually produces (the shading-rate
	// mask's eR8Uint default -- PhysicalResource.hpp's ShadingRateAttachmentDesc); every format
	// not listed here is read as normalized/float, which is true of the vast majority (color
	// attachments, sampled textures) and is the correct default.
	inline vk::ClearColorValue ZeroClearValueFor(vk::Format format) {
		switch (format) {
		case vk::Format::eR8Uint:
		case vk::Format::eR8G8Uint:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eR16Uint:
		case vk::Format::eR32Uint:
		case vk::Format::eR32G32B32A32Uint:
			return vk::ClearColorValue(0u, 0u, 0u, 0u);
		case vk::Format::eR8Sint:
		case vk::Format::eR16Sint:
		case vk::Format::eR32Sint:
			return vk::ClearColorValue(0, 0, 0, 0);
		default:
			return vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// A History<K> realization's physical resource genuinely has no producer within this frame's
	// own schedule -- PreviousFrame (Frame.hpp) exists purely to satisfy validation/phase
	// ordering and does zero real work, so on the very first frame that slot has never actually
	// been written by anyone (see PhysicalRegistry::ProvisionTemporalPairs). Every other
	// Read-access realization that looks "not yet defined" at this point in Execute() is not a
	// real gap -- it just means its producer, elsewhere in this same frame's schedule, has not
	// run *yet* (Provision runs before any node executes) and will define it correctly before
	// this consumer's stage. Scoping to isHistory specifically, rather than any undefined Read,
	// is what tells those two cases apart. Reading garbage isn't just visually wrong for a
	// History slot: a fragment-shading-rate attachment full of undefined bits is spec-invalid,
	// not merely ugly.
	// Takes Graph& (not a flat recipe span) and recurses into every Subgraph node's inner graph
	// -- a History<K> realization nested inside a Subgraph needs this exact same first-frame
	// treatment, and its recipes only exist on the inner Graph, not the outer one.
	inline void ZeroInitializeUndefinedReads(vk::CommandBuffer cmd, PhysicalResourceRegistry& registry, Graph& graph) {
		if (!cmd) {
			return;
		}

		const auto recipes = graph.Recipes();
		for (const auto& recipe : recipes) {
			if (!recipe.isActive) {
				continue;
			}
			for (const auto& r : recipe.realizations) {
				if (r.access != AccessKind::Read || !r.key->isHistory) {
					continue;
				}
				if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
					continue; // no zero-init needed/expressible for buffers or an AS here
				}

				auto tex = registry.GetTexture(r.key);
				if (!tex || tex->HasDefinedContents()) {
					continue;
				}

				const auto&         desc = tex->GetDesc();
				const auto          format = static_cast<vk::Format>(desc.formatCode);
				const bool          isDepth = IsDepthFormat(format);
				const std::uint32_t mips = desc.mips ? desc.mips : 1;
				const std::uint32_t layers = desc.layers ? desc.layers : 1;
				const auto          range = vk::ImageSubresourceRange(AspectFor(format), 0, mips, 0, layers);

				vk::ImageMemoryBarrier2 toTransferDst{};
				toTransferDst.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe);
				toTransferDst.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
				toTransferDst.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite);
				toTransferDst.setOldLayout(tex->GetCurrentLayout());
				toTransferDst.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
				toTransferDst.setImage(tex->GetImage());
				toTransferDst.setSubresourceRange(range);
				vk::DependencyInfo preClearDep{};
				preClearDep.setImageMemoryBarriers(toTransferDst);
				cmd.pipelineBarrier2(preClearDep);

				if (isDepth) {
					cmd.clearDepthStencilImage(
						tex->GetImage(),
						vk::ImageLayout::eTransferDstOptimal,
						vk::ClearDepthStencilValue{1.0f, 0},
						range
					);
				} else {
					cmd.clearColorImage(
						tex->GetImage(),
						vk::ImageLayout::eTransferDstOptimal,
						ZeroClearValueFor(format),
						range
					);
				}

				// Land back in whatever layout this realization's own access derives -- the same
				// rule Provision/BarrierTranslator use, so the very next Acquire barrier this
				// resource sees (this stage's own, synthesized right after this function
				// returns) starts from a state it actually agrees with.
				const ResourceState postClear = DeriveImageState(
					r.access,
					recipe.domain,
					desc.usageMask ? vk::ImageUsageFlags(desc.usageMask) : vk::ImageUsageFlags{},
					format
				);

				vk::ImageMemoryBarrier2 toFinal{};
				toFinal.setSrcStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
				toFinal.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
				toFinal.setDstStageMask(postClear.stage);
				toFinal.setDstAccessMask(postClear.access);
				toFinal.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
				toFinal.setNewLayout(postClear.layout);
				toFinal.setImage(tex->GetImage());
				toFinal.setSubresourceRange(range);
				vk::DependencyInfo postClearDep{};
				postClearDep.setImageMemoryBarriers(toFinal);
				cmd.pipelineBarrier2(postClearDep);

				tex->SetCurrentLayout(postClear.layout);
				tex->SetLastStageAccess(postClear.stage, postClear.access);
				tex->SetHasDefinedContents(true);
			}
		}

		for (std::size_t i = 0; i < recipes.size(); ++i) {
			if (Graph* inner = graph.InnerGraphOfNode(i)) {
				ZeroInitializeUndefinedReads(cmd, registry, *inner);
			}
		}
	}

	class PhysicalExecutionBackend {
	public:
		PhysicalExecutionBackend(PhysicalResourceRegistry& registry): m_registry(registry) {}

		void Execute(Graph& graph, const FrameContext& ctx, CommandBuffer& cmd, bool enableAliasing = true) {
			// 1. Setup recipes for the frame. Recursive: Subgraph::Setup (Frame.hpp) compiles
			// its own inner graph as part of this call, so every nesting level's Schedule/
			// Recipes are already current by the time step 2 below even runs.
			graph.Setup(ctx);

			// 2. Compile: levels nodes into dependency-respecting stages and synthesizes the
			// (resource, access, domain) metadata -- including cross-domain transfers -- that
			// sits at each stage boundary. Layouts/stages/access masks are deliberately not this
			// layer's job; see ResourceState.hpp's header comment for why.
			if (auto compileRes = graph.Compile(); !compileRes) {
				throw std::runtime_error("Graph compilation failed: " + compileRes.error().message);
			}

			// 3. Provision physical resources matching concrete recipes, at every nesting level
			// -- see ProvisionRecursive's comment for why aliasing is forced off below the top
			// level.
			ProvisionRecursive(graph, ctx.frameIndex, enableAliasing);

			// 4. The graph's opaque CommandBuffer handle, viewed as real Vulkan at this boundary
			// only -- Execution.hpp's CommandBuffer stays untouched and Vulkan-free.
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			// 4a. One NodeContext, reused across every node this frame (at every nesting level).
			// pipeline/pipelineLayout/globalUboOffset stay at their defaults -- each ported node
			// still resolves its own pipeline through its own PipelineLibrary pointer (the
			// GradientNode/DeferredNode pattern), not through ctx. globalSet is the one
			// genuinely shared, backend-level handle: the single bindless descriptor set every
			// ported node binds at set 0. m_registry itself satisfies BindlessIndexSource, so
			// ctx.Index<K>() already resolves real indices too.
			NodeContext nodeCtx{};
			nodeCtx.cmd = cmd;
			nodeCtx.width = ctx.width;
			nodeCtx.height = ctx.height;
			nodeCtx.frameIndex = ctx.frameIndex;
			nodeCtx.bindless = &m_registry;
			nodeCtx.globalSet = static_cast<void*>(static_cast<VkDescriptorSet>(m_registry.GetBindlessDescriptorSet()));
			nodeCtx.globalSetLayout = static_cast<void*>(
				static_cast<VkDescriptorSetLayout>(m_registry.GetBindlessDescriptorSetLayout())
			);

			// 4b. Any resource a node is about to Read but that has never actually been written
			// (a History<K> pair's first frame, most notably) gets a one-time zero-fill here,
			// before any node's barriers or commands are recorded -- recurses into every nested
			// Subgraph itself.
			ZeroInitializeUndefinedReads(vkCmd, m_registry, graph);

			// 5. Staged execution, recursing into any Subgraph node with this exact same
			// treatment rather than the naive per-node loop Graph::Execute/Subgraph::Execute
			// give it directly -- see RunSchedule.
			RunSchedule(graph, nodeCtx, vkCmd);
		}

	private:
		// Provisions this graph's own schedule/recipes, then recurses into every Subgraph
		// node's inner graph and does the same. All of it lands in m_registry, which is keyed
		// by ResourceId regardless of which Graph instance a call came from -- so a resource a
		// Subgraph produces is just as visible to an outer sibling reading it afterward as one
		// the outer graph produced directly.
		//
		// enableAliasing is forced off below the top level: ImageAliasPool's lifetime math is
		// keyed purely on stage index within *one* Schedule, so mixing an inner graph's
		// independently-numbered stages into the same pool as the outer graph's could alias two
		// resources that are actually live at the same time. Nothing exercises Subgraph-nested
		// aliasing today, so this is a safe default rather than a regression -- revisit only if
		// a real workload wants transient aliasing inside a Subgraph.
		void ProvisionRecursive(Graph& graph, std::uint64_t frameIndex, bool enableAliasing) {
			m_registry.Provision(graph.GetSchedule(), graph.Recipes(), frameIndex, enableAliasing);
			for (std::size_t i = 0; i < graph.Recipes().size(); ++i) {
				if (Graph* inner = graph.InnerGraphOfNode(i)) {
					ProvisionRecursive(*inner, frameIndex, /*enableAliasing=*/false);
				}
			}
		}

		// The staged-execution loop Execute() used to run inline, now shared between the
		// top-level call and every level of Subgraph recursion so a nested cluster gets the
		// exact same correctness guarantees (per-stage Acquire/Release barriers, dynamic-
		// rendering Begin/End around each node) as the top-level graph -- rather than the
		// zero-synchronization loop Graph::Execute/Subgraph::Execute give it when called
		// directly (still correct for a pure Vulkan-free test with no real barriers to issue,
		// just not what a real render needs).
		void RunSchedule(Graph& graph, NodeContext& nodeCtx, vk::CommandBuffer vkCmd) {
			const Schedule& schedule = graph.GetSchedule();
			const auto      recipes = graph.Recipes();

			// Nodes within a stage are provably independent of each other (ScheduleStage's
			// documented invariant), so per-node barriers between them would be wrong even if
			// they happened to be correct today -- everything a stage's nodes need transitioned
			// is therefore synthesized once, up front, into one local batch.
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

					// A Subgraph node is provably independent of everything else sharing its
					// stage (that's what put it there), so recursing here -- rather than calling
					// ExecuteNode, which would dispatch to Subgraph::Execute's naive loop --
					// is what gives its own inner nodes real barriers and dynamic-rendering.
					// This node's own realizations are empty (Subgraph::Setup returns a bare
					// Recipe -- its Resources are declared at the type level, not per-frame), so
					// there's nothing else this loop iteration would have done for it anyway.
					if (Graph* inner = graph.InnerGraphOfNode(nodeIndex)) {
						RunSchedule(*inner, nodeCtx, vkCmd);
						continue;
					}

					bool activeRendering = DynamicRenderingWrapper::Begin(vkCmd, m_registry, recipe);
					graph.ExecuteNode(nodeIndex, nodeCtx);
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

		PhysicalResourceRegistry& m_registry;
	};

} // namespace brassica::graph
