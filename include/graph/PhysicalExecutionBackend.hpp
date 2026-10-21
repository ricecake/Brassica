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

#ifdef MemoryBarrier
	#undef MemoryBarrier
#endif

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

			// From the real provisioned texture, not r.desc: a builder node's realization
			// (SetupBuilder.hpp) carries an empty desc by construction for any catalog-converted
			// key (ResourceCatalog.hpp) -- Provision()'s catalog-desc override only ever updates
			// a local copy for its own realizationsToProvision map, it never writes back into
			// this Recipe's own stored realizations. format/usage just above already read
			// through tex->GetDesc() for the same reason; width/height hadn't, which this closes.
			// Confirmed as a real bug, not a hypothetical: with r.desc.width/height silently
			// staying 0 for every attachment on a fully-converted node, DeriveAttachments
			// reported a 0x0 render size, and Begin's `derived.width ? ... : 1` fallback below
			// collapsed the real render area down to a single pixel -- no validation error
			// (1x1 fits inside any real image), but only that one pixel's worth of a frame's
			// geometry/shading ever actually got drawn.
			out.width = std::max(out.width, tex->GetDesc().width);
			out.height = std::max(out.height, tex->GetDesc().height);

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

	// Shared clear-barrier-clear-barrier sequence for a single undefined image, called from both
	// ZeroInitializeUndefinedReads (History<K>'s first-use case) and
	// ZeroInitializeInactiveReads (§E's "no active writer this frame" case, S5) below -- same
	// Vulkan sequence either way, just a different reason the image's contents aren't defined.
	inline void ClearImageToZero(
		vk::CommandBuffer cmd,
		PhysicalTexture&  tex,
		AccessKind        access,
		ExecutionDomain   domain
	) {
		const auto& desc = tex.GetDesc();

		// Defensive: vkCmdClearColorImage/clearDepthStencilImage both require the image to have
		// been created with eTransferDst -- a real hardware test found exactly this (GTAOTexture
		// had no reason to need it before ZeroInitializeInactiveReads/S5 started targeting it,
		// so its own Desc() never declared it, and the clear hit a real "image not compatible"
		// validation warning). That specific key is fixed at the source (ResourceKeys.hpp), but
		// skip rather than emit an invalid call for any other key with the same gap, so this
		// degrades to "reads stale/undefined content" (the pre-S5 behavior) instead of a
		// validation error.
		if (!(desc.usageMask & static_cast<std::uint32_t>(vk::ImageUsageFlagBits::eTransferDst))) {
			static bool warned = false;
			if (!warned) {
				spdlog::warn(
					"ClearImageToZero: skipping zero-init clear for a texture whose desc lacks "
					"eTransferDst usage -- add it to this key's Desc() to fix (see "
					"GTAOTextureDesc for the pattern)."
				);
				warned = true;
			}
			return;
		}

		const auto          format = static_cast<vk::Format>(desc.formatCode);
		const bool          isDepth = IsDepthFormat(format);
		const std::uint32_t mips = desc.mips ? desc.mips : 1;
		const std::uint32_t layers = desc.layers ? desc.layers : 1;
		const auto          range = vk::ImageSubresourceRange(AspectFor(format), 0, mips, 0, layers);

		// Src comes from tracked state (ClearBufferToZero's pattern below), not eTopOfPipe --
		// eTopOfPipe chains with nothing, so a prior read of this resource (e.g. a reader that
		// ran before this frame decided the key needed zero-init) could still be in flight when
		// the clear's layout transition starts.
		vk::ImageMemoryBarrier2 toTransferDst{};
		toTransferDst.setSrcStageMask(tex.GetLastStage() ? tex.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe);
		toTransferDst.setSrcAccessMask(tex.GetLastAccess());
		toTransferDst.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
		toTransferDst.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite);
		toTransferDst.setOldLayout(tex.GetCurrentLayout());
		toTransferDst.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
		toTransferDst.setImage(tex.GetImage());
		toTransferDst.setSubresourceRange(range);
		vk::DependencyInfo preClearDep{};
		preClearDep.setImageMemoryBarriers(toTransferDst);
		cmd.pipelineBarrier2(preClearDep);

		if (isDepth) {
			cmd.clearDepthStencilImage(
				tex.GetImage(),
				vk::ImageLayout::eTransferDstOptimal,
				vk::ClearDepthStencilValue{1.0f, 0},
				range
			);
		} else {
			cmd.clearColorImage(tex.GetImage(), vk::ImageLayout::eTransferDstOptimal, ZeroClearValueFor(format), range);
		}

		// Land back in whatever layout this realization's own access derives -- the same rule
		// Provision/BarrierTranslator use, so the very next Acquire barrier this resource sees
		// (this stage's own, synthesized right after this function returns) starts from a state
		// it actually agrees with.
		const ResourceState postClear =
			DeriveImageState(access, domain, desc.usageMask ? vk::ImageUsageFlags(desc.usageMask) : vk::ImageUsageFlags{}, format);

		vk::ImageMemoryBarrier2 toFinal{};
		toFinal.setSrcStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
		toFinal.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
		toFinal.setDstStageMask(postClear.stage);
		toFinal.setDstAccessMask(postClear.access);
		toFinal.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
		toFinal.setNewLayout(postClear.layout);
		toFinal.setImage(tex.GetImage());
		toFinal.setSubresourceRange(range);
		vk::DependencyInfo postClearDep{};
		postClearDep.setImageMemoryBarriers(toFinal);
		cmd.pipelineBarrier2(postClearDep);

		tex.SetCurrentLayout(postClear.layout);
		tex.SetLastStageAccess(postClear.stage, postClear.access);
		tex.SetHasDefinedContents(true);
	}

	// Buffer counterpart of ClearImageToZero -- same shape, minus a layout (buffers have none,
	// see PhysicalBuffer's own accessors). vkCmdFillBuffer(..., 0) is the buffer equivalent of
	// clearColorImage: a flat zero fill, not a format-aware clear value, since a buffer has no
	// format to be depth-vs-color about.
	inline void ClearBufferToZero(
		vk::CommandBuffer cmd,
		PhysicalBuffer&   buf,
		AccessKind        access,
		ExecutionDomain   domain
	) {
		const auto& desc = buf.GetDesc();

		// Defensive: vkCmdFillBuffer requires eTransferDst -- see ClearImageToZero's identical
		// guard and comment for why this degrades to a skip instead of an invalid call.
		if (!(desc.usageMask & static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eTransferDst))) {
			static bool warned = false;
			if (!warned) {
				spdlog::warn(
					"ClearBufferToZero: skipping zero-init clear for a buffer whose desc lacks "
					"eTransferDst usage -- add it to this key's Desc()."
				);
				warned = true;
			}
			return;
		}

		const vk::DeviceSize size = desc.byteSize ? desc.byteSize : VK_WHOLE_SIZE;

		vk::BufferMemoryBarrier2 toTransferDst{
			buf.GetLastStage() ? buf.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe,
			buf.GetLastAccess(),
			vk::PipelineStageFlagBits2::eAllTransfer,
			vk::AccessFlagBits2::eTransferWrite,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			buf.GetBuffer(),
			0,
			size,
		};
		vk::DependencyInfo preClearDep{};
		preClearDep.setBufferMemoryBarriers(toTransferDst);
		cmd.pipelineBarrier2(preClearDep);

		cmd.fillBuffer(buf.GetBuffer(), 0, size, 0);

		const ResourceState postClear =
			DeriveBufferState(access, domain, desc.usageMask ? vk::BufferUsageFlags(desc.usageMask) : vk::BufferUsageFlags{});

		vk::BufferMemoryBarrier2 toFinal{
			vk::PipelineStageFlagBits2::eAllTransfer,
			vk::AccessFlagBits2::eTransferWrite,
			postClear.stage,
			postClear.access,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			buf.GetBuffer(),
			0,
			size,
		};
		vk::DependencyInfo postClearDep{};
		postClearDep.setBufferMemoryBarriers(toFinal);
		cmd.pipelineBarrier2(postClearDep);

		buf.SetLastStageAccess(postClear.stage, postClear.access);
		buf.SetHasDefinedContents(true);
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

				ClearImageToZero(cmd, *tex, r.access, recipe.domain);
			}
		}
	}

	// Generalization of the same idea for §E's zero-init list (S5): Graph::ZeroInitReads()
	// already computed exactly which (node, key) Read accesses have no active writer preceding
	// them this frame -- any resource kind, not just History images (the live case this closes:
	// GTAO disabled, DeferredNode's plain Read<GTAOTexture>() would otherwise see whatever
	// garbage/stale content the texture physically has). Looks up each flagged read's own
	// realization (for its access/desc, the same ones the compiler itself used) rather than
	// re-deriving anything.
	//
	// Deliberately does NOT check HasDefinedContents() the way ZeroInitializeUndefinedReads
	// does -- that flag is right for isHistory's one-time "has this slot EVER been written"
	// question, but wrong here: ZeroInitReads() is recomputed fresh every Compile() from this
	// frame's actual activity, which can toggle (a feature can turn on, write real data, then
	// turn back off next frame) -- a sticky flag would clear it once, then skip re-clearing
	// forever even after a later active frame leaves real (non-zero) data behind. Always
	// re-clear when the compiler says this frame needs it; ClearImageToZero/ClearBufferToZero
	// still set the flag true as a side effect, for whatever else (just ZeroInitializeUndefinedReads
	// today) legitimately reads it.
	inline void ZeroInitializeInactiveReads(vk::CommandBuffer cmd, PhysicalResourceRegistry& registry, Graph& graph) {
		if (!cmd) {
			return;
		}

		const auto recipes = graph.Recipes();
		for (const auto& [nodeIndex, key] : graph.ZeroInitReads()) {
			if (nodeIndex >= recipes.size()) {
				continue;
			}
			const Recipe&               recipe = recipes[nodeIndex];
			const ResourceRealization*  realization = nullptr;
			for (const auto& r : recipe.realizations) {
				if (r.key == key) {
					realization = &r;
					break;
				}
			}
			if (!realization) {
				continue;
			}

			if (realization->desc.kind == ResourceDesc::Kind::Buffer) {
				auto buf = registry.GetBuffer(key);
				if (!buf) {
					continue;
				}
				ClearBufferToZero(cmd, *buf, realization->access, recipe.domain);
				continue;
			}

			if (realization->desc.kind != ResourceDesc::Kind::Image2D &&
			    realization->desc.kind != ResourceDesc::Kind::Image3D) {
				continue; // AccelerationStructure etc. -- no zero-init concept
			}

			auto tex = registry.GetTexture(key);
			if (!tex) {
				continue;
			}
			ClearImageToZero(cmd, *tex, realization->access, recipe.domain);
		}
	}

	class PhysicalExecutionBackend {
	public:
		PhysicalExecutionBackend(PhysicalResourceRegistry& registry): m_registry(registry) {}

		void SetQueueSet(const QueueSet& queues) { m_queues = queues; }

		const QueueSet& GetQueueSet() const { return m_queues; }

		void Execute(Graph& graph, const FrameContext& ctx, CommandBuffer& cmd, bool enableAliasing = true) {
			// 1. Setup recipes for the frame.
			graph.Setup(ctx);

			// 2. Compile: levels nodes into dependency-respecting stages and synthesizes the
			// (resource, access, domain) metadata -- including cross-domain transfers -- that
			// sits at each stage boundary. Layouts/stages/access masks are deliberately not this
			// layer's job; see ResourceState.hpp's header comment for why.
			if (auto compileRes = graph.Compile(); !compileRes) {
				throw std::runtime_error("Graph compilation failed: " + compileRes.error().message);
			}

			// 3. Provision physical resources matching concrete recipes.
			m_registry.Provision(graph.GetSchedule(), graph.Recipes(), ctx.frameIndex, enableAliasing, graph.Nodes(), ctx);

			// 4. The graph's opaque CommandBuffer handle, viewed as real Vulkan at this boundary
			// only -- Execution.hpp's CommandBuffer stays untouched and Vulkan-free.
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			// 4a. One NodeContext, reused across every node this frame (at every nesting level).
			// pipeline/pipelineLayout stay at their defaults -- each ported node still resolves
			// its own pipeline through its own PipelineLibrary pointer (the GradientNode/
			// DeferredNode pattern), not through ctx. frameSet/globalSet are the two genuinely
			// shared, backend-level handles every ported node binds: the always-bound per-frame
			// UBO set (set 0) and the single bindless descriptor set (set 1). m_registry itself
			// satisfies ResourceServices, so ctx.Index<K>()/Write<K>() already resolve/write
			// through it too.
			NodeContext nodeCtx{};
			nodeCtx.cmd = cmd;
			nodeCtx.width = ctx.width;
			nodeCtx.height = ctx.height;
			nodeCtx.frameIndex = ctx.frameIndex;
			nodeCtx.resources = &m_registry;
			nodeCtx.frameSet = static_cast<void*>(static_cast<VkDescriptorSet>(m_registry.GetFrameDescriptorSet()));
			nodeCtx.frameSetLayout = static_cast<void*>(
				static_cast<VkDescriptorSetLayout>(m_registry.GetFrameDescriptorSetLayout())
			);
			nodeCtx.globalSet = static_cast<void*>(static_cast<VkDescriptorSet>(m_registry.GetBindlessDescriptorSet()));
			nodeCtx.globalSetLayout = static_cast<void*>(
				static_cast<VkDescriptorSetLayout>(m_registry.GetBindlessDescriptorSetLayout())
			);

			// 4b. Any resource a node is about to Read but that has never actually been written
			// (a History<K> pair's first frame, most notably) gets a one-time zero-fill here,
			// before any node's barriers or commands are recorded.
			ZeroInitializeUndefinedReads(vkCmd, m_registry, graph);

			// 4c. §E's zero-init list (S5): a plain Read with no active writer this frame (e.g.
			// GTAO disabled, DeferredNode still reading GTAOTexture) -- any resource kind, not
			// just History images, and re-derived fresh every Compile() rather than a sticky
			// per-resource flag.
			ZeroInitializeInactiveReads(vkCmd, m_registry, graph);

			// 5. Staged execution -- see RunSchedule.
			RunSchedule(graph, nodeCtx, vkCmd);
		}

	private:
		// The staged-execution loop Execute() used to run inline, factored out here so the
		// per-stage Acquire/Release barrier and dynamic-rendering Begin/End logic has one home.
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

					bool activeRendering = DynamicRenderingWrapper::Begin(vkCmd, m_registry, recipe);
					nodeCtx.insideRendering = activeRendering;
					nodeCtx.domain = recipe.domain;
					nodeCtx.placedQueue = ExecutionDomain::Graphics; // the only queue this backend ever uses
					nodeCtx.declared = recipe.realizations;
					graph.ExecuteNode(nodeIndex, nodeCtx);
					if (activeRendering) {
						DynamicRenderingWrapper::End(vkCmd);
					}
					nodeCtx.insideRendering = false;

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
		QueueSet                  m_queues{};
	};

} // namespace brassica::graph
