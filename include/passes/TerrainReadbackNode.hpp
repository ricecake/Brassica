#pragma once

#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "ServiceLocator.hpp"
#include "terrain/TerrainManager.hpp"

namespace brassica {

	// Replaces TerrainManager's old side-channel transfer-queue submit (its own command
	// buffer/semaphore, racing the graph's own tracked state with no synchronization at all --
	// design plan's bug 6) with a real node: the region copy now goes through the same barrier-
	// synthesized command buffer as everything else this frame, and the CPU-visible result is
	// picked up by frame retirement (TerrainManager::PollReadbackData) instead of a dedicated
	// semaphore. Transfer domain: under S4's queue placement this becomes real transfer-queue
	// work; today (GraphicsOnly) it still executes on the same graphics queue as everything
	// else, just correctly tracked instead of unsynchronized.
	struct TerrainReadbackNode: render::NodeRegistrar<TerrainReadbackNode> {
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		TerrainManager*                  terrainManager = nullptr;
		TerrainReadbackRegion            planned{};
		bool                             active = false;

		void Init(const render::NodeServices& services) { physicalRegistry = services.physicalRegistry; }

		// No Vulkan resources of its own to tear down -- the staging buffer and command buffer
		// it uses both belong to TerrainManager/the graph's own backend, not this node.
		void Destroy(vk::Device) {}

		// Active only once TerrainManager actually wants a readback *and* the source texture
		// has real content to copy -- mirrors the HasDefinedContents()/currentLayout check
		// Engine::UpdateCamera used to make inline before calling TriggerImageRegionReadbackAsync.
		// Read<TerrainClipmapTexture>()/Modify<TerrainReadbackStaging>() are declared
		// unconditionally regardless (same convention HostWriteNode uses, Node.hpp) -- Active()
		// alone gates whether this node schedules at all; Graph::Compile contracts an inactive
		// node's realizations out of the schedule entirely either way.
		//
		// terrainManager is looked up here, not cached in Init(): every serviceLocator.Provide<>()
		// call (Engine::InitVulkan) happens *after* EngineNodeRegistry::InitAll() runs, so a
		// ServiceLocator lookup in Init() would always find nothing -- GTAONode's identical
		// ServiceLocator-in-Setup() pattern is the only safe place for this, not a stylistic
		// choice.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Transfer);

			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<TerrainManager>()) {
				terrainManager = ServiceLocator::Instance().Get<TerrainManager>().get();
			}

			active = false;
			if (terrainManager && terrainManager->HasPendingReadback() && physicalRegistry) {
				auto tex = physicalRegistry->GetTexture<TerrainClipmapTexture>();
				if (tex && tex->HasDefinedContents()) {
					planned = terrainManager->GetPendingReadback();
					active = true;
				}
			}
			r.Active(active);

			r.Read<TerrainClipmapTexture>();
			r.Modify<TerrainReadbackStaging>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto tex = physicalRegistry->GetTexture<TerrainClipmapTexture>();
			auto buf = physicalRegistry->GetBuffer<TerrainReadbackStaging>();
			if (!tex || !buf) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, planned.mipLevel, 1, planned.arrayLayer, 1};

			// Captured before this node's own hand-rolled transition touches tracked state:
			// PhysicalTexture has exactly one (layout, stage, access) triple for the *whole*
			// image, no per-subresource granularity at all -- the Acquire barrier already run
			// for this stage's own Read<TerrainClipmapTexture>() declaration transitioned every
			// mip and layer of the clipmap array to this same state, not just the one this node
			// is about to touch. These are what every *other* subresource is still actually in,
			// and (see below) what this one gets transitioned back to afterward.
			const vk::ImageLayout         priorLayout = tex->GetCurrentLayout();
			const vk::PipelineStageFlags2 priorStage = tex->GetLastStage();
			const vk::AccessFlags2        priorAccess = tex->GetLastAccess();

			// The Acquire barrier already run for this stage's Read<TerrainClipmapTexture>()
			// declaration landed the image wherever the generic (usage, domain) derivation
			// table puts a Transfer-domain Read of eSampled|eStorage|eTransferSrc usage --
			// eGeneral layout, storage/sampled-read access (Storage is checked before Sampled,
			// ResourceState.hpp), since this image's usage has no transfer-specific read shape
			// of its own. That's not what a real vkCmdCopyImageToBuffer needs -- it only ever
			// performs a TRANSFER_READ access -- so this transitions the rest of the way by
			// hand, same as TriggerImageRegionReadbackAsync used to. The one piece of this node
			// the generic table can't get right on its own: a graph image whose real consumer
			// here is a different access shape than its own declared usage implies.
			vk::ImageMemoryBarrier2 toTransferSrc{};
			toTransferSrc.setSrcStageMask(priorStage ? priorStage : vk::PipelineStageFlagBits2::eTopOfPipe);
			toTransferSrc.setSrcAccessMask(priorAccess);
			toTransferSrc.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
			toTransferSrc.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
			toTransferSrc.setOldLayout(priorLayout);
			toTransferSrc.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
			toTransferSrc.setImage(tex->GetImage());
			toTransferSrc.setSubresourceRange(range);
			vk::DependencyInfo dep1{};
			dep1.setImageMemoryBarriers(toTransferSrc);
			vkCmd.pipelineBarrier2(dep1);

			vk::BufferImageCopy copyRegion{};
			copyRegion.setBufferOffset(0);
			copyRegion.setBufferRowLength(planned.extent.width);
			copyRegion.setBufferImageHeight(planned.extent.height);
			copyRegion.setImageSubresource(
				vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, planned.mipLevel, planned.arrayLayer, 1}
			);
			copyRegion.setImageOffset(vk::Offset3D{planned.offset.x, planned.offset.y, 0});
			copyRegion.setImageExtent(vk::Extent3D{planned.extent.width, planned.extent.height, 1});
			vkCmd.copyImageToBuffer(tex->GetImage(), vk::ImageLayout::eTransferSrcOptimal, buf->GetBuffer(), copyRegion);

			// Transition this one subresource back to the layout every other subresource is
			// still actually in -- otherwise PhysicalTexture's tracked (whole-image) state would
			// go on claiming the *entire* clipmap array is in eTransferSrcOptimal, when only
			// this one mip+layer ever really was. Confirmed on real hardware: a later sampled
			// read's bindless descriptor got (re)written against that lie, then failed
			// validation on every *other* layer, which were actually still in eGeneral the
			// whole time.
			vk::ImageMemoryBarrier2 backToPrior{};
			backToPrior.setSrcStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
			backToPrior.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead);
			backToPrior.setDstStageMask(priorStage ? priorStage : vk::PipelineStageFlagBits2::eTopOfPipe);
			backToPrior.setDstAccessMask(priorAccess);
			backToPrior.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
			backToPrior.setNewLayout(priorLayout);
			backToPrior.setImage(tex->GetImage());
			backToPrior.setSubresourceRange(range);
			vk::DependencyInfo dep1b{};
			dep1b.setImageMemoryBarriers(backToPrior);
			vkCmd.pipelineBarrier2(dep1b);

			// Stage/access are OR'd (conservative, never wrong) rather than overwritten: the
			// touched layer really was read via transfer, but every other layer's own last
			// stage/access (whatever priorStage/priorAccess already captured) is still true too,
			// and tracked state can't represent the two separately.
			tex->SetCurrentLayout(priorLayout);
			tex->SetLastStageAccess(
				priorStage | vk::PipelineStageFlagBits2::eAllTransfer,
				priorAccess | vk::AccessFlagBits2::eTransferRead
			);

			// The CPU is the real consumer of this buffer, entirely outside the graph -- there
			// is no Read<> node for the Acquire machinery to synthesize an eHost-destined
			// barrier from, so this is the second (and last) hand-rolled piece: makes the
			// copy's write visible to TerrainManager::PollReadbackData's eventual
			// vmaInvalidateAllocation + memcpy.
			vk::BufferMemoryBarrier2 toHostRead{
				vk::PipelineStageFlagBits2::eAllTransfer,
				vk::AccessFlagBits2::eTransferWrite,
				vk::PipelineStageFlagBits2::eHost,
				vk::AccessFlagBits2::eHostRead,
				VK_QUEUE_FAMILY_IGNORED,
				VK_QUEUE_FAMILY_IGNORED,
				buf->GetBuffer(),
				0,
				VK_WHOLE_SIZE,
			};
			vk::DependencyInfo dep2{};
			dep2.setBufferMemoryBarriers(toHostRead);
			vkCmd.pipelineBarrier2(dep2);
			buf->SetLastStageAccess(vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);

			terrainManager->MarkReadbackSubmitted(ctx.placedQueue);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainReadbackNode);

} // namespace brassica
