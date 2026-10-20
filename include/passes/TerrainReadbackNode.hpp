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
			toTransferSrc.setSrcStageMask(tex->GetLastStage() ? tex->GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe);
			toTransferSrc.setSrcAccessMask(tex->GetLastAccess());
			toTransferSrc.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer);
			toTransferSrc.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
			toTransferSrc.setOldLayout(tex->GetCurrentLayout());
			toTransferSrc.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
			toTransferSrc.setImage(tex->GetImage());
			toTransferSrc.setSubresourceRange(range);
			vk::DependencyInfo dep1{};
			dep1.setImageMemoryBarriers(toTransferSrc);
			vkCmd.pipelineBarrier2(dep1);
			tex->SetCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);
			tex->SetLastStageAccess(vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead);

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

			terrainManager->MarkReadbackSubmitted(ctx.frameIndex);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainReadbackNode);

} // namespace brassica
