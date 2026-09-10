#pragma once

#include <vector>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	class BarrierTranslator {
	public:
		// Translates abstract MemoryBarrier items from a BarrierBatch into concrete
		// VkImageMemoryBarrier2 / VkBufferMemoryBarrier2 and issues vkCmdPipelineBarrier2.
		static void TranslateAndDispatch(
			VkCommandBuffer                 cmd,
			const PhysicalResourceRegistry& registry,
			const BarrierBatch&             batch
		) {
			if (batch.Empty() || cmd == VK_NULL_HANDLE) {
				return;
			}

#if __has_include(<vulkan/vulkan.h>) || __has_include("glad/vulkan.h")
			std::vector<VkImageMemoryBarrier2>  imageBarriers;
			std::vector<VkBufferMemoryBarrier2> bufferBarriers;

			for (const auto& mb : batch.Items()) {
				const auto* tex = registry.GetTexture(mb.resource);
				if (tex && tex->image != VK_NULL_HANDLE) {
					VkImageMemoryBarrier2 imb{};
					imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
					imb.srcStageMask = mb.srcStage ? static_cast<VkPipelineStageFlags2>(mb.srcStage)
												   : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
					imb.dstStageMask = mb.dstStage ? static_cast<VkPipelineStageFlags2>(mb.dstStage)
												   : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
					imb.srcAccessMask = static_cast<VkAccessFlags2>(mb.srcAccess);
					imb.dstAccessMask = static_cast<VkAccessFlags2>(mb.dstAccess);
					imb.oldLayout = static_cast<VkImageLayout>(mb.oldLayout);
					imb.newLayout = static_cast<VkImageLayout>(mb.newLayout);
					imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imb.image = tex->image;

					VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
					VkFormat format = static_cast<VkFormat>(tex->desc.formatCode);
					if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM) {
						aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
					}

					imb.subresourceRange.aspectMask = aspect;
					imb.subresourceRange.baseMipLevel = 0;
					imb.subresourceRange.levelCount = tex->desc.mips ? tex->desc.mips : 1;
					imb.subresourceRange.baseArrayLayer = 0;
					imb.subresourceRange.layerCount = tex->desc.layers ? tex->desc.layers : 1;

					imageBarriers.push_back(imb);
					continue;
				}

				const auto* buf = registry.GetBuffer(mb.resource);
				if (buf && buf->buffer != VK_NULL_HANDLE) {
					VkBufferMemoryBarrier2 bmb{};
					bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
					bmb.srcStageMask = mb.srcStage ? static_cast<VkPipelineStageFlags2>(mb.srcStage)
												   : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
					bmb.dstStageMask = mb.dstStage ? static_cast<VkPipelineStageFlags2>(mb.dstStage)
												   : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
					bmb.srcAccessMask = static_cast<VkAccessFlags2>(mb.srcAccess);
					bmb.dstAccessMask = static_cast<VkAccessFlags2>(mb.dstAccess);
					bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bmb.buffer = buf->buffer;
					bmb.offset = 0;
					bmb.size = buf->desc.byteSize ? buf->desc.byteSize : VK_WHOLE_SIZE;

					bufferBarriers.push_back(bmb);
				}
			}

			if (imageBarriers.empty() && bufferBarriers.empty()) {
				return;
			}

#if defined(VK_VERSION_1_3) || defined(VK_KHR_dynamic_rendering)
			VkDependencyInfo depInfo{};
			depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			depInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
			depInfo.pImageMemoryBarriers = imageBarriers.data();
			depInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
			depInfo.pBufferMemoryBarriers = bufferBarriers.data();

			// Function pointer resolution if needed or standard call
			PFN_vkCmdPipelineBarrier2 pfnCmdPipelineBarrier2 =
				reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(nullptr, "vkCmdPipelineBarrier2"));
			if (pfnCmdPipelineBarrier2) {
				pfnCmdPipelineBarrier2(cmd, &depInfo);
			}
#endif
#else
			(void)registry;
#endif
		}
	};

} // namespace brassica::graph
