#include <memory>
#include <span>
#include <cstring>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include "graph/PhysicalResource.hpp"
#include "graph/PhysicalRegistry.hpp"

namespace brassica::utils {

    template <graph::ResourceRef Key, typename T>
    std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
        vk::Device device,
        VmaAllocator allocator,
        vk::Queue transferQueue,
        vk::CommandPool transientPool,
        graph::PhysicalResourceRegistry& registry,
        std::span<const T> data,
        const graph::ResourceDesc& desc
    ) {
        const vk::DeviceSize bufferSize = data.size_bytes();

        // 1. Create the RAII destination buffer using your graph's wrapper
        auto destBuffer = std::make_shared<graph::PhysicalBuffer>(device, allocator, desc); //

        // 2. Allocate a temporary host-visible staging buffer directly via VMA
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer rawStagingBuffer;
        VmaAllocation stagingAllocation;
        VmaAllocationInfo vmaAllocInfo;
        vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &rawStagingBuffer, &stagingAllocation, &vmaAllocInfo);

        // 3. Copy CPU data to the mapped staging memory
        std::memcpy(vmaAllocInfo.pMappedData, data.data(), bufferSize);

        // 4. Record and submit the transfer command synchronously
        vk::CommandBufferAllocateInfo cmdAllocInfo{transientPool, vk::CommandBufferLevel::ePrimary, 1};
        vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAllocInfo).front();

        cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy copyRegion{0, 0, bufferSize};
        cmd.copyBuffer(rawStagingBuffer, destBuffer->GetBuffer(), 1, &copyRegion); //
        cmd.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(cmd);
        transferQueue.submit(1, &submitInfo, nullptr);
        transferQueue.waitIdle();

        // 5. Cleanup transient resources
        device.freeCommandBuffers(transientPool, 1, &cmd);
        vmaDestroyBuffer(allocator, rawStagingBuffer, stagingAllocation);

        // 6. Inject the populated handle into the graph registry
        registry.RegisterImportedBuffer<Key>(
            destBuffer->GetBuffer(), //[cite: 7]
            desc,
            true // hasDefinedContents is true so the graph preserves data
        );

        return destBuffer;
    }
}