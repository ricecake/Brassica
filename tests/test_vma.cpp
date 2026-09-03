#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "vk_mem_alloc.h"
#include "vulkan/vulkan.hpp"

TEST_CASE("Vulkan Memory Allocator Structures and Types") {
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

	CHECK(allocatorInfo.vulkanApiVersion == VK_API_VERSION_1_3);

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	CHECK(allocInfo.usage == VMA_MEMORY_USAGE_AUTO);
	CHECK((allocInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0);
}
