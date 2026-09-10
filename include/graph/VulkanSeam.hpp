#pragma once

#if __has_include(<vulkan/vulkan.h>)
	#include <vulkan/vulkan.h>
#elif defined(BRASSICA_HAS_VULKAN_HEADERS) && __has_include("glad/vulkan.h")
	#include "glad/vulkan.h"
	#ifndef PFN_vkGetBufferMemoryRequirements2KHR
		typedef PFN_vkGetBufferMemoryRequirements2 PFN_vkGetBufferMemoryRequirements2KHR;
		typedef PFN_vkGetImageMemoryRequirements2 PFN_vkGetImageMemoryRequirements2KHR;
		typedef PFN_vkBindBufferMemory2 PFN_vkBindBufferMemory2KHR;
		typedef PFN_vkBindImageMemory2 PFN_vkBindImageMemory2KHR;
		typedef PFN_vkGetPhysicalDeviceMemoryProperties2 PFN_vkGetPhysicalDeviceMemoryProperties2KHR;
		typedef PFN_vkGetDeviceBufferMemoryRequirements PFN_vkGetDeviceBufferMemoryRequirementsKHR;
		typedef PFN_vkGetDeviceImageMemoryRequirements PFN_vkGetDeviceImageMemoryRequirementsKHR;
		typedef PFN_vkGetPhysicalDeviceProperties2 PFN_vkGetPhysicalDeviceProperties2KHR;
	#endif
#else
	// Stubs for header-only test compilation when Vulkan headers/libs are not present
	typedef void* VkDevice;
	typedef void* VkCommandBuffer;
	typedef void* VkImage;
	typedef void* VkImageView;
	typedef void* VkBuffer;
	typedef void* VkSampler;
	typedef void* VkDescriptorSet;
	typedef uint32_t VkFormat;
	#define VK_NULL_HANDLE nullptr
#endif

#if __has_include("vk_mem_alloc.h")
	#include "vk_mem_alloc.h"
	#ifndef VMA_ALLOCATION_CREATE_ALIASED_BIT
		#ifdef VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT
			#define VMA_ALLOCATION_CREATE_ALIASED_BIT VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT
		#else
			#define VMA_ALLOCATION_CREATE_ALIASED_BIT 0x00000200
		#endif
	#endif
#else
	typedef void* VmaAllocator;
	typedef void* VmaAllocation;
	#define VMA_ALLOCATION_CREATE_ALIASED_BIT 0x00000200
	#define VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT 0x00000200
#endif
