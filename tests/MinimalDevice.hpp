#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "vk_mem_alloc.h"
#include "VkBootstrap.h"

// A minimal Vulkan 1.3 core bootstrap for tests that need a *real* device but not the full
// production Engine's feature set. Engine::InitVulkan (src/Engine.cpp) hard-requires
// VK_EXT_mesh_shader, VK_KHR_ray_query, VK_KHR_acceleration_structure, and
// VK_KHR_fragment_shading_rate -- real TerrainPass/DeferredPass dependencies -- and MoltenVK
// 1.4.2 implements none of them (confirmed directly against a real M1 Pro via `vulkaninfo`), so
// every test that bootstraps through Engine can only ever take the skip path on a Mac.
//
// Tests that only exercise the graph/physical layer with synthetic nodes (no mesh shaders, no
// ray query, no acceleration structures -- see test_physical_backend.cpp's PassA/PassB) don't
// need any of that. This gives them a real device wherever core Vulkan 1.3 (dynamic rendering +
// synchronization2) is available, MoltenVK included -- real barrier dispatch, real
// validation-layer coverage, on this machine, today.
//
// No surface, no swapchain: these tests render into offscreen images only, so there is nothing
// for a swapchain to attach to. vkb::PhysicalDeviceSelector normally expects a surface;
// defer_surface_initialization() is vk-bootstrap's documented way to opt out entirely.
namespace brassica::testing {

	class MinimalDevice {
	public:
		MinimalDevice() {
			vkb::InstanceBuilder instanceBuilder;
			instanceBuilder.set_app_name("brassica-tests")
				.require_api_version(1, 3, 0)
				.set_debug_callback(&MinimalDevice::DebugCallback)
				.set_debug_callback_user_data_pointer(this);

			auto instRes = instanceBuilder.request_validation_layers(true).build();
			if (!instRes) {
				auto fallback = instanceBuilder.request_validation_layers(false).build();
				if (!fallback) {
					spdlog::error("MinimalDevice: failed to create instance: {}", fallback.error().message());
					return;
				}
				m_vkbInstance = fallback.value();
			} else {
				m_vkbInstance = instRes.value();
			}
			m_instance = m_vkbInstance.instance;

			VkPhysicalDeviceVulkan13Features features13{};
			features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
			features13.dynamicRendering = VK_TRUE;
			features13.synchronization2 = VK_TRUE;

			// Bindless set 0 (PhysicalRegistry's bindless index machinery, Engine::InitGlobalDescriptors)
			// -- confirmed present on this Mac's MoltenVK 1.4.2 via vulkaninfo before relying on it
			// here, unlike mesh shader/ray query/AS/VRS, which MoltenVK genuinely doesn't implement
			// (see this header's class comment). Mirrors the same features Engine.cpp requests.
			VkPhysicalDeviceVulkan12Features features12{};
			features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
			features12.descriptorIndexing = VK_TRUE;
			features12.descriptorBindingPartiallyBound = VK_TRUE;
			features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
			features12.runtimeDescriptorArray = VK_TRUE;
			features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
			features12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
			features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
			features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;

			vkb::PhysicalDeviceSelector selector{m_vkbInstance};
			selector.set_minimum_version(1, 3)
				.set_required_features_13(features13)
				.set_required_features_12(features12)
				.defer_surface_initialization();

			auto physRes = selector.select();
			if (!physRes) {
				spdlog::error("MinimalDevice: no suitable physical device: {}", physRes.error().message());
				vkb::destroy_instance(m_vkbInstance);
				m_instance = nullptr;
				return;
			}
			m_physicalDevice = physRes.value().physical_device;

			vkb::DeviceBuilder deviceBuilder{physRes.value()};
			auto               devRes = deviceBuilder.build();
			if (!devRes) {
				spdlog::error("MinimalDevice: failed to create logical device: {}", devRes.error().message());
				vkb::destroy_instance(m_vkbInstance);
				m_instance = nullptr;
				return;
			}
			vkb::Device vkbDevice = devRes.value();
			m_device = vkbDevice.device;
			m_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
			m_queueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

			VmaAllocatorCreateInfo allocatorInfo{};
			allocatorInfo.physicalDevice = m_physicalDevice;
			allocatorInfo.device = m_device;
			allocatorInfo.instance = m_instance;
			allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

			if (vmaCreateAllocator(&allocatorInfo, &m_allocator) != VK_SUCCESS) {
				spdlog::error("MinimalDevice: failed to create VMA allocator.");
				m_device.destroy();
				vkb::destroy_instance(m_vkbInstance);
				m_device = nullptr;
				m_instance = nullptr;
				return;
			}
		}

		~MinimalDevice() {
			if (m_device) {
				m_device.waitIdle();
			}
			if (m_allocator) {
				vmaDestroyAllocator(m_allocator);
			}
			if (m_device) {
				m_device.destroy();
			}
			if (m_instance) {
				vkb::destroy_instance(m_vkbInstance);
			}
		}

		MinimalDevice(const MinimalDevice&) = delete;
		MinimalDevice& operator=(const MinimalDevice&) = delete;

		[[nodiscard]] bool IsValid() const { return static_cast<bool>(m_device); }

		[[nodiscard]] vk::Device GetDevice() const { return m_device; }

		[[nodiscard]] VmaAllocator GetAllocator() const { return m_allocator; }

		[[nodiscard]] vk::Queue GetQueue() const { return m_queue; }

		[[nodiscard]] uint32_t GetQueueFamily() const { return m_queueFamily; }

		[[nodiscard]] uint32_t GetValidationErrorCount() const { return m_errorCount; }

		[[nodiscard]] uint32_t GetValidationWarningCount() const { return m_warningCount; }

	private:
		static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
			const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
			void*                                       userData
		) {
			auto* self = static_cast<MinimalDevice*>(userData);
			if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
				spdlog::error("[Vulkan Validation Error] {}", callbackData->pMessage);
				if (self)
					self->m_errorCount++;
			} else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
				spdlog::warn("[Vulkan Validation Warning] {}", callbackData->pMessage);
				if (self)
					self->m_warningCount++;
			}
			return VK_FALSE;
		}

		vkb::Instance m_vkbInstance{};

		vk::Instance       m_instance{};
		vk::PhysicalDevice m_physicalDevice{};
		vk::Device         m_device{};
		vk::Queue          m_queue{};
		uint32_t           m_queueFamily{0};
		VmaAllocator       m_allocator{VK_NULL_HANDLE};

		uint32_t m_errorCount{0};
		uint32_t m_warningCount{0};
	};

} // namespace brassica::testing
