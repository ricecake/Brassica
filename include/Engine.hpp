#pragma once

#include <memory>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "GLFW/glfw3.h"
#include <random>

#include "passes/GradientPass.hpp"
#include "passes/MeshCubePass.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "ShaderWatcher.hpp"
#include "TaskScheduler.h"
#include "VkBootstrap.h"

namespace brassica {

	// Double-buffering data to prevent CPU/GPU stalling
	struct FrameData {
		vk::CommandPool   commandPool;
		vk::CommandBuffer commandBuffer;
		vk::Semaphore     swapchainSemaphore; // Signaled when image is acquired
		vk::Semaphore     renderSemaphore;    // Signaled when rendering finishes
		vk::Fence         renderFence;        // Signaled when GPU finishes frame execution
	};

	class Engine {
	public:
		void Init();
		void Run();
		void Cleanup();

		vk::Device GetDevice() const { return device; }

		vk::Extent2D GetSwapchainExtent() const {
			return vk::Extent2D{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		}

		vk::Format GetSwapchainFormat() const { return static_cast<vk::Format>(vkbSwapchain.image_format); }

		ShaderWatcher& GetShaderWatcher() { return shaderWatcher; }

	private:
		void InitWindow();
		bool InitVulkan();
		void InitSwapchain();
		void InitCommands();
		void InitSyncStructures();

		void DrawFrame();

		GLFWwindow*                   window{nullptr};
		uint32_t                      frameNumber{0};
		static constexpr unsigned int FRAME_OVERLAP = 2;
		FrameData                     frames[FRAME_OVERLAP];

		// Vulkan Core
		vkb::Instance              vkbInst;
		vkb::Device                vkbDevice;
		vk::Instance               instance;
		vk::PhysicalDevice         chosenGPU;
		vk::Device                 device;
		vk::SurfaceKHR             surface;
		vk::Queue                  graphicsQueue;
		uint32_t                   graphicsQueueFamily{0};
		vkb::Swapchain             vkbSwapchain;
		std::vector<vk::Image>     swapchainImages;
		std::vector<vk::ImageView> swapchainImageViews;

		std::unique_ptr<GradientPass> gradientPass;
		std::unique_ptr<MeshCubePass> meshCubePass;

		uint32_t      globalSeed{0};
		std::mt19937  rng;

		// Global Descriptor Set 0 (FrameUBO)
		vk::DescriptorSetLayout globalSet0Layout{nullptr};
		vk::DescriptorPool      globalDescriptorPool{nullptr};
		vk::Buffer              globalUboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DeviceMemory        globalUboMemory[FRAME_OVERLAP]{nullptr, nullptr};
		void*                   globalUboMapped[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet       globalDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		void InitGlobalUBO();
		void CleanupGlobalUBO();

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		ShaderWatcher       shaderWatcher;
		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica
