#pragma once

#include <vector>

#include "vulkan/vulkan.hpp"

#include "GLFW/glfw3.h"
#include "passes/GradientPass.hpp"
#include "TaskScheduler.h"
#include "VkBootstrap.h"

namespace brassica {

	// Double-buffering data to prevent CPU/GPU stalling
	struct FrameData {
		VkCommandPool   commandPool;
		VkCommandBuffer commandBuffer;
		VkSemaphore     swapchainSemaphore; // Signaled when image is acquired
		VkSemaphore     renderSemaphore;    // Signaled when rendering finishes
		VkFence         renderFence;        // Signaled when GPU finishes frame execution
	};

	class Engine {
	public:
		void Init();
		void Run();
		void Cleanup();

		VkDevice GetDevice() const { return device; }

		VkExtent2D GetSwapchainExtent() const { return vkbSwapchain.extent; }

		VkFormat GetSwapchainFormat() const { return vkbSwapchain.image_format; }

	private:
		void InitWindow();
		void InitVulkan();
		void InitSwapchain();
		void InitCommands();
		void InitSyncStructures();

		void DrawFrame();

		GLFWwindow*                   window{nullptr};
		uint32_t                      frameNumber{0};
		static constexpr unsigned int FRAME_OVERLAP = 2;
		FrameData                     frames[FRAME_OVERLAP];

		// Vulkan Core
		vkb::Instance            vkbInst;
		vkb::Device              vkbDevice;
		VkInstance               instance;
		VkPhysicalDevice         chosenGPU;
		VkDevice                 device;
		VkSurfaceKHR             surface;
		VkQueue                  graphicsQueue;
		uint32_t                 graphicsQueueFamily;
		vkb::Swapchain           vkbSwapchain;
		std::vector<VkImage>     swapchainImages;
		std::vector<VkImageView> swapchainImageViews;

		std::unique_ptr<GradientPass> gradientPass;

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica