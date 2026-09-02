#pragma once

#include <vector>

#include "vulkan/vulkan.h"

#include "GLFW/glfw3.h"
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

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica