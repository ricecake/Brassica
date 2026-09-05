#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

#include "GLFW/glfw3.h"
#include <random>

#include "InputHandler.hpp"
#include "passes/DeferredPass.hpp"
#include "passes/GradientPass.hpp"
#include "passes/MeshCubePass.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "ShaderWatcher.hpp"
#include "TaskScheduler.h"
#include "VkBootstrap.h"

namespace brassica {

	struct EngineOptions {
		bool     headless{false};
		uint32_t maxFrames{0};

		static EngineOptions FromArgs(int argc, char** argv) {
			EngineOptions opts;
			for (int i = 1; i < argc; ++i) {
				std::string arg = argv[i];
				if (arg == "--headless" || arg == "-headless") {
					opts.headless = true;
					if (i + 1 < argc && argv[i + 1][0] != '-') {
						try {
							opts.maxFrames = static_cast<uint32_t>(std::stoul(argv[i + 1]));
							i++;
						} catch (...) {
						}
					}
				} else if (arg == "--frames" || arg == "-frames") {
					if (i + 1 < argc) {
						try {
							opts.maxFrames = static_cast<uint32_t>(std::stoul(argv[i + 1]));
							i++;
						} catch (...) {
						}
					}
				}
			}
			return opts;
		}
	};

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
		void Init(const EngineOptions& opts = {});
		void Run();
		void Cleanup();

		const EngineOptions& GetOptions() const { return options; }

		uint32_t GetValidationErrorCount() const { return validationErrorCount; }
		uint32_t GetValidationWarningCount() const { return validationWarningCount; }

		static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT             messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void*                                       pUserData
		);

		vk::Device GetDevice() const { return device; }
		VmaAllocator GetAllocator() const { return allocator; }

		vk::Extent2D GetSwapchainExtent() const {
			return vk::Extent2D{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		}

		vk::Format GetSwapchainFormat() const { return static_cast<vk::Format>(vkbSwapchain.image_format); }

		ShaderWatcher& GetShaderWatcher() { return shaderWatcher; }

		void SetFov(float fov) { cameraFov = fov; }
		float GetFov() const { return cameraFov; }

		void SetInputHandler(std::shared_ptr<IInputHandler> handler) {
			inputHandler = std::move(handler);
		}

		template <InputHandlerConcept T, typename... Args>
		void SetInputHandler(Args&&... args) {
			if constexpr (std::derived_from<T, IInputHandler>) {
				inputHandler = std::make_shared<T>(std::forward<Args>(args)...);
			} else {
				inputHandler = std::make_shared<InputHandlerAdapter<T>>(T(std::forward<Args>(args)...));
			}
		}

		std::shared_ptr<IInputHandler> GetInputHandler() const { return inputHandler; }

		template <InputHandlerConcept T, typename... Args>
		static void SetDefaultInputHandlerType(Args&&... args) {
			brassica::SetDefaultInputHandlerType<T>(std::forward<Args>(args)...);
		}

		void OnFramebufferResize(int width, int height);

	private:
		void InitWindow();
		bool InitVulkan();
		void InitSwapchain();
		void InitCommands();
		void InitSyncStructures();

		void RecreateSwapchain();
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
		std::vector<vk::Semaphore> swapchainRenderSemaphores;

		bool  windowResized{false};
		float cameraFov{1.2f};

		std::shared_ptr<IInputHandler> inputHandler{nullptr};

		std::unique_ptr<GradientPass> gradientPass;
		std::unique_ptr<MeshCubePass> meshCubePass;
		std::unique_ptr<DeferredPass> deferredPass;

		uint32_t      globalSeed{0};
		std::mt19937  rng;

		// Vulkan Memory Allocator
		VmaAllocator allocator{VK_NULL_HANDLE};

		// Global Descriptor Set 0 (FrameUBO)
		vk::DescriptorSetLayout globalSet0Layout{nullptr};
		vk::DescriptorPool      globalDescriptorPool{nullptr};
		vk::Buffer              globalUboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation           globalUboAllocations[FRAME_OVERLAP]{nullptr, nullptr};
		void*                   globalUboMapped[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet       globalDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		void InitGlobalUBO();
		void CleanupGlobalUBO();

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		EngineOptions       options{};
		uint32_t            validationErrorCount{0};
		uint32_t            validationWarningCount{0};
		ShaderWatcher       shaderWatcher;
		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica
