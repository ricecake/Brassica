#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "GLFW/glfw3.h"
#include "graph/PhysicalRegistry.hpp"
#include "InputHandler.hpp"
#include "passes/DeferredPass.hpp"
#include "passes/GradientPass.hpp"
#include "passes/TerrainPass.hpp"
#include "passes/WaterPass.hpp"
#include "ShaderWatcher.hpp"
#include "TaskScheduler.h"
#include "terrain/AsyncTerrainUploader.hpp"
#include "terrain/TerrainClipmap.hpp"
#include "types/CameraData.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "vk_mem_alloc.h"
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

		vk::Instance GetInstance() const { return instance; }

		vk::Device GetDevice() const { return device; }

		VmaAllocator GetAllocator() const { return allocator; }

		vk::Extent2D GetSwapchainExtent() const {
			return vk::Extent2D{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		}

		vk::Format GetSwapchainFormat() const { return static_cast<vk::Format>(vkbSwapchain.image_format); }

		vk::PipelineCache GetPipelineCache() const { return pipelineCache; }

		ShaderWatcher& GetShaderWatcher() { return shaderWatcher; }

		void SetFov(float fov) { camera.fov = fov; }

		float GetFov() const { return camera.fov; }

		CameraData& GetCamera() { return camera; }

		const CameraData& GetCamera() const { return camera; }

		void UpdateCamera(float deltaTime);

		void SetInputHandler(std::shared_ptr<IInputHandler> handler) { inputHandler = std::move(handler); }

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
		vk::PipelineCache          pipelineCache{nullptr};
		vk::SurfaceKHR             surface;
		vk::Queue                  graphicsQueue;
		uint32_t                   graphicsQueueFamily{0};
		vkb::Swapchain             vkbSwapchain;
		std::vector<vk::Image>     swapchainImages;
		std::vector<vk::ImageView> swapchainImageViews;
		std::vector<vk::Semaphore> swapchainRenderSemaphores;
		vk::Semaphore              frameTimelineSemaphore{nullptr};

		bool       windowResized{false};
		CameraData camera{};
		double     lastFrameTime{0.0};
		double     lastMouseX{0.0};
		double     lastMouseY{0.0};
		bool       firstMouse{true};

		std::shared_ptr<IInputHandler> inputHandler{nullptr};

		std::unique_ptr<GradientPass> gradientPass;
		std::unique_ptr<TerrainPass>  terrainPass;
		std::unique_ptr<DeferredPass> deferredPass;
		std::unique_ptr<WaterPass>    waterPass;

		TerrainClipmap       terrainClipmap;
		AsyncTerrainUploader terrainUploader;

		uint32_t     globalSeed{0};
		std::mt19937 rng;

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

		// Engine-owned and persistent across frames (not a per-frame stack local, unlike the old
		// FrameGraph fg;) -- this is what lets a resource a node doesn't touch this frame simply
		// keep existing rather than being torn down and rebuilt, and is the entire mechanism
		// behind AtmosphereLUT-style regeneration throttling (see PhysicalResourceRegistry's
		// desc-match reuse in ProvisionTexture/ProvisionBuffer). Constructed with a default
		// device/allocator at Engine construction time; SetDeviceAndAllocator wires in the real
		// ones once InitVulkan has run.
		graph::PhysicalResourceRegistry physicalRegistry;

		EngineOptions       options{};
		uint32_t            validationErrorCount{0};
		uint32_t            validationWarningCount{0};
		ShaderWatcher       shaderWatcher;
		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica
