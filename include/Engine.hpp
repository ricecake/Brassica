#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "ArgparseManager.hpp"
#include "ConfigManager.hpp"
#include "EngineConstants.hpp"
#include "GLFW/glfw3.h"
#include "graph/PhysicalRegistry.hpp"
#include "ImGuiManager.hpp"
#include "InputHandler.hpp"
#include "lighting/LightManager.hpp"
#include "lighting/LightningManager.hpp"
#include "passes/AllNodes.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "TaskScheduler.h"
#include "terrain/AsyncTerrainUploader.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"
#include "terrain/TerrainClipmap.hpp"
#include <entt/entity/registry.hpp>

#include "SystemHandler.hpp"
#include "types/CameraData.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "types/ubo/LightingUBO.hpp"
#include "vk_mem_alloc.h"
#include "VkBootstrap.h"

namespace brassica {

	struct EngineOptions {
		bool        headless{false};
		uint32_t    maxFrames{0};
		std::string configFile{"config.ini"};
		std::string appName{"Sandbox"};

		static EngineOptions FromArgs(int argc, char** argv) {
			ArgparseManager argManager;
			argManager.Initialize();
			argManager.Parse(argc, argv);

			EngineOptions opts;
			opts.headless = argManager.GetHeadless();
			opts.maxFrames = argManager.GetMaxFrames();
			opts.configFile = argManager.GetConfigFile();
			opts.appName = argManager.GetAppName();
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

		ImGuiManager& GetImGuiManager() { return imguiManager; }

		const ImGuiManager& GetImGuiManager() const { return imguiManager; }

		ConfigManager& GetConfigManager() { return configManager; }

		const ConfigManager& GetConfigManager() const { return configManager; }

		ServiceLocator& GetServiceLocator() { return serviceLocator; }

		const ServiceLocator& GetServiceLocator() const { return serviceLocator; }

		template <typename T>
		std::shared_ptr<T> GetService() {
			return serviceLocator.Get<T>();
		}

		void SetFov(float fov) { camera.fov = fov; }

		float GetFov() const { return camera.fov; }

		CameraData& GetCamera() { return camera; }

		const CameraData& GetCamera() const { return camera; }

		LightManager& GetLightManager() { return lightManager; }

		const LightManager& GetLightManager() const { return lightManager; }

		LightningManager& GetLightningManager() { return lightningManager; }

		const LightningManager& GetLightningManager() const { return lightningManager; }

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

		entt::registry& GetRegistry() { return registry; }

		const entt::registry& GetRegistry() const { return registry; }

		void AddSystemHandler(std::shared_ptr<SystemHandler> handler);

		template <typename T, typename... Args>
		std::shared_ptr<T> AddSystemHandler(Args&&... args) {
			auto handler = std::make_shared<T>(std::forward<Args>(args)...);
			AddSystemHandler(handler);
			return handler;
		}

		const std::vector<std::shared_ptr<SystemHandler>>& GetSystemHandlers() const { return systemHandlers; }

	private:
		void InitWindow();
		bool InitVulkan();
		void InitSwapchain();
		void InitCommands();
		void InitSyncStructures();

		void RecreateSwapchain();
		void DrawFrame();

		GLFWwindow* window{nullptr};
		uint32_t    frameNumber{0};
		FrameData   frames[FRAME_OVERLAP]; // brassica::FRAME_OVERLAP, EngineConstants.hpp

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
		vk::Queue                  computeQueue;
		uint32_t                   computeQueueFamily{0};
		vk::Queue                  transferQueue;
		uint32_t                   transferQueueFamily{0};
		graph::QueueSet            queueSet{};
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

		LightManager     lightManager;
		LightningManager lightningManager;

		TerrainAccelerationStructure terrainAS;

		TerrainClipmap       terrainClipmap;
		AsyncTerrainUploader terrainUploader;

		uint32_t     globalSeed{0};
		std::mt19937 rng;

		// Vulkan Memory Allocator
		VmaAllocator allocator{VK_NULL_HANDLE};

		// Frame set (set 0): persistent engine-wide bindings for FrameUBO, LightingUBO, LightsBuffer, ClusterGridBuffer
		vk::DescriptorSetLayout frameSetLayout{nullptr};
		vk::DescriptorPool      frameDescriptorPool{nullptr};

		vk::Buffer    frameUboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation frameUboAllocations[FRAME_OVERLAP]{nullptr, nullptr};
		void*         frameUboMapped[FRAME_OVERLAP]{nullptr, nullptr};

		vk::Buffer    lightingUboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation lightingUboAllocations[FRAME_OVERLAP]{nullptr, nullptr};
		void*         lightingUboMapped[FRAME_OVERLAP]{nullptr, nullptr};

		vk::Buffer    lightsSSBOBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation lightsSSBOAllocations[FRAME_OVERLAP]{nullptr, nullptr};
		void*         lightsSSBOMapped[FRAME_OVERLAP]{nullptr, nullptr};

		vk::Buffer    clusterGridBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation clusterGridAllocations[FRAME_OVERLAP]{nullptr, nullptr};

		vk::DescriptorSet frameDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		void InitFrameSet();
		void CleanupFrameSet();

		vk::DescriptorSetLayout bindlessSetLayout{nullptr};
		vk::DescriptorPool      bindlessDescriptorPool{nullptr};
		vk::DescriptorSet       bindlessDescriptorSet{nullptr};
		vk::Sampler             bindlessSamplers[4]{nullptr, nullptr, nullptr, nullptr};
		std::uint32_t           maxBindlessSampledImages{0};

		void InitGlobalDescriptors();
		void CleanupGlobalDescriptors();

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		graph::PhysicalResourceRegistry physicalRegistry;
		render::PipelineLibrary         pipelineLibrary;

		ImGuiManager        imguiManager;
		ServiceLocator      serviceLocator;
		EngineOptions       options{};
		uint32_t            validationErrorCount{0};
		uint32_t            validationWarningCount{0};
		ArgparseManager     argparseManager;
		ConfigManager       configManager;
		ShaderWatcher       shaderWatcher;
		enki::TaskScheduler taskScheduler;

		entt::registry                             registry;
		std::vector<std::shared_ptr<SystemHandler>> systemHandlers;
	};

} // namespace brassica
