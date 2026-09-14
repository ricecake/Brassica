#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "EngineConstants.hpp"
#include "GLFW/glfw3.h"
#include "graph/PhysicalRegistry.hpp"
#include "InputHandler.hpp"
#include "passes/AllNodes.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "TaskScheduler.h"
#include "terrain/AsyncTerrainUploader.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"
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

		// Every top-level render node used to be a named member here (GradientNode, TerrainNode,
		// DeferredNode, WaterNode, TransmittanceLUTNode, MultiScatteringLUTNode,
		// ParticleSystemNode) with hand-written Init/Destroy/RegisterRef calls in Engine.cpp for
		// each. They're auto-registered now (render::EngineNodeRegistry, via
		// render::NodeRegistrar<T> -- see include/render/NodeLifecycle.hpp and
		// include/passes/AllNodes.hpp) and live in the registry's own storage instead -- Engine
		// reaches a specific one only where it still needs to (SetFrameParams, in DrawFrame) via
		// EngineNodeRegistry::Instance().Get<T>().

		// What's left of the old TerrainPass once its pipeline/shader ownership moved above --
		// the terrain BLAS/TLAS build, unchanged, now living in terrain/ rather than passes/
		// (see TerrainAccelerationStructure's own comment for why). Default-constructed, wired
		// via Init() once instance/device exist, same pattern as physicalRegistry/pipelineLibrary.
		TerrainAccelerationStructure terrainAS;

		TerrainClipmap       terrainClipmap;
		AsyncTerrainUploader terrainUploader;

		uint32_t     globalSeed{0};
		std::mt19937 rng;

		// Vulkan Memory Allocator
		VmaAllocator allocator{VK_NULL_HANDLE};

		// Frame set (set 0): just the FrameUBO, always bound by every node regardless of what
		// else it touches -- every shader gets ready access to camera/time/frame data without
		// declaring it as a graph resource dependency. Genuinely double-buffered (unlike the
		// bindless set below): the UBO's contents are CPU-written fresh every frame, so frame
		// N's write must not land in the copy frame N-1's still-in-flight GPU work may still be
		// reading.
		vk::DescriptorSetLayout frameSetLayout{nullptr};
		vk::DescriptorPool      frameDescriptorPool{nullptr};
		vk::Buffer              frameUboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		VmaAllocation           frameUboAllocations[FRAME_OVERLAP]{nullptr, nullptr};
		void*                   frameUboMapped[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet       frameDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		void InitFrameSet();
		void CleanupFrameSet();

		// The bindless set (set 1): one instance, never duplicated per frame -- a resource's
		// descriptor is written once at creation and read for the rest of its life, so unlike
		// the frame set above there is no in-flight copy to keep separate. (An earlier version
		// of this folded the per-frame UBO into this same set, which forced duplicating the
		// whole set -- and therefore every bindless descriptor write -- per frame just for the
		// UBO's sake, reintroducing exactly the in-flight write hazard bindless indexing exists
		// to avoid. See PhysicalResourceRegistry::BindlessBindings.)
		vk::DescriptorSetLayout bindlessSetLayout{nullptr};
		vk::DescriptorPool      bindlessDescriptorPool{nullptr};
		vk::DescriptorSet       bindlessDescriptorSet{nullptr};
		vk::Sampler             bindlessSamplers[4]{nullptr, nullptr, nullptr, nullptr};
		std::uint32_t           maxBindlessSampledImages{0};

		void InitGlobalDescriptors();
		void CleanupGlobalDescriptors();

		FrameData& GetCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

		// Engine-owned and persistent across frames (not a per-frame stack local, unlike the old
		// FrameGraph fg;) -- this is what lets a resource a node doesn't touch this frame simply
		// keep existing rather than being torn down and rebuilt, and is the entire mechanism
		// behind AtmosphereLUT-style regeneration throttling (see PhysicalResourceRegistry's
		// desc-match reuse in ProvisionTexture/ProvisionBuffer). Constructed with a default
		// device/allocator at Engine construction time; SetDeviceAndAllocator wires in the real
		// ones once InitVulkan has run.
		graph::PhysicalResourceRegistry physicalRegistry;

		// Engine-owned and persistent across frames, mirroring physicalRegistry immediately
		// above -- what lets a node reconstructed fresh every frame (GradientNode and everything
		// ported after it, none of which own a persistent Pass object anymore) resolve the exact
		// same pipeline request every frame as a cache hit instead of rebuilding a real
		// vk::Pipeline 60+ times a second. Constructed with a default device/cache at Engine
		// construction time; SetDeviceAndCache wires in the real ones once InitVulkan has run,
		// same as physicalRegistry.SetDeviceAndAllocator.
		render::PipelineLibrary pipelineLibrary;

		EngineOptions       options{};
		uint32_t            validationErrorCount{0};
		uint32_t            validationWarningCount{0};
		ShaderWatcher       shaderWatcher;
		enki::TaskScheduler taskScheduler;
	};

} // namespace brassica
