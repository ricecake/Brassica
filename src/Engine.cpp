#include "Engine.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

#include "spdlog/spdlog.h"

#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/Dot.hpp"
#include "graph/Node.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	void Engine::OnFramebufferResize(int width, int height) {
		windowResized = true;
		if (inputHandler) {
			inputHandler->OnFramebufferSize(window, width, height);
		}
	}

	VKAPI_ATTR VkBool32 VKAPI_CALL Engine::VulkanDebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT             messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void*                                       pUserData
	) {
		auto* engine = static_cast<Engine*>(pUserData);

		if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
			spdlog::error("[Vulkan Validation Error] {}", pCallbackData->pMessage);
			if (engine)
				engine->validationErrorCount++;
		} else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
			spdlog::warn("[Vulkan Validation Warning] {}", pCallbackData->pMessage);
			if (engine)
				engine->validationWarningCount++;
		} else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
			spdlog::info("[Vulkan Validation Info] {}", pCallbackData->pMessage);
		} else {
			spdlog::debug("[Vulkan Validation Debug] {}", pCallbackData->pMessage);
		}

		return VK_FALSE;
	}

	void Engine::InitWindow() {
		if (options.headless) {
			window = nullptr;
			return;
		}
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		window = glfwCreateWindow(1280, 720, "Brassica Engine", nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);

		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int width, int height) {
			auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(w));
			if (engine) {
				engine->OnFramebufferResize(width, height);
			}
		});

		glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
			auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(w));
			if (engine && engine->GetInputHandler()) {
				engine->GetInputHandler()->OnKey(w, key, scancode, action, mods);
			}
		});

		glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
			auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(w));
			if (engine && engine->GetInputHandler()) {
				engine->GetInputHandler()->OnMouseButton(w, button, action, mods);
			}
		});

		glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos) {
			auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(w));
			if (engine && engine->GetInputHandler()) {
				engine->GetInputHandler()->OnCursorPos(w, xpos, ypos);
			}
		});

		glfwSetScrollCallback(window, [](GLFWwindow* w, double xoffset, double yoffset) {
			auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(w));
			if (engine && engine->GetInputHandler()) {
				engine->GetInputHandler()->OnScroll(w, xoffset, yoffset);
			}
		});
	}

	void Engine::InitSwapchain() {
		vkb::SwapchainBuilder swapchainBuilder{chosenGPU, device, surface};
		auto                  swap_ret = swapchainBuilder
											 .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
											 .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
											 .add_fallback_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
											 .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
											 .set_desired_extent(1280, 720)
											 .add_image_usage_flags(
												 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
											 )
											 .build();

		if (!swap_ret) {
			spdlog::error("Failed to create swapchain: {}", swap_ret.error().message());
			return;
		}

		vkbSwapchain = swap_ret.value();

		auto raw_images = vkbSwapchain.get_images().value();
		swapchainImages.assign(raw_images.begin(), raw_images.end());

		auto raw_image_views = vkbSwapchain.get_image_views().value();
		swapchainImageViews.assign(raw_image_views.begin(), raw_image_views.end());

		for (auto sem : swapchainRenderSemaphores) {
			if (sem)
				device.destroySemaphore(sem);
		}
		swapchainRenderSemaphores.clear();

		vk::SemaphoreCreateInfo semaphoreCreateInfo{};
		for (size_t i = 0; i < swapchainImages.size(); i++) {
			swapchainRenderSemaphores.push_back(device.createSemaphore(semaphoreCreateInfo));
		}
	}

	void Engine::RecreateSwapchain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		for (auto view : swapchainImageViews) {
			device.destroyImageView(view);
		}
		swapchainImageViews.clear();
		swapchainImages.clear();

		vkb::SwapchainBuilder swapchainBuilder{chosenGPU, device, surface};
		auto                  swap_ret = swapchainBuilder
											 .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
											 .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
											 .add_fallback_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
											 .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
											 .set_desired_extent(static_cast<uint32_t>(width), static_cast<uint32_t>(height))
											 .set_old_swapchain(vkbSwapchain)
											 .add_image_usage_flags(
												 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
											 )
											 .build();

		if (!swap_ret) {
			spdlog::error("Failed to recreate swapchain: {}", swap_ret.error().message());
			return;
		}

		vkb::destroy_swapchain(vkbSwapchain);
		vkbSwapchain = swap_ret.value();

		auto raw_images = vkbSwapchain.get_images().value();
		swapchainImages.assign(raw_images.begin(), raw_images.end());

		auto raw_image_views = vkbSwapchain.get_image_views().value();
		swapchainImageViews.assign(raw_image_views.begin(), raw_image_views.end());

		windowResized = false;
	}

	void Engine::InitCommands() {
		vk::CommandPoolCreateInfo commandPoolCreateInfo{};
		commandPoolCreateInfo.setQueueFamilyIndex(graphicsQueueFamily);
		commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			frames[i].commandPool = device.createCommandPool(commandPoolCreateInfo);

			vk::CommandBufferAllocateInfo cmdAllocInfo{};
			cmdAllocInfo.setCommandPool(frames[i].commandPool);
			cmdAllocInfo.setCommandBufferCount(1);
			cmdAllocInfo.setLevel(vk::CommandBufferLevel::ePrimary);

			frames[i].commandBuffer = device.allocateCommandBuffers(cmdAllocInfo).front();
		}
	}

	void Engine::InitSyncStructures() {
		vk::SemaphoreTypeCreateInfo typeInfo{};
		typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
		typeInfo.setInitialValue(0);

		vk::SemaphoreCreateInfo timelineCreateInfo{};
		timelineCreateInfo.setPNext(&typeInfo);
		frameTimelineSemaphore = device.createSemaphore(timelineCreateInfo);

		vk::SemaphoreCreateInfo semaphoreCreateInfo{};

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			frames[i].swapchainSemaphore = device.createSemaphore(semaphoreCreateInfo);
		}
	}

	void Engine::Cleanup() {
		if (device) {
			device.waitIdle();

			shaderWatcher.StopWatching();

			// Must run before vmaDestroyAllocator/device.destroy() below -- PhysicalTexture's
			// destructor calls device.destroyImageView and vmaDestroyImage on whatever the
			// registry still owns (G-buffer textures, the gradient background, ...).
			physicalRegistry.Reset();

			if (waterPass) {
				waterPass->DestroyPipeline();
				waterPass.reset();
			}

			if (deferredPass) {
				deferredPass->DestroyPipeline();
				deferredPass.reset();
			}

			if (terrainPass) {
				terrainPass->DestroyPipeline();
				terrainPass.reset();
			}

			terrainUploader.Cleanup();
			terrainClipmap.Cleanup();

			if (gradientPass) {
				gradientPass->DestroyPipeline();
				gradientPass.reset();
			}

			CleanupGlobalUBO();

			for (int i = 0; i < FRAME_OVERLAP; i++) {
				device.destroySemaphore(frames[i].swapchainSemaphore);
				device.destroyCommandPool(frames[i].commandPool);
			}

			if (frameTimelineSemaphore) {
				device.destroySemaphore(frameTimelineSemaphore);
				frameTimelineSemaphore = nullptr;
			}

			for (auto sem : swapchainRenderSemaphores) {
				if (sem)
					device.destroySemaphore(sem);
			}
			swapchainRenderSemaphores.clear();

			for (auto view : swapchainImageViews) {
				device.destroyImageView(view);
			}
			vkb::destroy_swapchain(vkbSwapchain);

			if (allocator != VK_NULL_HANDLE) {
				vmaDestroyAllocator(allocator);
				allocator = VK_NULL_HANDLE;
			}

			if (pipelineCache) {
				size_t cacheSize = 0;
				if (device.getPipelineCacheData(pipelineCache, &cacheSize, nullptr) == vk::Result::eSuccess &&
				    cacheSize > 0) {
					std::vector<char> cacheData(cacheSize);
					if (device.getPipelineCacheData(pipelineCache, &cacheSize, cacheData.data()) ==
					    vk::Result::eSuccess) {
						std::ofstream outFile("pipeline_cache.bin", std::ios::binary);
						if (outFile.is_open()) {
							outFile.write(cacheData.data(), static_cast<std::streamsize>(cacheData.size()));
							spdlog::info("Saved pipeline cache data ({} bytes).", cacheSize);
						}
					}
				}
				device.destroyPipelineCache(pipelineCache);
				pipelineCache = nullptr;
			}

			device.destroy();
			instance.destroySurfaceKHR(surface);
			vkb::destroy_instance(vkbInst);
		}

		if (window) {
			glfwDestroyWindow(window);
			glfwTerminate();
		}
	}

	void Engine::Init(const EngineOptions& opts) {
		if (!inputHandler) {
			inputHandler = CreateDefaultInputHandler();
		}

		options = opts;
		InitWindow();
		if (!InitVulkan()) {
			spdlog::error("Vulkan initialization failed; engine cannot start.");
			return;
		}
		InitSwapchain();
		InitCommands();
		InitSyncStructures();

		std::random_device rd;
		globalSeed = rd();
		rng.seed(globalSeed);

		InitGlobalUBO();

		std::string shaderDir = "shaders";
		if (!std::filesystem::exists(shaderDir)) {
			if (std::filesystem::exists("bin/shaders")) {
				shaderDir = "bin/shaders";
			} else if (std::filesystem::exists(std::string(BRASSICA_BUILD_DIR) + "/bin/shaders")) {
				shaderDir = std::string(BRASSICA_BUILD_DIR) + "/bin/shaders";
			} else if (std::filesystem::exists(std::string(BRASSICA_BUILD_DIR) + "/../shaders")) {
				shaderDir = std::string(BRASSICA_BUILD_DIR) + "/../shaders";
			}
		}
		shaderWatcher.WatchDirectory(shaderDir);

		gradientPass =
			std::make_unique<GradientPass>(device, vk::Format::eR16G16B16A16Sfloat, &shaderWatcher, GetPipelineCache());
		terrainPass =
			std::make_unique<TerrainPass>(instance, device, globalSet0Layout, &shaderWatcher, GetPipelineCache());
		deferredPass = std::make_unique<DeferredPass>(
			device,
			globalSet0Layout,
			GetSwapchainFormat(),
			&shaderWatcher,
			GetPipelineCache()
		);
		waterPass = std::make_unique<WaterPass>(
			instance,
			device,
			globalSet0Layout,
			GetSwapchainFormat(),
			&shaderWatcher,
			GetPipelineCache()
		);

		// Position camera 50 units above terrain, looking downslope towards the y-zero plane
		float     camX = 0.0f;
		float     camZ = 0.0f;
		glm::vec4 terrainSample = TerrainClipmap::SampleTerrain(camX, camZ, 0.5f);
		float     terrainY = terrainSample.x;
		camera.position = glm::vec3(camX, terrainY + 50.0f, camZ);

		glm::vec2 downslope(terrainSample.y, terrainSample.w);
		if (glm::length(downslope) > 0.001f) {
			downslope = glm::normalize(downslope);
		} else {
			downslope = glm::vec2(0.0f, 1.0f);
		}

		glm::vec3 target = camera.position + glm::vec3(downslope.x, 0.0f, downslope.y) * 100.0f;
		target.y = 0.0f;

		glm::vec3 lookDir = glm::normalize(target - camera.position);
		camera.pitch = std::asin(std::clamp(lookDir.y, -0.99f, 0.99f));
		camera.yaw = std::atan2(-lookDir.x, -lookDir.z);
		camera.roll = 0.0f;

		terrainClipmap.Init(device, allocator, 8, 0.5f, camera.farPlane, camera.position);
		terrainUploader.Init(device, allocator, graphicsQueueFamily, 32);

		// Async upload initial heightmaps
		for (uint32_t l = 0; l < terrainClipmap.GetNumLODs(); ++l) {
			auto mapData = terrainClipmap.GenerateLevelMap(l);
			terrainUploader.UploadLevelAsync(
				l,
				mapData,
				terrainClipmap.GetImage(),
				TERRAIN_MAP_DIM,
				TERRAIN_MAP_DIM,
				graphicsQueue
			);
		}
		terrainPass->UpdateClipmapDescriptor(terrainClipmap.GetImageView(), terrainClipmap.GetSampler());

		taskScheduler.Initialize();
		camera.UpdateMatrices(16.0f / 9.0f);
		lastFrameTime = glfwGetTime();
		spdlog::info("Brassica Engine Initialized (headless: {}).", options.headless);
	}

	void Engine::UpdateCamera(float deltaTime) {
		auto* defaultHandler = dynamic_cast<DefaultInputHandler*>(inputHandler.get());

		if (defaultHandler) {
			if (defaultHandler->IsKeyJustPressed(GLFW_KEY_0)) {
				camera.isCaptured = !camera.isCaptured;
				if (window) {
					glfwSetInputMode(
						window,
						GLFW_CURSOR,
						camera.isCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL
					);
				}
				firstMouse = true;
			}

			if (defaultHandler->IsKeyJustPressed(GLFW_KEY_PAGE_UP)) {
				camera.speed = std::min(camera.maxSpeed, camera.speed + camera.speedStep);
			}
			if (defaultHandler->IsKeyJustPressed(GLFW_KEY_PAGE_DOWN)) {
				camera.speed = std::max(camera.minSpeed, camera.speed - camera.speedStep);
			}
			if (defaultHandler->IsKeyJustPressed(GLFW_KEY_HOME)) {
				camera.speed = camera.defaultSpeed;
			}
			if (defaultHandler->IsKeyJustPressed(GLFW_KEY_END)) {
				camera.speed = camera.maxSpeed;
			}
		}

		if (camera.isCaptured && defaultHandler) {
			auto [mx, my] = defaultHandler->GetCursorPos();
			if (firstMouse) {
				lastMouseX = mx;
				lastMouseY = my;
				firstMouse = false;
			}
			double dx = mx - lastMouseX;
			double dy = my - lastMouseY;
			lastMouseX = mx;
			lastMouseY = my;

			constexpr float sensitivity = 0.002f;
			camera.yaw -= static_cast<float>(dx) * sensitivity;
			camera.pitch -= static_cast<float>(dy) * sensitivity;
			camera.pitch = std::clamp(camera.pitch, -1.55f, 1.55f);

			constexpr float rollSpeed = 1.5f;
			if (defaultHandler->IsKeyPressed(GLFW_KEY_Q)) {
				camera.roll += rollSpeed * deltaTime;
			}
			if (defaultHandler->IsKeyPressed(GLFW_KEY_E)) {
				camera.roll -= rollSpeed * deltaTime;
			}

			glm::vec3 moveDir{0.0f};
			if (defaultHandler->IsKeyPressed(GLFW_KEY_W)) {
				moveDir += camera.GetForward();
			}
			if (defaultHandler->IsKeyPressed(GLFW_KEY_S)) {
				moveDir -= camera.GetForward();
			}
			if (defaultHandler->IsKeyPressed(GLFW_KEY_A)) {
				moveDir -= camera.GetRight();
			}
			if (defaultHandler->IsKeyPressed(GLFW_KEY_D)) {
				moveDir += camera.GetRight();
			}

			if (defaultHandler->IsKeyPressed(GLFW_KEY_SPACE)) {
				moveDir += glm::vec3(0.0f, 1.0f, 0.0f);
			}
			if (defaultHandler->IsKeyPressed(GLFW_KEY_LEFT_SHIFT) ||
			    defaultHandler->IsKeyPressed(GLFW_KEY_RIGHT_SHIFT)) {
				moveDir -= glm::vec3(0.0f, 1.0f, 0.0f);
			}

			if (glm::length(moveDir) > 0.0001f) {
				camera.position += glm::normalize(moveDir) * camera.speed * deltaTime;
			}
		} else if (defaultHandler) {
			auto [mx, my] = defaultHandler->GetCursorPos();
			lastMouseX = mx;
			lastMouseY = my;
		}
	}

	bool Engine::InitVulkan() {
		// Query maximum available Vulkan API version from the system
		uint32_t systemVersion = VK_API_VERSION_1_0;
		if (vk::enumerateInstanceVersion(&systemVersion) != vk::Result::eSuccess) {
			systemVersion = VK_API_VERSION_1_0;
		}

		constexpr uint32_t targetMajor = 1;
		constexpr uint32_t targetMinor = 3;
		const uint32_t     targetVersion = VK_MAKE_API_VERSION(0, targetMajor, targetMinor, 0);

		uint32_t chosenMajor = targetMajor;
		uint32_t chosenMinor = targetMinor;

		if (systemVersion < targetVersion) {
			chosenMajor = VK_API_VERSION_MAJOR(systemVersion);
			chosenMinor = VK_API_VERSION_MINOR(systemVersion);
			spdlog::warn(
				"Requested Vulkan API version {}.{} is higher than maximum supported version {}.{}. Using version "
				"{}.{}.",
				targetMajor,
				targetMinor,
				chosenMajor,
				chosenMinor,
				chosenMajor,
				chosenMinor
			);
		}

		// 1. Instance
		vkb::InstanceBuilder builder;
		builder.set_app_name("Brassica")
			.require_api_version(chosenMajor, chosenMinor, 0)
			.set_debug_callback(Engine::VulkanDebugCallback)
			.set_debug_callback_user_data_pointer(this);

		if (options.headless) {
			builder.set_headless(true);
			builder.enable_extension(VK_KHR_SURFACE_EXTENSION_NAME);

			uint32_t count = 0;
			if (vk::enumerateInstanceExtensionProperties(nullptr, &count, nullptr) == vk::Result::eSuccess &&
			    count > 0) {
				std::vector<vk::ExtensionProperties> exts(count);
				if (vk::enumerateInstanceExtensionProperties(nullptr, &count, exts.data()) == vk::Result::eSuccess) {
					for (const auto& ext : exts) {
						if (std::string(ext.extensionName.data()) == VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) {
							builder.enable_extension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
							break;
						}
					}
				}
			}
		}

		auto inst_res = builder.request_validation_layers(true).build();
		if (!inst_res) {
			auto fallback_res = builder.request_validation_layers(false).build();
			if (!fallback_res) {
				spdlog::critical("Failed to create Vulkan instance: {}", fallback_res.error().message());
				return false;
			}
			vkbInst = fallback_res.value();
		} else {
			vkbInst = inst_res.value();
		}
		instance = vkbInst.instance;

		if (options.headless) {
			auto vkCreateHeadlessSurfaceEXT = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
				vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT")
			);
			if (!vkCreateHeadlessSurfaceEXT) {
				spdlog::critical("Failed to load vkCreateHeadlessSurfaceEXT function pointer.");
				return false;
			}
			VkHeadlessSurfaceCreateInfoEXT createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
			VkSurfaceKHR c_surface = VK_NULL_HANDLE;
			VkResult     res = vkCreateHeadlessSurfaceEXT(instance, &createInfo, nullptr, &c_surface);
			if (res != VK_SUCCESS) {
				spdlog::critical("Failed to create headless surface: {}", static_cast<int>(res));
				return false;
			}
			surface = c_surface;
		} else {
			VkSurfaceKHR c_surface;
			glfwCreateWindowSurface(instance, window, nullptr, &c_surface);
			surface = c_surface;
		}

		// 2. Physical Device (Enable Dynamic Rendering & Sync2)
		VkPhysicalDeviceVulkan13Features features13{};
		features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		features13.dynamicRendering = VK_TRUE;
		features13.synchronization2 = VK_TRUE;

		// Optional but required for bindless later:
		VkPhysicalDeviceVulkan12Features features12{};
		features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		features12.descriptorIndexing = VK_TRUE;
		features12.descriptorBindingPartiallyBound = VK_TRUE;
		features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
		features12.timelineSemaphore = VK_TRUE;
		features12.bufferDeviceAddress = VK_TRUE;

		VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
		meshFeatures.meshShader = VK_TRUE;
		meshFeatures.taskShader = VK_TRUE;
		meshFeatures.primitiveFragmentShadingRateMeshShader = VK_TRUE;

		VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
		};
		asFeatures.accelerationStructure = VK_TRUE;

		VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
		rqFeatures.rayQuery = VK_TRUE;

		VkPhysicalDeviceFragmentShadingRateFeaturesKHR variableShadingRate{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR
		};
		variableShadingRate.primitiveFragmentShadingRate = VK_TRUE;
		variableShadingRate.attachmentFragmentShadingRate = VK_TRUE;

		vkb::PhysicalDeviceSelector selector{vkbInst};
		selector.set_surface(surface)
			.set_minimum_version(chosenMajor, chosenMinor)
			.set_required_features_13(features13)
			.set_required_features_12(features12)
			.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
			.add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
			.add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
			.add_required_extension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)
			.add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
			.add_required_extension_features(meshFeatures)
			.add_required_extension_features(asFeatures)
			.add_required_extension_features(rqFeatures)
			.add_required_extension_features(variableShadingRate);

		auto phys_ret = selector.select();
		if (!phys_ret) {
			spdlog::critical("Failed to select physical device: {}", phys_ret.error().message());
			return false;
		}

		chosenGPU = phys_ret.value().physical_device;

		// 3. Logical Device
		vkb::DeviceBuilder deviceBuilder{phys_ret.value()};
		auto               dev_ret = deviceBuilder.build();
		if (!dev_ret) {
			spdlog::critical("Failed to create logical device: {}", dev_ret.error().message());
			return false;
		}
		vkbDevice = dev_ret.value();
		device = vkbDevice.device;

		graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
		graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

		// Initialize VMA
		VmaAllocatorCreateInfo allocatorInfo{};
		allocatorInfo.physicalDevice = chosenGPU;
		allocatorInfo.device = device;
		allocatorInfo.instance = instance;
		allocatorInfo.vulkanApiVersion = VK_MAKE_API_VERSION(0, chosenMajor, chosenMinor, 0);
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

		if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
			spdlog::critical("Failed to create Vulkan Memory Allocator.");
			return false;
		}

		physicalRegistry.SetDeviceAndAllocator(device, allocator);

		// Initialize Pipeline Cache
		std::vector<char> pipelineCacheData;
		std::ifstream     cacheFile("pipeline_cache.bin", std::ios::binary | std::ios::ate);
		if (cacheFile.is_open()) {
			std::streamsize size = cacheFile.tellg();
			cacheFile.seekg(0, std::ios::beg);
			pipelineCacheData.resize(static_cast<size_t>(size));
			if (cacheFile.read(pipelineCacheData.data(), size)) {
				spdlog::info("Loaded pipeline cache data ({} bytes).", size);
			} else {
				pipelineCacheData.clear();
			}
		}

		vk::PipelineCacheCreateInfo cacheCreateInfo{};
		if (!pipelineCacheData.empty()) {
			cacheCreateInfo.setInitialDataSize(pipelineCacheData.size());
			cacheCreateInfo.setPInitialData(pipelineCacheData.data());
		}

		pipelineCache = device.createPipelineCache(cacheCreateInfo);

		return true;
	}

	void Engine::Run() {
		if (!device) {
			spdlog::warn("Engine::Run called but Vulkan device is null.");
			return;
		}

		if (options.headless || options.maxFrames > 0) {
			uint32_t targetFrames = (options.maxFrames > 0) ? options.maxFrames : 10;
			spdlog::info("Running engine in headless mode for {} frames...", targetFrames);
			for (uint32_t i = 0; i < targetFrames; ++i) {
				enki::TaskSet frameTask(1, [this](enki::TaskSetPartition range, uint32_t threadnum) { DrawFrame(); });

				taskScheduler.AddTaskSetToPipe(&frameTask);
				taskScheduler.WaitforTask(&frameTask);
			}
			spdlog::info("Completed {} frames.", targetFrames);
			return;
		}

		while (window && !glfwWindowShouldClose(window)) {
			glfwPollEvents();

			// Wrap the frame in an enkiTS task so the main thread remains free
			// for OS event pumping and window resizing.
			enki::TaskSet frameTask(1, [this](enki::TaskSetPartition range, uint32_t threadnum) { DrawFrame(); });

			taskScheduler.AddTaskSetToPipe(&frameTask);
			taskScheduler.WaitforTask(&frameTask);
		}
	}

	void Engine::DrawFrame() {
		if (windowResized) {
			RecreateSwapchain();
		}

		shaderWatcher.ProcessPendingReloads(device);

		FrameData& frame = GetCurrentFrame();

		// 1. Wait for GPU to finish the last time this frame context was used
		if (frameNumber >= FRAME_OVERLAP) {
			uint64_t              waitValue = frameNumber - FRAME_OVERLAP + 1;
			vk::SemaphoreWaitInfo waitInfo{};
			waitInfo.setSemaphores(frameTimelineSemaphore);
			waitInfo.setValues(waitValue);
			(void)device.waitSemaphores(waitInfo, 1000000000);
		}

		// 2. Acquire Swapchain Image
		auto acquireResult = device.acquireNextImageKHR(vkbSwapchain.swapchain, 1000000000, frame.swapchainSemaphore);
		if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
			RecreateSwapchain();
			return;
		} else if (acquireResult.result != vk::Result::eSuccess && acquireResult.result != vk::Result::eSuboptimalKHR) {
			spdlog::error("Failed to acquire swapchain image!");
			return;
		}
		uint32_t swapchainImageIndex = acquireResult.value;

		// 3. Record Commands
		frame.commandBuffer.reset();
		vk::CommandBufferBeginInfo cmdBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		frame.commandBuffer.begin(cmdBeginInfo);

		vk::Extent2D extent{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		vk::Format   format = GetSwapchainFormat();

		// Re-imported fresh every frame: a freshly constructed PhysicalTexture always starts
		// {eUndefined, hasDefinedContents=false}, which is correct here -- DeferredNode's
		// Modify<Swapchain> fully overwrites every pixel via a fullscreen triangle, so there is
		// nothing worth preserving from whatever the driver left behind after the last present.
		physicalRegistry.RegisterImportedTexture<Swapchain>(
			swapchainImages[swapchainImageIndex],
			swapchainImageViews[swapchainImageIndex],
			graph::ColorAttachmentDesc(extent.width, extent.height, format)
		);

		uint32_t activeFrame = frameNumber % FRAME_OVERLAP;

		double currentTime = glfwGetTime();
		float  deltaTime = static_cast<float>(currentTime - lastFrameTime);
		if (lastFrameTime == 0.0 || deltaTime <= 0.0f || deltaTime > 1.0f) {
			deltaTime = 1.0f / 60.0f;
		}
		lastFrameTime = currentTime;

		float aspect = 16.0f / 9.0f;
		if (extent.height > 0) {
			aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
		}

		UpdateCamera(deltaTime);
		camera.UpdateMatrices(aspect);

		FrameUBO ubo{};
		ubo.time = static_cast<float>(currentTime);
		ubo.fov = camera.fov;
		ubo.aspectRatio = camera.aspectRatio;
		ubo.frameIndex = frameNumber;
		ubo.globalSeed = globalSeed;
		ubo.frameRandom = static_cast<uint32_t>(rng());

		if (globalUboMapped[activeFrame]) {
			std::memcpy(globalUboMapped[activeFrame], &ubo, sizeof(FrameUBO));
		}

		terrainUploader.Poll();

		terrainClipmap.UpdateCameraPosition(camera.position, terrainUploader, graphicsQueue);

		uint32_t lods = terrainClipmap.GetNumLODs();
		uint32_t meshletsPerRow = 16;
		uint32_t totalMeshlets = lods * meshletsPerRow * meshletsPerRow;

		TerrainPushConstants terrainPush{};
		terrainPush.viewProj = camera.viewProjMatrix;
		terrainPush.cameraPos = glm::vec4(camera.position, terrainClipmap.GetBaseTexelSize());
		terrainPush.gridParams = glm::uvec4(lods, meshletsPerRow, totalMeshlets, TERRAIN_MAP_DIM);

		glm::uvec4 offsets0_3{0u};
		glm::uvec4 offsets4_7{0u};

		for (uint32_t i = 0; i < terrainClipmap.GetNumLODs(); ++i) {
			const auto& info = terrainClipmap.GetLevelInfo(i);
			uint32_t    packed = (static_cast<uint32_t>(info.gridOffset.x) & 0xFFFFu) |
				((static_cast<uint32_t>(info.gridOffset.y) & 0xFFFFu) << 16u);
			if (i < 4) {
				offsets0_3[i] = packed;
			} else if (i < 8) {
				offsets4_7[i - 4] = packed;
			}
		}
		terrainPush.lodOffsets0_3 = offsets0_3;
		terrainPush.lodOffsets4_7 = offsets4_7;

		// TLAS build stays fully out-of-band: its own transient command pool/queue, its own
		// camera-movement throttle, its own synchronous device.waitIdle() -- unchanged from
		// before this migration. Only the *result* flows into the graph, registered just like
		// the swapchain above. See the AccelerationStructure resource-kind plan for why moving
		// the build itself into the graph's command buffer was rejected (a real use-after-free
		// risk against frames still in flight).
		terrainPass->BuildOrUpdateAccelerationStructure(
			allocator,
			glm::vec3(terrainPush.cameraPos),
			terrainPush.cameraPos.w,
			terrainPush.gridParams.x
		);
		physicalRegistry.RegisterImportedAccelerationStructure<TerrainTLAS>(terrainPass->GetTLAS());

		graph::Graph frameGraph;
		frameGraph.Register<graph::Import<Swapchain>>(graph::Import<Swapchain>());
		frameGraph.Register<GradientNode>(GradientNode{.pass = gradientPass.get(), .extent = extent});
		frameGraph.Register<TerrainNode>(TerrainNode{
			.pass = terrainPass.get(),
			.extent = extent,
			.globalDescriptorSet = globalDescriptorSets[activeFrame],
			.pushConstants = terrainPush,
		});
		frameGraph.Register<DeferredNode>(DeferredNode{
			.pass = deferredPass.get(),
			.registry = &physicalRegistry,
			.extent = extent,
			.swapchainFormat = format,
			.globalDescriptorSet = globalDescriptorSets[activeFrame],
			.activeFrame = activeFrame,
			.clipmapImageView = terrainClipmap.GetImageView(),
			.clipmapSampler = terrainClipmap.GetSampler(),
			.pushConstants = terrainPush,
		});
		frameGraph.Register<WaterNode>(WaterNode{
			.pass = waterPass.get(),
			.registry = &physicalRegistry,
			.extent = extent,
			.swapchainFormat = format,
			.globalDescriptorSet = globalDescriptorSets[activeFrame],
			.activeFrame = activeFrame,
			.clipmapImageView = terrainClipmap.GetImageView(),
			.clipmapSampler = terrainClipmap.GetSampler(),
			.pushConstants = terrainPush,
		});

		graph::FrameContext             ctx{.width = extent.width, .height = extent.height, .frameIndex = frameNumber};
		graph::PhysicalExecutionBackend backend(physicalRegistry);
		graph::CommandBuffer            graphCmd{static_cast<void*>(static_cast<VkCommandBuffer>(frame.commandBuffer))};

		try {
			backend.Execute(frameGraph, ctx, graphCmd, true);
		} catch (const std::exception& e) {
			spdlog::error("Frame graph execution failed: {}", e.what());
			frame.commandBuffer.end();

			// backend.Execute throws before recording anything into frame.commandBuffer
			// (Provision, which is where this can fail, runs before the command buffer is ever
			// touched) -- so this is submitting an empty but valid begin/end pair, purely to
			// consume frame.swapchainSemaphore's signal from the acquire above. Skipping the
			// submit entirely would leave that semaphore signaled, and the next time this frame
			// slot's semaphore is reused for acquireNextImageKHR (FRAME_OVERLAP frames from now),
			// the validation layer correctly flags "Semaphore must not be currently signaled".
			// presentKHR is skipped on purpose: the swapchain image's layout was never
			// transitioned to ePresentSrcKHR (the graph never ran), so presenting it now would be
			// invalid -- this frame is simply dropped, not shown with stale/undefined content.
			vk::CommandBufferSubmitInfo cmdSubmitInfo{};
			cmdSubmitInfo.setCommandBuffer(frame.commandBuffer);

			vk::SemaphoreSubmitInfo waitInfo{};
			waitInfo.setSemaphore(frame.swapchainSemaphore);
			waitInfo.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

			vk::SemaphoreSubmitInfo frameTimelineSignalInfo{};
			frameTimelineSignalInfo.setSemaphore(frameTimelineSemaphore);
			frameTimelineSignalInfo.setValue(frameNumber + 1);
			frameTimelineSignalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

			vk::SubmitInfo2 recoverySubmitInfo{};
			recoverySubmitInfo.setWaitSemaphoreInfos(waitInfo);
			recoverySubmitInfo.setSignalSemaphoreInfos(frameTimelineSignalInfo);
			recoverySubmitInfo.setCommandBufferInfos(cmdSubmitInfo);
			graphicsQueue.submit2(recoverySubmitInfo, nullptr);

			frameNumber++;
			return;
		}

		// Transition swapchain image layout to PRESENT_SRC_KHR for presentation. oldLayout/
		// srcStage/srcAccess now come from the registry's tracked state rather than being
		// hardcoded -- DeferredNode's Modify<Swapchain> always leaves it at exactly
		// {eColorAttachmentOptimal, eColorAttachmentOutput, eColorAttachmentWrite|Read} today
		// (see tests/test_resource_state.cpp's swapchain-chain case), but reading it instead of
		// assuming it means this stays correct the day a different node becomes the last writer.
		auto                    swapchainTex = physicalRegistry.GetTexture<Swapchain>();
		vk::ImageMemoryBarrier2 presentBarrier{};
		presentBarrier.setSrcStageMask(
			swapchainTex ? swapchainTex->GetLastStage() : vk::PipelineStageFlagBits2::eColorAttachmentOutput
		);
		presentBarrier.setSrcAccessMask(
			swapchainTex ? swapchainTex->GetLastAccess() : vk::AccessFlagBits2::eColorAttachmentWrite
		);
		presentBarrier.setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe);
		presentBarrier.setDstAccessMask(vk::AccessFlagBits2::eNone);
		presentBarrier.setOldLayout(
			swapchainTex ? swapchainTex->GetCurrentLayout() : vk::ImageLayout::eColorAttachmentOptimal
		);
		presentBarrier.setNewLayout(vk::ImageLayout::ePresentSrcKHR);
		presentBarrier.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		presentBarrier.setImage(swapchainImages[swapchainImageIndex]);

		vk::DependencyInfo presentDepInfo{};
		presentDepInfo.setImageMemoryBarriers(presentBarrier);
		frame.commandBuffer.pipelineBarrier2(presentDepInfo);

		frame.commandBuffer.end();

		// 4. Submit to GPU (Using Vulkan 1.4 / Sync2 API)
		vk::CommandBufferSubmitInfo cmdSubmitInfo{};
		cmdSubmitInfo.setCommandBuffer(frame.commandBuffer);

		auto waitInfos = terrainUploader.GetWaitSemaphores();

		vk::SemaphoreSubmitInfo waitInfo{};
		waitInfo.setSemaphore(frame.swapchainSemaphore);
		waitInfo.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

		waitInfos.push_back(waitInfo);

		std::vector<vk::SemaphoreSubmitInfo> signalInfos;

		vk::SemaphoreSubmitInfo renderSignalInfo{};
		renderSignalInfo.setSemaphore(swapchainRenderSemaphores[swapchainImageIndex]);
		renderSignalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllGraphics);
		signalInfos.push_back(renderSignalInfo);

		vk::SemaphoreSubmitInfo frameTimelineSignalInfo{};
		frameTimelineSignalInfo.setSemaphore(frameTimelineSemaphore);
		frameTimelineSignalInfo.setValue(frameNumber + 1);
		frameTimelineSignalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllGraphics);
		signalInfos.push_back(frameTimelineSignalInfo);

		vk::SubmitInfo2 submitInfo{};
		submitInfo.setWaitSemaphoreInfos(waitInfos);
		submitInfo.setSignalSemaphoreInfos(signalInfos);
		submitInfo.setCommandBufferInfos(cmdSubmitInfo);

		graphicsQueue.submit2(submitInfo, nullptr);

		// 5. Present
		vk::PresentInfoKHR presentInfo{};
		presentInfo.setWaitSemaphores(swapchainRenderSemaphores[swapchainImageIndex]);
		vk::SwapchainKHR swapchain = vkbSwapchain.swapchain;
		presentInfo.setSwapchains(swapchain);
		presentInfo.setImageIndices(swapchainImageIndex);

		vk::Result presentResult = graphicsQueue.presentKHR(presentInfo);
		if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
		    windowResized) {
			windowResized = false;
			RecreateSwapchain();
		}

		frameNumber++;
	}

	void Engine::InitGlobalUBO() {
		// 1. Set 0 Layout
		vk::DescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.setBinding(0);
		layoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		layoutBinding.setDescriptorCount(1);
		layoutBinding.setStageFlags(vk::ShaderStageFlagBits::eAll);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(layoutBinding);
		globalSet0Layout = device.createDescriptorSetLayout(layoutInfo);

		// 2. Descriptor Pool
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eUniformBuffer);
		poolSize.setDescriptorCount(FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSize);
		globalDescriptorPool = device.createDescriptorPool(poolInfo);

		// 3. Allocate Descriptor Sets & UBO Buffers
		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, globalSet0Layout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
		allocInfo.setDescriptorPool(globalDescriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < FRAME_OVERLAP; i++) {
			globalDescriptorSets[i] = allocatedSets[i];

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = sizeof(FrameUBO);
			bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          buffer = VK_NULL_HANDLE;
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(
					allocator,
					&bufferInfo,
					&allocCreateInfo,
					&buffer,
					&globalUboAllocations[i],
					&allocResultInfo
				) != VK_SUCCESS) {
				spdlog::error("Failed to create Global UBO buffer with VMA");
				return;
			}

			globalUboBuffers[i] = buffer;
			globalUboMapped[i] = allocResultInfo.pMappedData;

			vk::DescriptorBufferInfo bufferDescInfo{};
			bufferDescInfo.setBuffer(globalUboBuffers[i]);
			bufferDescInfo.setOffset(0);
			bufferDescInfo.setRange(sizeof(FrameUBO));

			vk::WriteDescriptorSet descriptorWrite{};
			descriptorWrite.setDstSet(globalDescriptorSets[i]);
			descriptorWrite.setDstBinding(0);
			descriptorWrite.setDstArrayElement(0);
			descriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
			descriptorWrite.setBufferInfo(bufferDescInfo);

			device.updateDescriptorSets(descriptorWrite, nullptr);
		}
	}

	void Engine::CleanupGlobalUBO() {
		for (size_t i = 0; i < FRAME_OVERLAP; i++) {
			if (globalUboBuffers[i] && globalUboAllocations[i]) {
				vmaDestroyBuffer(allocator, globalUboBuffers[i], globalUboAllocations[i]);
				globalUboBuffers[i] = nullptr;
				globalUboAllocations[i] = nullptr;
				globalUboMapped[i] = nullptr;
			}
		}

		if (globalDescriptorPool) {
			device.destroyDescriptorPool(globalDescriptorPool);
			globalDescriptorPool = nullptr;
		}

		if (globalSet0Layout) {
			device.destroyDescriptorSetLayout(globalSet0Layout);
			globalSet0Layout = nullptr;
		}
	}

} // namespace brassica
