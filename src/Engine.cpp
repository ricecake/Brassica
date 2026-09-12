#include "Engine.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

#include "spdlog/spdlog.h"

#include "graph/PhysicalExecutionBackend.hpp"
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

			terrainAS.DestroyAccelerationStructures();

			terrainUploader.Cleanup();
			terrainClipmap.Cleanup();

			pipelineLibrary.Reset();
			gradientNode.Destroy(device);
			terrainNode.Destroy(device);
			deferredNode.Destroy(device);
			waterNode.Destroy(device);
			transmittanceNode.Destroy(device);
			multiScatteringNode.Destroy(device);

			CleanupGlobalUBO();
			CleanupGlobalDescriptors();

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
		InitGlobalDescriptors();

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

		terrainAS.Init(instance, device);

		gradientNode.Init(device, &pipelineLibrary, &shaderWatcher);
		terrainNode.Init(device, &pipelineLibrary, &terrainAS, &shaderWatcher);
		deferredNode.Init(device, &pipelineLibrary, GetSwapchainFormat(), &shaderWatcher);
		waterNode.Init(device, &pipelineLibrary, &terrainAS.GetDls(), GetSwapchainFormat(), &shaderWatcher);
		transmittanceNode.Init(device, &pipelineLibrary, &shaderWatcher);
		multiScatteringNode.Init(device, &pipelineLibrary, &shaderWatcher);

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

		// Registered once, here -- the clipmap's image/view handles are stable for the engine's
		// entire lifetime (only its *contents* mutate, via terrainUploader), so re-registering it
		// every frame would just churn a fresh bindless index for no reason (RegisterImportedTexture
		// has no desc-match reuse the way ProvisionTexture does for owned resources). Real state,
		// not the eUndefined/false defaults: the uploader always leaves the image in
		// eShaderReadOnlyOptimal by the time its timeline semaphore signals (AsyncTerrainUploader.cpp),
		// and DrawFrame's submission already waits on that semaphore before this image is ever
		// touched -- see RegisterImportedTexture's own comment on why a long-lived import must pass
		// its real state.
		physicalRegistry.RegisterImportedTexture<TerrainClipmapTexture>(
			terrainClipmap.GetImage(),
			terrainClipmap.GetImageView(),
			TerrainClipmapDesc(terrainClipmap.GetNumLODs()),
			vk::ImageLayout::eShaderReadOnlyOptimal,
			/*hasDefinedContents=*/true
		);

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

		// Bindless set 0 (PhysicalRegistry.hpp's bindless index machinery, Engine::InitGlobalDescriptors):
		// runtimeDescriptorArray + shaderSampledImageArrayNonUniformIndexing/
		// shaderStorageImageArrayNonUniformIndexing let a shader index an unsized
		// texture2D[]/image2D[] with nonuniformEXT; the two UpdateAfterBind bits let the registry
		// write a new texture's descriptor without invalidating command buffers that reference the
		// same set but a different index. Deliberately NOT requesting
		// descriptorBindingAccelerationStructureUpdateAfterBind (the spottiest of this family) or
		// descriptorBindingVariableDescriptorCount (unused -- see ShadingRateAttachmentDesc-style
		// fixed-size arrays in InitGlobalDescriptors).
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

		// Bindless sampled-image array size: clamp to a generous fixed default rather than the
		// device's real (often absurdly high, e.g. 1000000+ on this Mac's MoltenVK) limit --
		// descriptorBindingVariableDescriptorCount is deliberately not used (PhysicalRegistry.hpp),
		// so this is a real, if small, chunk of descriptor memory reserved up front regardless of
		// how many textures actually exist. Logged so a future device swap with a *lower* limit
		// than 4096 doesn't silently truncate.
		{
			auto chain =
				chosenGPU
					.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>();
			const auto& descIndexingProps = chain.get<vk::PhysicalDeviceDescriptorIndexingProperties>();
			maxBindlessSampledImages = std::min<std::uint32_t>(
				4096,
				descIndexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages
			);
			spdlog::info(
				"Bindless sampled-image array size: {} (device max {})",
				maxBindlessSampledImages,
				descIndexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages
			);
		}

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
		pipelineLibrary.SetDeviceAndCache(device, pipelineCache);

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
		//
		// Unlike an Owning resource (where usageMask drives real image creation and is therefore
		// always accurate), an Imported one's desc is just metadata describing an image this
		// registry didn't create -- it must match what the image was *actually* created with, not
		// what would be generically convenient. ColorAttachmentDesc() claims eSampled (correct for
		// G-buffer-style targets the bindless array is meant to hold), but vk-bootstrap's
		// SwapchainBuilder here never requests VK_IMAGE_USAGE_SAMPLED_BIT (Engine.cpp's swapchain
		// creation has no set_image_usage_flags call, so it's the vk-bootstrap default of
		// eColorAttachment | eTransferDst only) -- so claiming eSampled made
		// AssignAndWriteBindlessIndices try to write an invalid SAMPLED_IMAGE descriptor pointing
		// at a view that was never created with that usage. Stripped here rather than fixed at the
		// preset: every *other* ColorAttachmentDesc consumer is Owning and does want eSampled.
		graph::ResourceDesc swapchainDesc = graph::ColorAttachmentDesc(extent.width, extent.height, format);
		swapchainDesc.usageMask &= ~static_cast<std::uint32_t>(vk::ImageUsageFlagBits::eSampled);

		physicalRegistry.RegisterImportedTexture<Swapchain>(
			swapchainImages[swapchainImageIndex],
			swapchainImageViews[swapchainImageIndex],
			swapchainDesc,
			vk::ImageLayout::eUndefined,
			false,
			frameNumber
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
		ubo.viewMatrix = camera.viewMatrix;
		ubo.invViewMatrix = camera.invViewMatrix;
		ubo.projMatrix = camera.projMatrix;
		ubo.invProjMatrix = camera.invProjMatrix;
		ubo.viewProjMatrix = camera.viewProjMatrix;
		ubo.invViewProjMatrix = camera.invViewProjMatrix;
		ubo.cameraPosition = glm::vec4(camera.position, terrainClipmap.GetBaseTexelSize());
		ubo.time = static_cast<float>(currentTime);
		ubo.fov = camera.fov;
		ubo.aspectRatio = camera.aspectRatio;
		ubo.nearPlane = camera.nearPlane;
		ubo.farPlane = camera.farPlane;
		ubo.frameIndex = frameNumber;
		ubo.globalSeed = globalSeed;
		ubo.frameRandom = static_cast<uint32_t>(rng());

		if (globalUboMapped[activeFrame]) {
			std::memcpy(globalUboMapped[activeFrame], &ubo, sizeof(FrameUBO));
			vmaFlushAllocation(allocator, globalUboAllocations[activeFrame], 0, sizeof(FrameUBO));
		}

		graph::PhysicalResourceRegistry::BindlessBindings bindlessBindings{};
		bindlessBindings.set = globalDescriptorSets[activeFrame];
		bindlessBindings.sets.assign(globalDescriptorSets, globalDescriptorSets + FRAME_OVERLAP);
		bindlessBindings.layout = bindlessSetLayout;
		bindlessBindings.uboBinding = 0;
		bindlessBindings.sampledImage2DBinding = 1;
		bindlessBindings.sampledImage2DArrayBinding = 2;
		bindlessBindings.samplerBinding = 3;
		bindlessBindings.storageImageBinding = 4;
		bindlessBindings.accelerationStructureBinding = 5;
		physicalRegistry.SetGlobalDescriptorSet(bindlessBindings);

		terrainUploader.Poll();

		terrainClipmap.UpdateCameraPosition(camera.position, terrainUploader, graphicsQueue);

		uint32_t lods = terrainClipmap.GetNumLODs();
		uint32_t meshletsPerRow = 16;
		uint32_t totalMeshlets = lods * meshletsPerRow * meshletsPerRow;

		TerrainPushConstants terrainPush{};
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

		terrainAS.BuildOrUpdate(
			allocator,
			camera.position,
			terrainClipmap.GetBaseTexelSize(),
			terrainPush.gridParams.x
		);
		physicalRegistry.RegisterImportedAccelerationStructure<TerrainTLAS>(terrainAS.GetTLAS());

		terrainNode.SetFrameParams(terrainPush);
		deferredNode.SetFrameParams(DeferredPushConstants{
			.gridParams = terrainPush.gridParams,
			.lodOffsets0_3 = terrainPush.lodOffsets0_3,
			.lodOffsets4_7 = terrainPush.lodOffsets4_7,
		});
		waterNode.SetFrameParams(WaterPushConstants{
			.waterColor = glm::vec3(0.05f, 0.45f, 0.85f),
			.waterLevel = 0.0f,
		});

		graph::Graph frameGraph;
		frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
		frameGraph.RegisterRef(transmittanceNode);
		frameGraph.RegisterRef(multiScatteringNode);
		frameGraph.RegisterRef(gradientNode);
		frameGraph.RegisterRef(terrainNode);
		frameGraph.RegisterRef(deferredNode);
		frameGraph.RegisterRef(waterNode);

		graph::FrameContext             ctx{.width = extent.width, .height = extent.height, .frameIndex = frameNumber};
		graph::PhysicalExecutionBackend backend(physicalRegistry);
		graph::CommandBuffer            graphCmd{static_cast<void*>(static_cast<VkCommandBuffer>(frame.commandBuffer))};

		try {
			// enableAliasing=false: ImageAliasPool/BufferAliasPool's block bookkeeping
			// (PhysicalResource.hpp) records neither an occupant's identity nor its liveness, only
			// the schedule stage its lifetime last touched -- and that bookkeeping is never
			// refreshed for a key that keeps hitting ProvisionTexture's desc-match early return
			// (PhysicalRegistry.hpp), which every steady-state resource here does every frame. A
			// window resize (or any new/changed key) can then bind a fresh image onto a block a
			// live resource still occupies. See FRAME_GRAPH_MIGRATION_TODO.md for the full writeup;
			// this was already the migration plan's own recommendation before shipping true.
			backend.Execute(frameGraph, ctx, graphCmd, false);
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
		InitGlobalDescriptors();
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
	}

	void Engine::InitGlobalDescriptors() {
		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
		// Binding 0: FrameUBO
		bindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 1: uTextures2D
		bindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(maxBindlessSampledImages)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 2: uTextureArrays
		bindings[2]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(64)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 3: uSamplers
		bindings[3]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 4: uImagesRGBA32F
		bindings[4]
			.setBinding(4)
			.setDescriptorType(vk::DescriptorType::eStorageImage)
			.setDescriptorCount(256)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 5: uTLAS
		bindings[5]
			.setBinding(5)
			.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 6> bindingFlags{
			vk::DescriptorBindingFlags{},
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		bindlessSetLayout = device.createDescriptorSetLayout(layoutInfo);
		globalSet0Layout = bindlessSetLayout;

		std::array<vk::DescriptorPoolSize, 5> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, FRAME_OVERLAP},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, (maxBindlessSampledImages + 64) * FRAME_OVERLAP},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4 * FRAME_OVERLAP},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 256 * FRAME_OVERLAP},
			vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, 4 * FRAME_OVERLAP},
		};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		bindlessDescriptorPool = device.createDescriptorPool(poolInfo);
		globalDescriptorPool = bindlessDescriptorPool;

		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, bindlessSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
		allocInfo.setDescriptorPool(bindlessDescriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = device.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
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

			vk::WriteDescriptorSet uboWrite{};
			uboWrite.setDstSet(globalDescriptorSets[i]);
			uboWrite.setDstBinding(0);
			uboWrite.setDstArrayElement(0);
			uboWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
			uboWrite.setBufferInfo(bufferDescInfo);

			device.updateDescriptorSets(uboWrite, nullptr);
		}
		bindlessDescriptorSet = globalDescriptorSets[0];

		// Sampler catalog: 4 engine-owned entries
		auto makeSampler = [&](vk::Filter filter, vk::SamplerAddressMode addressMode, bool mipmap) {
			vk::SamplerCreateInfo info{};
			info.setMagFilter(filter)
				.setMinFilter(filter)
				.setAddressModeU(addressMode)
				.setAddressModeV(addressMode)
				.setAddressModeW(addressMode)
				.setMipmapMode(mipmap ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest)
				.setMinLod(0.0f)
				.setMaxLod(mipmap ? VK_LOD_CLAMP_NONE : 0.25f);
			return device.createSampler(info);
		};
		bindlessSamplers[0] = makeSampler(vk::Filter::eNearest, vk::SamplerAddressMode::eClampToEdge, false);
		bindlessSamplers[1] = makeSampler(vk::Filter::eLinear, vk::SamplerAddressMode::eClampToEdge, false);
		bindlessSamplers[2] = makeSampler(vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat, true);
		bindlessSamplers[3] = makeSampler(vk::Filter::eNearest, vk::SamplerAddressMode::eRepeat, false);

		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

		std::array<vk::DescriptorImageInfo, 4> samplerInfos{};
		for (std::uint32_t i = 0; i < 4; ++i) {
			samplerInfos[i].setSampler(bindlessSamplers[i]);
		}
		for (size_t i = 0; i < FRAME_OVERLAP; i++) {
			vk::WriteDescriptorSet samplerWrite{};
			samplerWrite.setDstSet(globalDescriptorSets[i]);
			samplerWrite.setDstBinding(3);
			samplerWrite.setDstArrayElement(0);
			samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
			samplerWrite.setImageInfo(samplerInfos);
			device.updateDescriptorSets(samplerWrite, {});
		}

		graph::PhysicalResourceRegistry::BindlessBindings bindlessBindings{};
		bindlessBindings.set = globalDescriptorSets[0];
		bindlessBindings.sets.assign(globalDescriptorSets, globalDescriptorSets + FRAME_OVERLAP);
		bindlessBindings.layout = bindlessSetLayout;
		bindlessBindings.uboBinding = 0;
		bindlessBindings.sampledImage2DBinding = 1;
		bindlessBindings.sampledImage2DArrayBinding = 2;
		bindlessBindings.samplerBinding = 3;
		bindlessBindings.storageImageBinding = 4;
		bindlessBindings.accelerationStructureBinding = 5;
		physicalRegistry.SetGlobalDescriptorSet(bindlessBindings);
	}

	void Engine::CleanupGlobalDescriptors() {
		for (vk::Sampler& sampler : bindlessSamplers) {
			if (sampler) {
				device.destroySampler(sampler);
				sampler = nullptr;
			}
		}
		if (bindlessDescriptorPool) {
			device.destroyDescriptorPool(bindlessDescriptorPool);
			bindlessDescriptorPool = nullptr;
			globalDescriptorPool = nullptr;
		}
		if (bindlessSetLayout) {
			device.destroyDescriptorSetLayout(bindlessSetLayout);
			bindlessSetLayout = nullptr;
			globalSet0Layout = nullptr;
		}
	}

} // namespace brassica
