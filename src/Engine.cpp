#include "Engine.hpp"

#include <cstring>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include "fg/Blackboard.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	VKAPI_ATTR VkBool32 VKAPI_CALL Engine::VulkanDebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT             messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void*                                       pUserData
	) {
		auto* engine = static_cast<Engine*>(pUserData);

		if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
			spdlog::error("[Vulkan Validation Error] {}", pCallbackData->pMessage);
			if (engine) engine->validationErrorCount++;
		} else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
			spdlog::warn("[Vulkan Validation Warning] {}", pCallbackData->pMessage);
			if (engine) engine->validationWarningCount++;
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
	}

	void Engine::InitSwapchain() {
		vkb::SwapchainBuilder swapchainBuilder{chosenGPU, device, surface};
		auto                  swap_ret = swapchainBuilder
											 .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
											 .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
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
			if (sem) device.destroySemaphore(sem);
		}
		swapchainRenderSemaphores.clear();

		vk::SemaphoreCreateInfo semaphoreCreateInfo{};
		for (size_t i = 0; i < swapchainImages.size(); i++) {
			swapchainRenderSemaphores.push_back(device.createSemaphore(semaphoreCreateInfo));
		}
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
		vk::FenceCreateInfo     fenceCreateInfo{vk::FenceCreateFlagBits::eSignaled};
		vk::SemaphoreCreateInfo semaphoreCreateInfo{};

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			frames[i].renderFence = device.createFence(fenceCreateInfo);
			frames[i].swapchainSemaphore = device.createSemaphore(semaphoreCreateInfo);
		}
	}

	void Engine::Cleanup() {
		if (device) {
			device.waitIdle();

			shaderWatcher.StopWatching();

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

			if (meshCubePass) {
				meshCubePass->DestroyPipeline();
				meshCubePass.reset();
			}

			if (gradientPass) {
				gradientPass->DestroyPipeline();
				gradientPass.reset();
			}

			CleanupGlobalUBO();

			for (int i = 0; i < FRAME_OVERLAP; i++) {
				device.destroyFence(frames[i].renderFence);
				device.destroySemaphore(frames[i].swapchainSemaphore);
				device.destroyCommandPool(frames[i].commandPool);
			}

			for (auto sem : swapchainRenderSemaphores) {
				if (sem) device.destroySemaphore(sem);
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

		meshCubePass = std::make_unique<MeshCubePass>(instance, device, globalSet0Layout, &shaderWatcher);
		gradientPass = std::make_unique<GradientPass>(device, vk::Format::eR16G16B16A16Sfloat, &shaderWatcher);
		terrainPass = std::make_unique<TerrainPass>(instance, device, globalSet0Layout, &shaderWatcher);
		deferredPass = std::make_unique<DeferredPass>(device, globalSet0Layout, GetSwapchainFormat(), &shaderWatcher);

		terrainClipmap.Init(device, allocator, 4, 0.5f);
		terrainUploader.Init(device, allocator, graphicsQueueFamily, 8);

		// Async upload initial heightmaps
		for (uint32_t l = 0; l < terrainClipmap.GetNumLODs(); ++l) {
			auto mapData = TerrainClipmap::GenerateSineWaveMap(l, terrainClipmap.GetBaseTexelSize());
			terrainUploader.UploadLevelAsync(l, mapData, terrainClipmap.GetImage(), TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, graphicsQueue);
		}
		terrainPass->UpdateClipmapDescriptor(terrainClipmap.GetImageView(), terrainClipmap.GetSampler());

		taskScheduler.Initialize();
		spdlog::info("Brassica Engine Initialized (headless: {}).", options.headless);
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
			builder.enable_extension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
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
			VkResult res = vkCreateHeadlessSurfaceEXT(instance, &createInfo, nullptr, &c_surface);
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

		VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
		meshFeatures.meshShader = VK_TRUE;
		meshFeatures.taskShader = VK_TRUE;

		vkb::PhysicalDeviceSelector selector{vkbInst};
		selector.set_surface(surface)
			.set_minimum_version(chosenMajor, chosenMinor)
			.set_required_features_13(features13)
			.set_required_features_12(features12)
			.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
			.add_required_extension_features(meshFeatures);

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

		if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
			spdlog::critical("Failed to create Vulkan Memory Allocator.");
			return false;
		}

		return true;
	}

	void Engine::Run() {
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

			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}

			// Wrap the frame in an enkiTS task so the main thread remains free
			// for OS event pumping and window resizing.
			enki::TaskSet frameTask(1, [this](enki::TaskSetPartition range, uint32_t threadnum) { DrawFrame(); });

			taskScheduler.AddTaskSetToPipe(&frameTask);
			taskScheduler.WaitforTask(&frameTask);
		}
	}

	void Engine::DrawFrame() {
		shaderWatcher.ProcessPendingReloads(device);

		FrameData& frame = GetCurrentFrame();

		// 1. Wait for GPU to finish the last time this frame context was used
		(void)device.waitForFences(frame.renderFence, VK_TRUE, 1000000000);
		device.resetFences(frame.renderFence);

		// 2. Acquire Swapchain Image
		auto acquireResult = device.acquireNextImageKHR(vkbSwapchain.swapchain, 1000000000, frame.swapchainSemaphore);
		uint32_t swapchainImageIndex = acquireResult.value;

		// 3. Record Commands
		frame.commandBuffer.reset();
		vk::CommandBufferBeginInfo cmdBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		frame.commandBuffer.begin(cmdBeginInfo);

		// FrameGraph Setup and Execution
		FrameGraph           fg;
		FrameGraphBlackboard blackboard;
		FrameGraphTexture2D  swapchainTexWrapper{
			swapchainImages[swapchainImageIndex],
			swapchainImageViews[swapchainImageIndex]
		};

		vk::Extent2D extent{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		vk::Format   format = GetSwapchainFormat();

		FrameGraphResource swapchainRes = fg.import("SwapchainImage", {extent, format}, std::move(swapchainTexWrapper));
		blackboard.add<SwapchainData>() = SwapchainData{.target = swapchainRes};

		uint32_t activeFrame = frameNumber % FRAME_OVERLAP;

		FrameUBO ubo{};
		ubo.time = static_cast<float>(glfwGetTime());
		ubo.frameIndex = frameNumber;
		ubo.globalSeed = globalSeed;
		ubo.frameRandom = static_cast<uint32_t>(rng());

		if (globalUboMapped[activeFrame]) {
			std::memcpy(globalUboMapped[activeFrame], &ubo, sizeof(FrameUBO));
		}

		gradientPass->RegisterPass(fg, blackboard, extent, allocator);

		terrainUploader.Poll();

		float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
		glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 500.0f);
		proj[1][1] *= -1.0f; // Vulkan inverted Y
		glm::vec3 camPos(0.0f, 15.0f, 30.0f);
		glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		TerrainPushConstants terrainPush{};
		terrainPush.viewProj = proj * view;
		terrainPush.cameraPos = glm::vec4(camPos, terrainClipmap.GetBaseTexelSize());
		terrainPush.gridParams = glm::uvec4(terrainClipmap.GetNumLODs(), 16, 256, 0);

		terrainPass->RegisterPass(fg, blackboard, extent, globalDescriptorSets[activeFrame], terrainPush, allocator);
		deferredPass->RegisterPass(fg, blackboard, extent, globalDescriptorSets[activeFrame], activeFrame);

		RenderContext renderCtx{
			.commandBuffer = frame.commandBuffer,
			.allocator = allocator,
			.device = device
		};

		fg.compile();
		vk::CommandBuffer rawCmd = frame.commandBuffer;
		fg.execute(&rawCmd, &renderCtx);

		// Transition swapchain image layout to PRESENT_SRC_KHR for presentation
		vk::ImageMemoryBarrier2 presentBarrier{};
		presentBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
		presentBarrier.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite);
		presentBarrier.setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe);
		presentBarrier.setDstAccessMask(vk::AccessFlagBits2::eNone);
		presentBarrier.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal);
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

		vk::SemaphoreSubmitInfo waitInfo{};
		waitInfo.setSemaphore(frame.swapchainSemaphore);
		waitInfo.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

		vk::SemaphoreSubmitInfo signalInfo{};
		signalInfo.setSemaphore(swapchainRenderSemaphores[swapchainImageIndex]);
		signalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllGraphics);

		vk::SubmitInfo2 submitInfo{};
		submitInfo.setWaitSemaphoreInfos(waitInfo);
		submitInfo.setSignalSemaphoreInfos(signalInfo);
		submitInfo.setCommandBufferInfos(cmdSubmitInfo);

		graphicsQueue.submit2(submitInfo, frame.renderFence);

		// 5. Present
		vk::PresentInfoKHR presentInfo{};
		presentInfo.setWaitSemaphores(swapchainRenderSemaphores[swapchainImageIndex]);
		vk::SwapchainKHR swapchain = vkbSwapchain.swapchain;
		presentInfo.setSwapchains(swapchain);
		presentInfo.setImageIndices(swapchainImageIndex);

		(void)graphicsQueue.presentKHR(presentInfo);

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
		vk::DescriptorSetAllocateInfo allocInfo{};
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
			allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

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
