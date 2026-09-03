#include "Engine.hpp"

#include "spdlog/spdlog.h"

namespace brassica {

	void Engine::InitWindow() {
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

		swapchainImages.clear();
		for (VkImage img : vkbSwapchain.get_images().value()) {
			swapchainImages.push_back(img);
		}

		swapchainImageViews.clear();
		for (VkImageView view : vkbSwapchain.get_image_views().value()) {
			swapchainImageViews.push_back(view);
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
		vk::FenceCreateInfo fenceCreateInfo{vk::FenceCreateFlagBits::eSignaled};
		vk::SemaphoreCreateInfo semaphoreCreateInfo{};

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			frames[i].renderFence = device.createFence(fenceCreateInfo);
			frames[i].swapchainSemaphore = device.createSemaphore(semaphoreCreateInfo);
			frames[i].renderSemaphore = device.createSemaphore(semaphoreCreateInfo);
		}
	}

	void Engine::Cleanup() {
		if (device) {
			device.waitIdle();

			shaderWatcher.StopWatching();

			if (gradientPass) {
				gradientPass->DestroyPipeline(device);
				gradientPass.reset();
			}

			for (int i = 0; i < FRAME_OVERLAP; i++) {
				device.destroyFence(frames[i].renderFence);
				device.destroySemaphore(frames[i].swapchainSemaphore);
				device.destroySemaphore(frames[i].renderSemaphore);
				device.destroyCommandPool(frames[i].commandPool);
			}

			for (auto view : swapchainImageViews) {
				device.destroyImageView(view);
			}
			vkb::destroy_swapchain(vkbSwapchain);

			device.destroy();
			instance.destroySurfaceKHR(surface);
			vkb::destroy_instance(vkbInst);
		}

		if (window) {
			glfwDestroyWindow(window);
			glfwTerminate();
		}
	}

	void Engine::Init() {
		InitWindow();
		InitVulkan();
		InitSwapchain();
		InitCommands();
		InitSyncStructures();

		shaderWatcher.WatchDirectory("shaders");

		gradientPass = std::make_unique<GradientPass>(device, GetSwapchainFormat(), &shaderWatcher);

		taskScheduler.Initialize();
		spdlog::info("Brassica Engine Initialized.");
	}

	void Engine::InitVulkan() {
		// 1. Instance (Require Vulkan 1.4)
		vkb::InstanceBuilder builder;
		builder.set_app_name("Brassica")
			.require_api_version(1, 4, 0)
			.use_default_debug_messenger();

		auto inst_res = builder.request_validation_layers(true).build();
		vkbInst = inst_res ? inst_res.value() : builder.request_validation_layers(false).build().value();
		instance = vkbInst.instance;

		VkSurfaceKHR c_surface;
		glfwCreateWindowSurface(instance, window, nullptr, &c_surface);
		surface = c_surface;

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

		vkb::PhysicalDeviceSelector selector{vkbInst};
		auto                        phys_ret = selector.set_surface(surface)
												   .set_minimum_version(1, 4)
												   .add_required_extension_features(features13)
												   .add_required_extension_features(features12)
												   .select();
		chosenGPU = phys_ret.value().physical_device;

		// 3. Logical Device
		vkb::DeviceBuilder deviceBuilder{phys_ret.value()};
		auto               dev_ret = deviceBuilder.build();
		vkbDevice = dev_ret.value();
		device = vkbDevice.device;

		graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
		graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
	}

	void Engine::Run() {
		while (!glfwWindowShouldClose(window)) {
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
		FrameGraph        fg;
		FrameGraphTexture swapchainTexWrapper{
			swapchainImages[swapchainImageIndex],
			swapchainImageViews[swapchainImageIndex]
		};

		vk::Extent2D extent{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
		vk::Format   format = GetSwapchainFormat();

		FrameGraphResource swapchainRes = fg.import(
			"SwapchainImage",
			{extent, format},
			std::move(swapchainTexWrapper)
		);

		gradientPass->RegisterPass(fg, swapchainRes, extent);

		fg.compile();
		vk::CommandBuffer rawCmd = frame.commandBuffer;
		fg.execute(&rawCmd);

		frame.commandBuffer.end();

		// 4. Submit to GPU (Using Vulkan 1.4 / Sync2 API)
		vk::CommandBufferSubmitInfo cmdSubmitInfo{};
		cmdSubmitInfo.setCommandBuffer(frame.commandBuffer);

		vk::SemaphoreSubmitInfo waitInfo{};
		waitInfo.setSemaphore(frame.swapchainSemaphore);
		waitInfo.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

		vk::SemaphoreSubmitInfo signalInfo{};
		signalInfo.setSemaphore(frame.renderSemaphore);
		signalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllGraphics);

		vk::SubmitInfo2 submitInfo{};
		submitInfo.setWaitSemaphoreInfos(waitInfo);
		submitInfo.setSignalSemaphoreInfos(signalInfo);
		submitInfo.setCommandBufferInfos(cmdSubmitInfo);

		graphicsQueue.submit2(submitInfo, frame.renderFence);

		// 5. Present
		vk::PresentInfoKHR presentInfo{};
		presentInfo.setWaitSemaphores(frame.renderSemaphore);
		vk::SwapchainKHR swapchain = vkbSwapchain.swapchain;
		presentInfo.setSwapchains(swapchain);
		presentInfo.setImageIndices(swapchainImageIndex);

		(void)graphicsQueue.presentKHR(presentInfo);

		frameNumber++;
	}

} // namespace brassica
