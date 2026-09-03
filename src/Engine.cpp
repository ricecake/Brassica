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
		swapchainImages = vkbSwapchain.get_images().value();
		swapchainImageViews = vkbSwapchain.get_image_views().value();
	}

	void Engine::InitCommands() {
		VkCommandPoolCreateInfo commandPoolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		commandPoolCreateInfo.queueFamilyIndex = graphicsQueueFamily;
		commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &frames[i].commandPool);

			VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
			cmdAllocInfo.commandPool = frames[i].commandPool;
			cmdAllocInfo.commandBufferCount = 1;
			cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

			vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].commandBuffer);
		}
	}

	void Engine::InitSyncStructures() {
		VkFenceCreateInfo fenceCreateInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		VkSemaphoreCreateInfo semaphoreCreateInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].renderFence);
			vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].swapchainSemaphore);
			vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].renderSemaphore);
		}
	}

	void Engine::Cleanup() {
		if (device) {
			vkDeviceWaitIdle(device);

			if (gradientPass) {
				gradientPass->DestroyPipeline(device);
				gradientPass.reset();
			}

			for (int i = 0; i < FRAME_OVERLAP; i++) {
				vkDestroyFence(device, frames[i].renderFence, nullptr);
				vkDestroySemaphore(device, frames[i].swapchainSemaphore, nullptr);
				vkDestroySemaphore(device, frames[i].renderSemaphore, nullptr);
				vkDestroyCommandPool(device, frames[i].commandPool, nullptr);
			}

			for (auto view : swapchainImageViews) {
				vkDestroyImageView(device, view, nullptr);
			}
			vkb::destroy_swapchain(vkbSwapchain);

			vkDestroyDevice(device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
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

		gradientPass = std::make_unique<GradientPass>(device, vkbSwapchain.image_format);

		taskScheduler.Initialize();
		spdlog::info("Brassica Engine Initialized.");
	}

	void Engine::InitVulkan() {
		// 1. Instance (Require Vulkan 1.3)
		vkb::InstanceBuilder builder;
		auto                 inst_ret = builder.set_app_name("Brassica")
											.request_validation_layers(true)
											.require_api_version(1, 3, 0)
											.use_default_debug_messenger()
											.build();
		vkbInst = inst_ret.value();
		instance = vkbInst.instance;

		glfwCreateWindowSurface(instance, window, nullptr, &surface);

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
												   .set_minimum_version(1, 3)
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
		FrameData& frame = GetCurrentFrame();

		// 1. Wait for GPU to finish the last time this frame context was used
		vkWaitForFences(device, 1, &frame.renderFence, VK_TRUE, 1000000000);
		vkResetFences(device, 1, &frame.renderFence);

		// 2. Acquire Swapchain Image
		uint32_t swapchainImageIndex;
		vkAcquireNextImageKHR(
			device,
			vkbSwapchain.swapchain,
			1000000000,
			frame.swapchainSemaphore,
			VK_NULL_HANDLE,
			&swapchainImageIndex
		);

		// 3. Record Commands
		vkResetCommandBuffer(frame.commandBuffer, 0);
		VkCommandBufferBeginInfo cmdBeginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(frame.commandBuffer, &cmdBeginInfo);

		// FrameGraph Setup and Execution
		FrameGraph        fg;
		FrameGraphTexture swapchainTexWrapper{
			swapchainImages[swapchainImageIndex],
			swapchainImageViews[swapchainImageIndex]
		};
		FrameGraphResource swapchainRes = fg.import(
			"SwapchainImage",
			{vkbSwapchain.extent, vkbSwapchain.image_format},
			std::move(swapchainTexWrapper)
		);

		gradientPass->RegisterPass(fg, swapchainRes, vkbSwapchain.extent);

		fg.compile();
		fg.execute(&frame.commandBuffer);

		vkEndCommandBuffer(frame.commandBuffer);

		// 4. Submit to GPU (Using Vulkan 1.3 Sync2 API)
		VkCommandBufferSubmitInfo cmdSubmitInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
		cmdSubmitInfo.commandBuffer = frame.commandBuffer;

		VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
		waitInfo.semaphore = frame.swapchainSemaphore;
		waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;

		VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
		signalInfo.semaphore = frame.renderSemaphore;
		signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

		VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
		submitInfo.waitSemaphoreInfoCount = 1;
		submitInfo.pWaitSemaphoreInfos = &waitInfo;
		submitInfo.signalSemaphoreInfoCount = 1;
		submitInfo.pSignalSemaphoreInfos = &signalInfo;
		submitInfo.commandBufferInfoCount = 1;
		submitInfo.pCommandBufferInfos = &cmdSubmitInfo;

		vkQueueSubmit2(graphicsQueue, 1, &submitInfo, frame.renderFence);

		// 5. Present
		VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &frame.renderSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &vkbSwapchain.swapchain;
		presentInfo.pImageIndices = &swapchainImageIndex;

		vkQueuePresentKHR(graphicsQueue, &presentInfo);

		frameNumber++;
	}

} // namespace brassica