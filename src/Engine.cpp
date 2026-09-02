#include "Engine.hpp"

#include "spdlog/spdlog.h"

namespace brassica {

	void Engine::Init() {
		InitWindow();
		InitVulkan();
		InitSwapchain();
		InitCommands();
		InitSyncStructures();

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

		// --- YOUR RENDER GRAPH GOES HERE ---
		// Example: Transition swapchain image to color attachment optimal via Sync2
		// Example: vkCmdBeginRendering (Dynamic Rendering)
		// Example: vkCmdDrawIndexedIndirect
		// Example: vkCmdEndRendering
		// Example: Transition swapchain image to present source via Sync2

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