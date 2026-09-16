#include "ImGuiManager.hpp"

#include "spdlog/spdlog.h"

#include "GLFW/glfw3.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace brassica {

	ImGuiManager::~ImGuiManager() {
		if (m_initialized) {
			Shutdown();
		}
	}

	void ImGuiManager::Initialize() {
		if (m_initialized)
			return;

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.IniFilename = "imgui.ini"; // Initialize ImGui config system to store values

		ImGui::StyleColorsDark();

		m_initialized = true;
	}

	void ImGuiManager::InitVulkanAndGlfw(
		GLFWwindow*        window,
		vk::Instance       instance,
		vk::PhysicalDevice physicalDevice,
		vk::Device         device,
		uint32_t           queueFamily,
		vk::Queue          queue,
		vk::Format         swapchainFormat,
		uint32_t           imageCount
	) {
		Initialize();

		m_window = window;
		m_device = device;

		// Create descriptor pool for ImGui
		vk::DescriptorPoolSize poolSizes[] = {
			{vk::DescriptorType::eCombinedImageSampler, 100},
			{vk::DescriptorType::eSampledImage, 100},
			{vk::DescriptorType::eSampler, 100},
			{vk::DescriptorType::eUniformBuffer, 100},
			{vk::DescriptorType::eStorageBuffer, 100}
		};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		poolInfo.setMaxSets(500);
		poolInfo.setPoolSizes(poolSizes);

		try {
			m_descriptorPool = device.createDescriptorPool(poolInfo);
		} catch (const vk::SystemError& err) {
			spdlog::error("Failed to create ImGui descriptor pool: {}", err.what());
			return;
		}

		if (m_window) {
			ImGui_ImplGlfw_InitForVulkan(m_window, true);
		}

		ImGui_ImplVulkan_InitInfo initInfo{};
		initInfo.ApiVersion = VK_API_VERSION_1_3;
		initInfo.Instance = instance;
		initInfo.PhysicalDevice = physicalDevice;
		initInfo.Device = device;
		initInfo.QueueFamily = queueFamily;
		initInfo.Queue = queue;
		initInfo.DescriptorPool = m_descriptorPool;
		initInfo.MinImageCount = 2;
		initInfo.ImageCount = imageCount;
		initInfo.UseDynamicRendering = true;
		initInfo.PipelineInfoMain.PipelineRenderingCreateInfo
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
		initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
		VkFormat colorFormat = static_cast<VkFormat>(swapchainFormat);
		initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;

		if (!ImGui_ImplVulkan_Init(&initInfo)) {
			spdlog::error("ImGui_ImplVulkan_Init failed!");
			return;
		}

		m_vkInitialized = true;
		spdlog::info("ImGui layer initialized successfully.");
	}

	void ImGuiManager::Shutdown() {
		if (m_vkInitialized) {
			if (m_device) {
				m_device.waitIdle();
			}
			ImGui_ImplVulkan_Shutdown();
			if (m_window) {
				ImGui_ImplGlfw_Shutdown();
			}
			m_vkInitialized = false;
		}

		if (ImGui::GetCurrentContext()) {
			ImGui::DestroyContext();
		}

		if (m_device && m_descriptorPool) {
			m_device.destroyDescriptorPool(m_descriptorPool);
			m_descriptorPool = nullptr;
		}

		m_initialized = false;
	}

	void ImGuiManager::OnKey(int key, int action) {
		if ((key == GLFW_KEY_GRAVE_ACCENT || key == 96) && action == GLFW_PRESS) {
			m_visible = !m_visible;
			spdlog::info("ImGui overlay visibility toggled: {}", m_visible);
		}
	}

	void ImGuiManager::NewFrame(const glm::vec3& cameraPosition) {
		if (!m_initialized)
			return;

		if (m_vkInitialized) {
			ImGui_ImplVulkan_NewFrame();
			if (m_window) {
				ImGui_ImplGlfw_NewFrame();
			}
		}

		ImGui::NewFrame();

		if (m_visible) {
			ImGuiIO&       io = ImGui::GetIO();
			ImGuiViewport* viewport = ImGui::GetMainViewport();

			ImGui::SetNextWindowPos(
				ImVec2(
					viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
					viewport->WorkPos.y + viewport->WorkSize.y - 10.0f
				),
				ImGuiCond_Always,
				ImVec2(0.5f, 1.0f)
			);
			ImGui::SetNextWindowBgAlpha(0.6f);

			ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
				ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

			if (ImGui::Begin("Brassica Minimal Overlay", nullptr, flags)) {
				float fps = io.Framerate;
				float frameTime = fps > 0.0f ? (1000.0f / fps) : 0.0f;
				ImGui::Text(
					"Location: (%.2f, %.2f, %.2f) | %.1f FPS (%.2f ms)",
					cameraPosition.x,
					cameraPosition.y,
					cameraPosition.z,
					fps,
					frameTime
				);
			}
			ImGui::End();
		}
	}

	void ImGuiManager::Render(vk::CommandBuffer cmd) {
		ImGui::Render();
		if (m_vkInitialized && cmd) {
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		}
	}

} // namespace brassica
