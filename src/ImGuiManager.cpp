#include "ImGuiManager.hpp"

#include <algorithm>
#include <map>
#include <string>

#include "spdlog/spdlog.h"

#include "GLFW/glfw3.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "ui/QuickSettingsWidget.hpp"

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

		// Add default Quick Settings widget
		AddWidget(std::make_shared<ui::QuickSettingsWidget>());
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

	void ImGuiManager::AddWidget(std::shared_ptr<ui::IWidget> widget) {
		if (widget) {
			m_widgets.push_back(widget);
		}
	}

	void ImGuiManager::OnKey(int key, int action) {
		if ((key == GLFW_KEY_GRAVE_ACCENT || key == 96) && action == GLFW_PRESS) {
			m_visible = !m_visible;
			spdlog::info("ImGui overlay visibility toggled: {}", m_visible);
		}
	}

	void ImGuiManager::NewFrame(const CameraData& camera) {
		if (!m_initialized)
			return;

		if (m_vkInitialized) {
			ImGui_ImplVulkan_NewFrame();
			if (m_window) {
				ImGui_ImplGlfw_NewFrame();
			}
		}

		ImGui::NewFrame();

		bool anyHudVisible = std::any_of(m_widgets.begin(), m_widgets.end(), [](const auto& widget) {
			return widget->IsHud() && widget->IsVisible();
		});

		if (m_visible) {
			// Top level main menu bar
			if (ImGui::BeginMainMenuBar()) {
				// Group widgets by category
				std::map<std::string, std::vector<std::shared_ptr<ui::IWidget>>> categorizedWidgets;
				for (const auto& widget : m_widgets) {
					categorizedWidgets[std::string(widget->GetCategory())].push_back(widget);
				}

				for (auto& [category, widgets] : categorizedWidgets) {
					if (ImGui::BeginMenu(category.c_str())) {
						for (auto& widget : widgets) {
							bool        visible = widget->IsVisible();
							std::string titleStr(widget->GetTitle());
							if (ImGui::MenuItem(titleStr.c_str(), nullptr, &visible)) {
								widget->SetVisible(visible);
								if (visible) {
									ImGui::SetWindowCollapsed(titleStr.c_str(), false);
								}
							}
						}
						ImGui::EndMenu();
					}
				}

				ImGui::EndMainMenuBar();
			}

			// Render bottom HUD overlay (Location, Speed, Orientation, and FPS status)
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
				float pitchDeg = glm::degrees(camera.pitch);
				float yawDeg = glm::degrees(camera.yaw);
				float rollDeg = glm::degrees(camera.roll);

				ImGui::Text(
					"Location: (%.2f, %.2f, %.2f) | Speed: %.1f m/s | Rot: (P: %.1f deg, Y: %.1f deg, R: %.1f deg) | "
					"%.1f FPS (%.2f ms)",
					camera.position.x,
					camera.position.y,
					camera.position.z,
					camera.speed,
					pitchDeg,
					yawDeg,
					rollDeg,
					fps,
					frameTime
				);
			}
			ImGui::End();
		}

		// Draw registered widgets
		for (const auto& widget : m_widgets) {
			if (widget->IsHud() || m_visible) {
				widget->Draw();
			}
		}

		if (m_visible) {
			PositionMinimizedWindows();
		}
	}

	void ImGuiManager::NewFrame(const glm::vec3& cameraPosition) {
		CameraData dummyCam{};
		dummyCam.position = cameraPosition;
		NewFrame(dummyCam);
	}

	void ImGuiManager::PositionMinimizedWindows() {
		ImGuiContext& g = *GImGui;
		ImGuiIO&      io = ImGui::GetIO();

		const float padding = 10.0f;
		const float uniform_width = 250.0f;
		float       current_y = padding + 20.0f; // Below main menu bar

		std::vector<ImGuiWindow*> collapsed_windows;

		for (int i = 0; i < g.Windows.Size; i++) {
			ImGuiWindow* window = g.Windows[i];

			if (window->Hidden || (window->Flags & ImGuiWindowFlags_ChildWindow))
				continue;

			if (window->LastFrameActive < g.FrameCount)
				continue;

			if (window->Flags & ImGuiWindowFlags_Tooltip || window->Flags & ImGuiWindowFlags_Popup)
				continue;

			// Skip bottom overlay window from collapsed stack positioning
			if (std::string_view(window->Name).find("Brassica Minimal Overlay") != std::string_view::npos)
				continue;

			WindowState& state = m_windowStates[window->ID];

			if (window->Collapsed) {
				if (!state.was_collapsed) {
					state.last_expanded_pos = window->Pos;
					state.last_expanded_size = window->SizeFull;
					state.was_collapsed = true;
				}
				collapsed_windows.push_back(window);
			} else {
				if (state.was_collapsed) {
					window->SizeFull = state.last_expanded_size;
					window->Pos = ImVec2(io.DisplaySize.x - window->SizeFull.x - padding, padding + 20.0f);
					state.was_collapsed = false;
				} else {
					state.last_expanded_pos = window->Pos;
					state.last_expanded_size = window->SizeFull;
				}
			}
		}

		std::sort(collapsed_windows.begin(), collapsed_windows.end(), [](ImGuiWindow* a, ImGuiWindow* b) {
			return a->ID < b->ID;
		});

		for (ImGuiWindow* window : collapsed_windows) {
			window->SizeFull.x = uniform_width;
			window->Pos = ImVec2(io.DisplaySize.x - uniform_width - padding, current_y);
			current_y += window->TitleBarHeight + 5.0f;
		}
	}

	void ImGuiManager::Render(vk::CommandBuffer cmd) {
		ImGui::Render();
		if (m_vkInitialized && cmd) {
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		}
	}

} // namespace brassica
