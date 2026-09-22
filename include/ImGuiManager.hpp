#pragma once

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "GLFW/glfw3.h"
#include "IManager.hpp"
#include "imgui.h"
#include "types/CameraData.hpp"
#include "ui/IWidget.hpp"

namespace brassica {

	struct WindowState {
		ImVec2 last_expanded_pos{0.0f, 0.0f};
		ImVec2 last_expanded_size{0.0f, 0.0f};
		bool   was_collapsed{false};
	};

	struct ImGuiManagerState {
		bool visible{false};

		auto GetReflection() const {
			return std::make_tuple(MakeField("visible", "UI Visible", &ImGuiManagerState::visible));
		}
	};

	class ImGuiManager: public ManagerBase<ImGuiManager, ImGuiManagerState> {
	public:
		using State = ImGuiManagerState;

		ImGuiManager() = default;
		~ImGuiManager() override;

		void Initialize() override;
		void Shutdown() override;

		std::string GetManagerName() const override { return "ImGuiManager"; }

		State GetState() const override { return State{m_visible}; }

		void SetState(const State& state) override { m_visible = state.visible; }

		void InitVulkanAndGlfw(
			GLFWwindow*        window,
			vk::Instance       instance,
			vk::PhysicalDevice physicalDevice,
			vk::Device         device,
			uint32_t           queueFamily,
			vk::Queue          queue,
			vk::Format         swapchainFormat,
			uint32_t           imageCount
		);

		void OnKey(int key, int action);

		void NewFrame(const CameraData& camera);
		void NewFrame(const glm::vec3& cameraPosition);
		void Render(vk::CommandBuffer cmd);

		void ToggleVisibility() { m_visible = !m_visible; }

		[[nodiscard]] bool IsVisible() const { return m_visible; }

		void SetVisible(bool visible) { m_visible = visible; }

		void AddWidget(std::shared_ptr<ui::IWidget> widget);

		template <typename T>
		std::shared_ptr<T> GetWidget() {
			for (auto& widget : m_widgets) {
				if (auto casted = std::dynamic_pointer_cast<T>(widget)) {
					return casted;
				}
			}
			return nullptr;
		}

		[[nodiscard]] const std::vector<std::shared_ptr<ui::IWidget>>& GetWidgets() const { return m_widgets; }

	private:
		void PositionMinimizedWindows();

		GLFWwindow*                                   m_window{nullptr};
		vk::Device                                    m_device{nullptr};
		vk::DescriptorPool                            m_descriptorPool{nullptr};
		bool                                          m_vkInitialized{false};
		bool                                          m_visible{false};
		std::vector<std::shared_ptr<ui::IWidget>>     m_widgets;
		std::unordered_map<unsigned int, WindowState> m_windowStates;
	};

} // namespace brassica
