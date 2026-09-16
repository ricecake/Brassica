#pragma once

#include <memory>

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>
#include "IManager.hpp"
#include "imgui.h"

namespace brassica {

	class ImGuiManager: public IManager {
	public:
		ImGuiManager() = default;
		~ImGuiManager() override;

		void Initialize() override;
		void Shutdown() override;

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

		void NewFrame(const glm::vec3& cameraPosition);
		void Render(vk::CommandBuffer cmd);

		void ToggleVisibility() { m_visible = !m_visible; }
		[[nodiscard]] bool IsVisible() const { return m_visible; }
		void SetVisible(bool visible) { m_visible = visible; }

	private:
		GLFWwindow*        m_window{nullptr};
		vk::Device         m_device{nullptr};
		vk::DescriptorPool m_descriptorPool{nullptr};
		bool               m_vkInitialized{false};
		bool               m_visible{false};
	};

} // namespace brassica
