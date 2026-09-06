#include "InputHandler.hpp"
#include "GLFW/glfw3.h"

namespace brassica {

	static std::function<std::shared_ptr<IInputHandler>()> s_defaultInputHandlerFactory = []() -> std::shared_ptr<IInputHandler> {
		return std::make_shared<DefaultInputHandler>();
	};

	void SetDefaultInputHandlerFactory(std::function<std::shared_ptr<IInputHandler>()> factory) {
		if (factory) {
			s_defaultInputHandlerFactory = std::move(factory);
		}
	}

	std::shared_ptr<IInputHandler> CreateDefaultInputHandler() {
		return s_defaultInputHandlerFactory ? s_defaultInputHandlerFactory() : std::make_shared<DefaultInputHandler>();
	}

	void DefaultInputHandler::OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (key >= 0 && static_cast<size_t>(key) < keys.size()) {
			if (action == GLFW_PRESS) {
				keys[key] = true;
				justPressedKeys[key] = true;
			} else if (action == GLFW_RELEASE) {
				keys[key] = false;
			}
		}
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && window) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
	}

	bool DefaultInputHandler::IsKeyJustPressed(int key) {
		if (key >= 0 && static_cast<size_t>(key) < justPressedKeys.size()) {
			bool val = justPressedKeys[key];
			justPressedKeys[key] = false;
			return val;
		}
		return false;
	}

	void DefaultInputHandler::OnMouseButton(GLFWwindow* window, int button, int action, int mods) {
		if (button >= 0 && static_cast<size_t>(button) < mouseButtons.size()) {
			mouseButtons[button] = (action != GLFW_RELEASE);
		}
	}

	void DefaultInputHandler::OnCursorPos(GLFWwindow* window, double xpos, double ypos) {
		mouseX = xpos;
		mouseY = ypos;
	}

	void DefaultInputHandler::OnScroll(GLFWwindow* window, double xoffset, double yoffset) {
		scrollX += xoffset;
		scrollY += yoffset;
	}

	void DefaultInputHandler::OnFramebufferSize(GLFWwindow* window, int width, int height) {
	}

	bool DefaultInputHandler::IsKeyPressed(int key) const {
		if (key >= 0 && static_cast<size_t>(key) < keys.size()) {
			return keys[key];
		}
		return false;
	}

	bool DefaultInputHandler::IsMouseButtonPressed(int button) const {
		if (button >= 0 && static_cast<size_t>(button) < mouseButtons.size()) {
			return mouseButtons[button];
		}
		return false;
	}

	std::pair<double, double> DefaultInputHandler::GetCursorPos() const {
		return {mouseX, mouseY};
	}

	std::pair<double, double> DefaultInputHandler::GetScrollOffset() const {
		return {scrollX, scrollY};
	}

} // namespace brassica
