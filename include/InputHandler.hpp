#pragma once

#include <array>
#include <concepts>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

struct GLFWwindow;

namespace brassica {

	// Concept defining the specification for input handler classes
	template <typename T>
	concept InputHandlerConcept = requires(
		T           handler,
		GLFWwindow* window,
		int         key,
		int         scancode,
		int         action,
		int         mods,
		int         button,
		double      x,
		double      y,
		int         width,
		int         height
	) {
		{ handler.OnKey(window, key, scancode, action, mods) } -> std::same_as<void>;
		{ handler.OnMouseButton(window, button, action, mods) } -> std::same_as<void>;
		{ handler.OnCursorPos(window, x, y) } -> std::same_as<void>;
		{ handler.OnScroll(window, x, y) } -> std::same_as<void>;
		{ handler.OnFramebufferSize(window, width, height) } -> std::same_as<void>;
	};

	// Polymorphic interface for runtime input handling integration
	class IInputHandler {
	public:
		virtual ~IInputHandler() = default;

		virtual void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) = 0;
		virtual void OnMouseButton(GLFWwindow* window, int button, int action, int mods) = 0;
		virtual void OnCursorPos(GLFWwindow* window, double xpos, double ypos) = 0;
		virtual void OnScroll(GLFWwindow* window, double xoffset, double yoffset) = 0;
		virtual void OnFramebufferSize(GLFWwindow* window, int width, int height) = 0;
	};

	// Default implementation wrapping GLFW key and mouse events
	class DefaultInputHandler : public IInputHandler {
	public:
		DefaultInputHandler() = default;
		~DefaultInputHandler() override = default;

		void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) override;
		void OnMouseButton(GLFWwindow* window, int button, int action, int mods) override;
		void OnCursorPos(GLFWwindow* window, double xpos, double ypos) override;
		void OnScroll(GLFWwindow* window, double xoffset, double yoffset) override;
		void OnFramebufferSize(GLFWwindow* window, int width, int height) override;

		bool IsKeyPressed(int key) const;
		bool IsKeyJustPressed(int key);
		bool IsMouseButtonPressed(int button) const;
		std::pair<double, double> GetCursorPos() const;
		std::pair<double, double> GetScrollOffset() const;

	private:
		std::array<bool, 512> keys{};
		std::array<bool, 512> justPressedKeys{};
		std::array<bool, 16>  mouseButtons{};
		double                mouseX{0.0};
		double                mouseY{0.0};
		double                scrollX{0.0};
		double                scrollY{0.0};
	};

	// Adapter for any type T that satisfies InputHandlerConcept but does not inherit from IInputHandler
	template <InputHandlerConcept T>
	class InputHandlerAdapter : public IInputHandler {
	public:
		explicit InputHandlerAdapter(T handler) : instance(std::move(handler)) {}

		void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) override {
			instance.OnKey(window, key, scancode, action, mods);
		}
		void OnMouseButton(GLFWwindow* window, int button, int action, int mods) override {
			instance.OnMouseButton(window, button, action, mods);
		}
		void OnCursorPos(GLFWwindow* window, double xpos, double ypos) override {
			instance.OnCursorPos(window, xpos, ypos);
		}
		void OnScroll(GLFWwindow* window, double xoffset, double yoffset) override {
			instance.OnScroll(window, xoffset, yoffset);
		}
		void OnFramebufferSize(GLFWwindow* window, int width, int height) override {
			instance.OnFramebufferSize(window, width, height);
		}

		T& GetInstance() { return instance; }
		const T& GetInstance() const { return instance; }

	private:
		T instance;
	};

	static_assert(InputHandlerConcept<DefaultInputHandler>);

	// Global default input handler configuration functions
	void SetDefaultInputHandlerFactory(std::function<std::shared_ptr<IInputHandler>()> factory);
	std::shared_ptr<IInputHandler> CreateDefaultInputHandler();

	template <InputHandlerConcept T, typename... Args>
	void SetDefaultInputHandlerType(Args&&... args) {
		SetDefaultInputHandlerFactory([args = std::make_tuple(std::forward<Args>(args)...)]() -> std::shared_ptr<IInputHandler> {
			return std::apply([](auto&&... a) {
				if constexpr (std::derived_from<T, IInputHandler>) {
					return std::make_shared<T>(std::forward<decltype(a)>(a)...);
				} else {
					return std::make_shared<InputHandlerAdapter<T>>(T(std::forward<decltype(a)>(a)...));
				}
			}, args);
		});
	}

} // namespace brassica
