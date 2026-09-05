#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "InputHandler.hpp"

class CustomTestHandler {
public:
	bool keyHandled{false};
	bool mouseHandled{false};
	bool cursorPosHandled{false};
	bool scrollHandled{false};
	bool sizeHandled{false};

	void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
		keyHandled = true;
	}
	void OnMouseButton(GLFWwindow* window, int button, int action, int mods) {
		mouseHandled = true;
	}
	void OnCursorPos(GLFWwindow* window, double x, double y) {
		cursorPosHandled = true;
	}
	void OnScroll(GLFWwindow* window, double x, double y) {
		scrollHandled = true;
	}
	void OnFramebufferSize(GLFWwindow* window, int width, int height) {
		sizeHandled = true;
	}
};

static_assert(brassica::InputHandlerConcept<brassica::DefaultInputHandler>);
static_assert(brassica::InputHandlerConcept<CustomTestHandler>);

TEST_CASE("InputHandlerConcept with DefaultInputHandler") {
	brassica::DefaultInputHandler handler;
	handler.OnKey(nullptr, 65, 0, 1, 0); // Key Press
	CHECK(handler.IsKeyPressed(65));

	handler.OnKey(nullptr, 65, 0, 0, 0); // Key Release
	CHECK(!handler.IsKeyPressed(65));

	handler.OnMouseButton(nullptr, 0, 1, 0);
	CHECK(handler.IsMouseButtonPressed(0));

	handler.OnCursorPos(nullptr, 100.0, 200.0);
	auto [mx, my] = handler.GetCursorPos();
	CHECK(mx == 100.0);
	CHECK(my == 200.0);

	handler.OnScroll(nullptr, 1.0, -1.0);
	auto [sx, sy] = handler.GetScrollOffset();
	CHECK(sx == 1.0);
	CHECK(sy == -1.0);
}

TEST_CASE("InputHandlerAdapter for Custom Non-Inheriting Types") {
	CustomTestHandler custom;
	brassica::InputHandlerAdapter<CustomTestHandler> adapter(custom);

	adapter.OnKey(nullptr, 0, 0, 0, 0);
	adapter.OnMouseButton(nullptr, 0, 0, 0);
	adapter.OnCursorPos(nullptr, 0, 0);
	adapter.OnScroll(nullptr, 0, 0);
	adapter.OnFramebufferSize(nullptr, 800, 600);

	CHECK(adapter.GetInstance().keyHandled);
	CHECK(adapter.GetInstance().mouseHandled);
	CHECK(adapter.GetInstance().cursorPosHandled);
	CHECK(adapter.GetInstance().scrollHandled);
	CHECK(adapter.GetInstance().sizeHandled);
}

TEST_CASE("Global Default Input Handler Type Configuration Before Engine Creation") {
	brassica::SetDefaultInputHandlerType<CustomTestHandler>();

	auto created = brassica::CreateDefaultInputHandler();
	REQUIRE(created != nullptr);

	created->OnKey(nullptr, 0, 0, 0, 0);

	// Reset back to default
	brassica::SetDefaultInputHandlerType<brassica::DefaultInputHandler>();
}

TEST_CASE("Engine Camera FOV and Input Handler Configuration") {
	brassica::Engine engine;
	CHECK(engine.GetFov() == doctest::Approx(1.2f));

	engine.SetFov(1.5f);
	CHECK(engine.GetFov() == doctest::Approx(1.5f));

	auto customHandler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(customHandler);
	CHECK(engine.GetInputHandler() == customHandler);
}
