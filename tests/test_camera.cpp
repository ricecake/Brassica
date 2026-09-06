#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <glm/gtc/epsilon.hpp>
#include "Engine.hpp"
#include "InputHandler.hpp"
#include "types/CameraData.hpp"

TEST_CASE("CameraData Default Values and Direction Vectors") {
	brassica::CameraData cam;

	CHECK(cam.position == glm::vec3(0.0f, 15.0f, 30.0f));
	CHECK(cam.fov == doctest::Approx(1.2f));
	CHECK(cam.speed == doctest::Approx(10.0f));
	CHECK(cam.defaultSpeed == doctest::Approx(10.0f));
	CHECK(cam.minSpeed == doctest::Approx(1.0f));
	CHECK(cam.maxSpeed == doctest::Approx(100.0f));
	CHECK(!cam.isCaptured);

	cam.UpdateOrientation();
	glm::vec3 fwd = cam.GetForward();
	glm::vec3 up = cam.GetUp();
	glm::vec3 right = cam.GetRight();

	// Check unit lengths
	CHECK(glm::length(fwd) == doctest::Approx(1.0f));
	CHECK(glm::length(up) == doctest::Approx(1.0f));
	CHECK(glm::length(right) == doctest::Approx(1.0f));
}

TEST_CASE("CameraData Matrix and Frustum Plane Updates") {
	brassica::CameraData cam;
	cam.UpdateMatrices(16.0f / 9.0f);

	CHECK(cam.aspectRatio == doctest::Approx(16.0f / 9.0f));
	CHECK(cam.viewMatrix != glm::mat4(1.0f));
	CHECK(cam.projMatrix != glm::mat4(1.0f));
	CHECK(cam.viewProjMatrix == cam.projMatrix * cam.viewMatrix);

	// Verify all 6 frustum planes are non-zero and normalized
	for (int i = 0; i < 6; ++i) {
		float len = glm::length(glm::vec3(cam.frustumPlanes[i]));
		CHECK(len == doctest::Approx(1.0f));
	}
}

TEST_CASE("Engine GetCamera Access and Uniform Configuration") {
	brassica::Engine engine;
	auto& cam = engine.GetCamera();

	CHECK(cam.fov == doctest::Approx(1.2f));
	CHECK(cam.position == glm::vec3(0.0f, 15.0f, 30.0f));

	cam.position = glm::vec3(10.0f, 20.0f, 30.0f);
	CHECK(engine.GetCamera().position == glm::vec3(10.0f, 20.0f, 30.0f));

	engine.SetFov(1.5f);
	CHECK(engine.GetFov() == doctest::Approx(1.5f));
	CHECK(engine.GetCamera().fov == doctest::Approx(1.5f));
}

TEST_CASE("Camera Controls: Capture Toggle with 0 Key") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	CHECK(!engine.GetCamera().isCaptured);

	// Press '0' key to capture
	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().isCaptured);

	// Press '0' key again to release
	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(!engine.GetCamera().isCaptured);
}

TEST_CASE("Camera Controls: Speed Adjustment (PageUp, PageDown, Home, End)") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	CHECK(engine.GetCamera().speed == doctest::Approx(10.0f));

	// Increase speed with Page Up
	handler->OnKey(nullptr, GLFW_KEY_PAGE_UP, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(15.0f));

	// Set to max speed with End
	handler->OnKey(nullptr, GLFW_KEY_END, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(100.0f));

	// Decrease speed with Page Down
	handler->OnKey(nullptr, GLFW_KEY_PAGE_DOWN, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(95.0f));

	// Reset to default speed with Home
	handler->OnKey(nullptr, GLFW_KEY_HOME, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(10.0f));
}

TEST_CASE("Camera Controls: WASD, Space, Shift, Q, E, Mouse Look") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	// Capture camera
	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	REQUIRE(engine.GetCamera().isCaptured);

	glm::vec3 initPos = engine.GetCamera().position;

	// Hold W (Move Forward)
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.1f);
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_RELEASE, 0);

	glm::vec3 movedPos = engine.GetCamera().position;
	CHECK(movedPos != initPos);

	// Hold Spacebar (Ascent in world space)
	initPos = engine.GetCamera().position;
	handler->OnKey(nullptr, GLFW_KEY_SPACE, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.1f);
	handler->OnKey(nullptr, GLFW_KEY_SPACE, 0, GLFW_RELEASE, 0);

	CHECK(engine.GetCamera().position.y > initPos.y);

	// Hold Left Shift (Descent in world space)
	initPos = engine.GetCamera().position;
	handler->OnKey(nullptr, GLFW_KEY_LEFT_SHIFT, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.1f);
	handler->OnKey(nullptr, GLFW_KEY_LEFT_SHIFT, 0, GLFW_RELEASE, 0);

	CHECK(engine.GetCamera().position.y < initPos.y);

	// Hold Q (Roll Counter-Clockwise) and E (Roll Clockwise)
	float initRoll = engine.GetCamera().roll;
	handler->OnKey(nullptr, GLFW_KEY_Q, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.1f);
	handler->OnKey(nullptr, GLFW_KEY_Q, 0, GLFW_RELEASE, 0);
	CHECK(engine.GetCamera().roll < initRoll);

	handler->OnKey(nullptr, GLFW_KEY_E, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.2f);
	handler->OnKey(nullptr, GLFW_KEY_E, 0, GLFW_RELEASE, 0);
	CHECK(engine.GetCamera().roll > initRoll);

	// Mouse Look
	handler->OnCursorPos(nullptr, 100.0, 100.0);
	engine.UpdateCamera(0.016f); // Initialize lastMouse
	float initYaw = engine.GetCamera().yaw;

	handler->OnCursorPos(nullptr, 150.0, 100.0);
	engine.UpdateCamera(0.016f); // Mouse moved right -> Yaw decreases
	CHECK(engine.GetCamera().yaw != initYaw);
}
