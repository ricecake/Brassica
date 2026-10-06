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
	CHECK(cam.maxSpeed == doctest::Approx(50000.0f));
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
	CHECK(cam.invViewMatrix != glm::mat4(1.0f));
	CHECK(cam.invProjMatrix != glm::mat4(1.0f));
	CHECK(cam.invViewProjMatrix != glm::mat4(1.0f));

	glm::mat4 identityView = cam.viewMatrix * cam.invViewMatrix;
	CHECK(identityView[0][0] == doctest::Approx(1.0f));
	CHECK(identityView[1][1] == doctest::Approx(1.0f));
	CHECK(identityView[2][2] == doctest::Approx(1.0f));
	CHECK(identityView[3][3] == doctest::Approx(1.0f));

	// Verify all 6 frustum planes are non-zero and normalized
	for (int i = 0; i < 6; ++i) {
		float len = glm::length(glm::vec3(cam.frustumPlanes[i]));
		CHECK(len == doctest::Approx(1.0f));
	}
}

TEST_CASE("Engine GetCamera Access and Uniform Configuration") {
	brassica::Engine engine;
	auto& cam = engine.GetCamera();
	cam.mode = brassica::CameraMode::Instant;

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

TEST_CASE("Camera Controls: Mode Cycle with Equal Key") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	engine.GetCamera().mode = brassica::CameraMode::Instant;

	CHECK(engine.GetCamera().mode == brassica::CameraMode::Instant);

	// Press '=' key to cycle to Accelerated mode
	handler->OnKey(nullptr, GLFW_KEY_EQUAL, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().mode == brassica::CameraMode::Accelerated);

	// Press '=' key to cycle to First Person mode
	handler->OnKey(nullptr, GLFW_KEY_EQUAL, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().mode == brassica::CameraMode::FirstPerson);

	// Press '=' key again to cycle back to Instant mode
	handler->OnKey(nullptr, GLFW_KEY_EQUAL, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().mode == brassica::CameraMode::Instant);
}

TEST_CASE("Camera Controls: Minus Key Toggles First Person Mode") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	engine.GetCamera().mode = brassica::CameraMode::Accelerated;

	// Press '-' key to enable FirstPerson mode
	handler->OnKey(nullptr, GLFW_KEY_MINUS, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().mode == brassica::CameraMode::FirstPerson);

	// Press '-' key again to return to Accelerated mode
	handler->OnKey(nullptr, GLFW_KEY_MINUS, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().mode == brassica::CameraMode::Accelerated);
}

TEST_CASE("First Person Camera: Standing Height, Crouching Height, and Speeds") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	// Press '-' key to enable First Person mode and capture
	handler->OnKey(nullptr, GLFW_KEY_MINUS, 0, GLFW_PRESS, 0);
	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	REQUIRE(engine.GetCamera().mode == brassica::CameraMode::FirstPerson);
	REQUIRE(engine.GetCamera().isCaptured);

	// No real device in this test, so TerrainManager's async readback never completes --
	// UpdateCamera's ground height falls back to its floor default (clamped to 0.0f via
	// std::max), not a real terrain sample. This is real, legitimate device-less fallback
	// behavior, not an approximation of it.
	float expectedStandingGroundLevel = 0.0f + 3.0f;

	// Ground snapping check: standing camera height should be about 3 units above ground
	CHECK(engine.GetCamera().position.y == doctest::Approx(expectedStandingGroundLevel).epsilon(0.01));

	// Test Normal Run (W forward)
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
	for (int i = 0; i < 20; ++i) {
		engine.UpdateCamera(0.05f);
	}
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_RELEASE, 0);

	// Speed should accelerate towards superhero running speed (~12 m/s)
	CHECK(engine.GetCamera().currentSpeed > 10.0f);
	CHECK(engine.GetCamera().currentSpeed <= 12.001f);

	// Test Crouching (Left Ctrl)
	handler->OnKey(nullptr, GLFW_KEY_LEFT_CONTROL, 0, GLFW_PRESS, 0);
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
	for (int i = 0; i < 30; ++i) {
		engine.UpdateCamera(0.05f);
	}
	float expectedCrouchGroundLevel = 0.0f + 1.5f;
	// Height should lower to ~1.5 units above ground
	CHECK(engine.GetCamera().position.y == doctest::Approx(expectedCrouchGroundLevel).epsilon(0.01));
	// Speed should cap at brisk walking speed (~4.0 m/s)
	CHECK(engine.GetCamera().currentSpeed <= 4.2f);

	handler->OnKey(nullptr, GLFW_KEY_LEFT_CONTROL, 0, GLFW_RELEASE, 0);
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_RELEASE, 0);

	// Test Sprinting (Left Shift + W)
	handler->OnKey(nullptr, GLFW_KEY_LEFT_SHIFT, 0, GLFW_PRESS, 0);
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
	for (int i = 0; i < 30; ++i) {
		engine.UpdateCamera(0.05f);
	}
	handler->OnKey(nullptr, GLFW_KEY_LEFT_SHIFT, 0, GLFW_RELEASE, 0);
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_RELEASE, 0);

	// Speed should accelerate towards hard sprint speed (~24 m/s)
	CHECK(engine.GetCamera().currentSpeed > 20.0f);
	CHECK(engine.GetCamera().currentSpeed <= 24.001f);
}

TEST_CASE("First Person Camera: Earth Gravity and Jump Physics") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	// Press '-' key to enable First Person mode and capture
	handler->OnKey(nullptr, GLFW_KEY_MINUS, 0, GLFW_PRESS, 0);
	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	REQUIRE(engine.GetCamera().mode == brassica::CameraMode::FirstPerson);

	// No real device in this test, so TerrainManager's async readback never completes --
	// UpdateCamera's ground height falls back to its floor default (clamped to 0.0f via
	// std::max), not a real terrain sample.
	float groundLevel = 0.0f + 3.0f;

	// Initial position is grounded
	CHECK(engine.GetCamera().position.y == doctest::Approx(groundLevel).epsilon(0.01));

	// Trigger Jump with Space key
	handler->OnKey(nullptr, GLFW_KEY_SPACE, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.05f); // Takeoff!
	handler->OnKey(nullptr, GLFW_KEY_SPACE, 0, GLFW_RELEASE, 0);

	// Camera should be in air above ground
	float peakHeight = engine.GetCamera().position.y;
	CHECK(peakHeight > groundLevel);
	CHECK(engine.GetCamera().velocity.y > 0.0f);

	// Advance time to top of jump trajectory under Earth gravity (9.81 m/s^2)
	for (int i = 0; i < 15; ++i) {
		engine.UpdateCamera(0.05f);
		peakHeight = std::max(peakHeight, engine.GetCamera().position.y);
	}

	// Peak jump height should be superhero jump height (~4-5 units above ground level)
	CHECK(peakHeight >= groundLevel + 3.5f);

	// Continue advancing time until falling back to ground
	for (int i = 0; i < 25; ++i) {
		engine.UpdateCamera(0.05f);
	}

	// Camera lands back on ground level
	CHECK(engine.GetCamera().position.y == doctest::Approx(groundLevel).epsilon(0.01));
	CHECK(engine.GetCamera().velocity.y == doctest::Approx(0.0f));
}

TEST_CASE("First Person Camera: Underwater Zero-Gravity Swimming") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	engine.GetCamera().mode = brassica::CameraMode::FirstPerson;
	// Position camera underwater in ocean region where terrain is underwater (< 0.0)
	engine.GetCamera().position = glm::vec3(1000.0f, -10.0f, 1000.0f);

	handler->OnKey(nullptr, GLFW_KEY_0, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);

	// No real device in this test, so TerrainManager's async readback never completes --
	// UpdateCamera's terrain height falls back to -1024.0f, always < 0.0f here, and the camera's
	// own Y (-10.0f, set above) is also < 0.0f -- isUnderwater's real condition
	// (camera.position.y < 0.0f && terrainHeight < 0.0f) is therefore unconditionally true in
	// this context, so this no longer needs (or should have) a SampleTerrain-based guard.
	glm::vec3 underwaterPos = engine.GetCamera().position;

	// No key pressed -> no gravity falling underwater
	engine.UpdateCamera(0.1f);
	CHECK(engine.GetCamera().position.y == doctest::Approx(underwaterPos.y));

	// Move forward underwater
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
	for (int i = 0; i < 20; ++i) {
		engine.UpdateCamera(0.05f);
	}
	handler->OnKey(nullptr, GLFW_KEY_W, 0, GLFW_RELEASE, 0);

	// Speed caps at swimming speed (~5.0 m/s)
	CHECK(engine.GetCamera().currentSpeed <= 5.001f);
}

TEST_CASE("Camera Controls: Speed Adjustment (PageUp, PageDown, Home, End)") {
	brassica::Engine engine;
	auto handler = std::make_shared<brassica::DefaultInputHandler>();
	engine.SetInputHandler(handler);

	engine.GetCamera().speed = 10.0f;

	CHECK(engine.GetCamera().speed == doctest::Approx(10.0f));

	// Increase speed with Page Up
	handler->OnKey(nullptr, GLFW_KEY_PAGE_UP, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(20.0f));

	// Set to max speed with End
	handler->OnKey(nullptr, GLFW_KEY_END, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == engine.GetCamera().maxSpeed);

	engine.GetCamera().speed = 1000.0f;

	// Decrease speed with Page Down
	handler->OnKey(nullptr, GLFW_KEY_PAGE_DOWN, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == doctest::Approx(990.0f));

	// Reset to default speed with Home
	handler->OnKey(nullptr, GLFW_KEY_HOME, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.016f);
	CHECK(engine.GetCamera().speed == engine.GetCamera().defaultSpeed);
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
	CHECK(engine.GetCamera().roll > initRoll);

	handler->OnKey(nullptr, GLFW_KEY_E, 0, GLFW_PRESS, 0);
	engine.UpdateCamera(0.2f);
	handler->OnKey(nullptr, GLFW_KEY_E, 0, GLFW_RELEASE, 0);
	CHECK(engine.GetCamera().roll < initRoll);

	// Mouse Look
	handler->OnCursorPos(nullptr, 100.0, 100.0);
	engine.UpdateCamera(0.016f); // Initialize lastMouse
	float initYaw = engine.GetCamera().yaw;

	handler->OnCursorPos(nullptr, 150.0, 100.0);
	engine.UpdateCamera(0.016f); // Mouse moved right -> Yaw decreases
	CHECK(engine.GetCamera().yaw != initYaw);
}
