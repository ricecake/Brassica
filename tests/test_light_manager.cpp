#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "LightManager.hpp"
#include "types/Light.hpp"

using namespace brassica;

TEST_CASE("Light Struct Sizes and Alignment") {
	CHECK(sizeof(LightGPU) == 64);
	CHECK(sizeof(LightingUbo) == 1744);
	CHECK(sizeof(LightsSSBOData) == 16 + 1024 * 64);

	LightGPU gpu{};
	CHECK(gpu.position == glm::vec3(0.0f));
	CHECK(gpu.intensity == 0.0f);
	CHECK(gpu.color == glm::vec3(1.0f));
	CHECK(gpu.type == POINT_LIGHT);
	CHECK(gpu.flags == 0);
}

TEST_CASE("Light Factory Methods") {
	SUBCASE("CreatePoint") {
		auto l = Light::CreatePoint({1.0f, 2.0f, 3.0f}, 5.0f, {1.0f, 0.0f, 0.0f}, 10.0f, true);
		CHECK(l.type == POINT_LIGHT);
		CHECK(l.position == glm::vec3(1.0f, 2.0f, 3.0f));
		CHECK(l.intensity == 5.0f);
		CHECK(l.color == glm::vec3(1.0f, 0.0f, 0.0f));
		CHECK(l.outer_cutoff == 10.0f);
		CHECK(l.casts_shadow == true);

		auto gpu = l.ToGPU();
		CHECK(gpu.type == POINT_LIGHT);
		CHECK((gpu.flags & LIGHT_FLAG_CASTS_SHADOW) != 0);
	}

	SUBCASE("CreateDirectional Angles") {
		auto l = Light::CreateDirectional(90.0f, 0.0f, 10.0f, {1.0f, 1.0f, 1.0f}, true);
		CHECK(l.type == DIRECTIONAL_LIGHT);
		CHECK(l.azimuth == doctest::Approx(90.0f));
		CHECK(l.elevation == doctest::Approx(0.0f));
		CHECK(l.direction.x == doctest::Approx(-1.0f));
		CHECK(l.direction.y == doctest::Approx(0.0f));
		CHECK(l.volumetric_shadow == true);

		auto gpu = l.ToGPU();
		CHECK((gpu.flags & LIGHT_FLAG_VOLUMETRIC_SHADOW) != 0);
	}

	SUBCASE("CreateSpot") {
		auto l = Light::CreateSpot({0.0f, 5.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, 8.0f, {0.0f, 1.0f, 0.0f}, 15.0f, 30.0f);
		CHECK(l.type == SPOT_LIGHT);
		CHECK(l.inner_cutoff == doctest::Approx(glm::cos(glm::radians(15.0f))));
		CHECK(l.outer_cutoff == doctest::Approx(glm::cos(glm::radians(30.0f))));
	}

	SUBCASE("CreateEmissive") {
		auto l = Light::CreateEmissive({0.0f, 1.0f, 0.0f}, 12.0f, {1.0f, 1.0f, 0.0f}, 2.5f);
		CHECK(l.type == EMISSIVE_LIGHT);
		CHECK(l.inner_cutoff == 2.5f);
	}

	SUBCASE("CreateFlash") {
		auto l = Light::CreateFlash({10.0f, 0.0f, 0.0f}, 50.0f, {1.0f, 0.5f, 0.0f}, 40.0f, 3.0f);
		CHECK(l.type == FLASH_LIGHT);
		CHECK(l.inner_cutoff == 40.0f);
		CHECK(l.outer_cutoff == 3.0f);
	}
}

TEST_CASE("LightManager CRUD and Updates") {
	LightManager manager;

	SUBCASE("Initial Default Directional Lights") {
		CHECK(manager.GetLights().size() == 2);
		CHECK(manager.GetLights()[0].type == DIRECTIONAL_LIGHT);
		CHECK(manager.GetLights()[1].type == DIRECTIONAL_LIGHT);
	}

	SUBCASE("Add and Remove Lights") {
		auto pointLight = Light::CreatePoint({0.0f, 2.0f, 0.0f}, 10.0f, {1.0f, 1.0f, 1.0f});
		int id = manager.AddLight(pointLight);
		CHECK(id > 0);
		CHECK(manager.GetLight(id) != nullptr);
		CHECK(manager.GetLight(id)->type == POINT_LIGHT);

		manager.RemoveLight(id);
		CHECK(manager.GetLight(id) == nullptr);
	}

	SUBCASE("Day/Night Cycle Progression") {
		auto& cycle = manager.GetDayNightCycle();
		cycle.enabled = true;
		cycle.paused = false;
		cycle.speed = 1.0f;
		cycle.time = 12.0f; // Noon

		manager.Update(1.0f);
		CHECK(cycle.time == doctest::Approx(13.0f));

		const auto& lights = manager.GetLights();
		CHECK(lights[0].base_intensity > 0.0f); // Sun is bright at midday
	}

	SUBCASE("Light Behaviors - Morse and Blink") {
		Light blinkLight = Light::CreatePoint({0.0f, 0.0f, 0.0f}, 10.0f, {1.0f, 1.0f, 1.0f});
		blinkLight.SetBlink(2.0f, 0.5f); // 2s period, 50% duty cycle
		int blinkId = manager.AddLight(blinkLight);

		manager.Update(0.5f); // Half second in -> ON
		CHECK(manager.GetLight(blinkId)->intensity == doctest::Approx(10.0f));

		manager.Update(1.0f); // 1.5s in -> OFF
		CHECK(manager.GetLight(blinkId)->intensity == doctest::Approx(0.0f));

		Light morseLight = Light::CreatePoint({0.0f, 0.0f, 0.0f}, 20.0f, {1.0f, 1.0f, 1.0f});
		morseLight.SetMorse("SOS", 0.1f);
		int morseId = manager.AddLight(morseLight);

		manager.Update(0.01f);
		CHECK(manager.GetLight(morseId)->behavior.morse_sequence.empty() == false);
	}
}

TEST_CASE("LightManager GPU CPU Data Structures Sync") {
	LightManager manager;

	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
	glm::vec3 cameraPos{0.0f, 0.0f, 5.0f};
	glm::vec3 cameraDir{0.0f, 0.0f, -1.0f};

	manager.Update(0.016f);
	manager.UpdateGpuBuffers(0, view, proj, cameraPos, cameraDir, 0.1f, 1000.0f, 1.5f);

	const auto& ubo = manager.GetLightingUbo();
	CHECK(ubo.num_lights == static_cast<int32_t>(manager.GetLights().size()));
	CHECK(ubo.view_pos == cameraPos);
	CHECK(ubo.view_dir == cameraDir);
	CHECK(ubo.time == 1.5f);
	CHECK(ubo.view == view);
	CHECK(ubo.projection == proj);

	const auto& ssbo = manager.GetLightsSSBOData();
	CHECK(ssbo.count == static_cast<uint32_t>(ubo.num_lights));
	CHECK(ssbo.lights[0].type == DIRECTIONAL_LIGHT);
}
