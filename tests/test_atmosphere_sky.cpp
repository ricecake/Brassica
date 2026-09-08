#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstddef>
#include <glm/gtc/epsilon.hpp>
#include <string>

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/AtmosphereSkyPass.hpp"
#include "Shader.hpp"
#include "types/Light.hpp"

TEST_CASE("Directional Light Logic and Angle Conversions") {
	brassica::DirectionalLight light = brassica::DirectionalLight::CreateFromAngles(90.0f, 45.0f, 3.0f, glm::vec3(1.0f, 0.9f, 0.8f));

	CHECK(light.azimuth == doctest::Approx(90.0f));
	CHECK(light.elevation == doctest::Approx(45.0f));
	CHECK(light.intensity == doctest::Approx(3.0f));

	glm::vec3 lightDir = light.GetLightDir();
	CHECK(lightDir.y == doctest::Approx(std::sin(glm::radians(45.0f))));

	float azOut = 0.0f, elOut = 0.0f;
	brassica::DirectionalLight::GetAnglesFromDirection(light.direction, azOut, elOut);
	CHECK(azOut == doctest::Approx(90.0f));
	CHECK(elOut == doctest::Approx(45.0f));

	brassica::LightGPU gpu = light.ToGPU();
	CHECK(gpu.type == static_cast<int32_t>(brassica::LightType::Directional));
	CHECK(gpu.intensity == doctest::Approx(3.0f));
	CHECK(gpu.flags == brassica::LIGHT_FLAG_VOLUMETRIC_SHADOW);
}

TEST_CASE("GlobalLightingData Day Night Cycle") {
	brassica::GlobalLightingData lighting{};
	lighting.UpdateSunMoonFromTime(12.0f); // Solar noon

	CHECK(lighting.sun.elevation > 0.0f);
	CHECK(lighting.moon.elevation < 0.0f);

	lighting.UpdateSunMoonFromTime(0.0f); // Midnight
	CHECK(lighting.sun.elevation < 0.0f);
	CHECK(lighting.moon.elevation > 0.0f);
}

TEST_CASE("AtmosphereSkyPushConstants Struct Layout and Size") {
	CHECK(sizeof(brassica::AtmosphereSkyPushConstants) == 128);

	brassica::AtmosphereSkyPushConstants push{};
	CHECK(offsetof(brassica::AtmosphereSkyPushConstants, invViewProj) == 0);
	CHECK(offsetof(brassica::AtmosphereSkyPushConstants, cameraPosAndScale) == 64);
	CHECK(offsetof(brassica::AtmosphereSkyPushConstants, sunDirAndAureole) == 80);
	CHECK(offsetof(brassica::AtmosphereSkyPushConstants, moonDirAndCirrus) == 96);
	CHECK(offsetof(brassica::AtmosphereSkyPushConstants, sunRadianceAndSkyExp) == 112);
}

TEST_CASE("Sky Task Mesh Fragment Shaders Compilation") {
	brassica::TaskShader taskShader;
	bool taskLoaded = taskShader.LoadFromFile("shaders/atmosphere/sky.task");
	CHECK(taskLoaded);
	if (taskLoaded) {
		std::string taskSource = taskShader.GetSource();
		CHECK(taskSource.find("#version 460") != std::string::npos);
		CHECK(taskSource.find("EmitMeshTasksEXT") != std::string::npos);
	}

	brassica::MeshShader meshShader;
	bool meshLoaded = meshShader.LoadFromFile("shaders/atmosphere/sky.mesh");
	CHECK(meshLoaded);
	if (meshLoaded) {
		std::string meshSource = meshShader.GetSource();
		CHECK(meshSource.find("#version 460") != std::string::npos);
		CHECK(meshSource.find("SetMeshOutputsEXT") != std::string::npos);
	}

	brassica::FragmentShader fragShader;
	bool fragLoaded = fragShader.LoadFromFile("shaders/atmosphere/sky.frag");
	CHECK(fragLoaded);
	if (fragLoaded) {
		std::string fragSource = fragShader.GetSource();
		CHECK(fragSource.find("#version 460") != std::string::npos);
		CHECK(fragSource.find("sampleSkyView") != std::string::npos);
		CHECK(fragSource.find("computeStars") != std::string::npos);
		CHECK(fragSource.find("computeNebula") != std::string::npos);
	}
}

TEST_CASE("AtmosphereSkyPass FrameGraph Registration") {
	FrameGraph           fg;
	FrameGraphBlackboard blackboard;

	brassica::AtmosphereLUTPass lutPass(nullptr, nullptr);
	brassica::AtmospherePushConstants atmospherePush{};
	brassica::SkyViewPushConstants skyPush{};
	lutPass.RegisterPass(fg, blackboard, 0, atmospherePush, skyPush);

	brassica::AtmosphereSkyPass skyPass(nullptr, nullptr, nullptr);
	brassica::AtmosphereSkyPushConstants push{};
	auto data = skyPass.RegisterPass(fg, blackboard, {1280, 720}, nullptr, 0, push);

	CHECK(data.background != FrameGraphResource{});
}
