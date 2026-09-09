#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstddef>
#include <string>

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/VolumetricLightingPass.hpp"
#include "Shader.hpp"

TEST_CASE("VolumetricPushConstants Struct Layout and Size") {
	CHECK(sizeof(brassica::VolumetricPushConstants) == 192);

	brassica::VolumetricPushConstants push{};
	CHECK(offsetof(brassica::VolumetricPushConstants, invViewProj) == 0);
	CHECK(offsetof(brassica::VolumetricPushConstants, cameraPos) == 64);
	CHECK(offsetof(brassica::VolumetricPushConstants, sunDir) == 80);
	CHECK(offsetof(brassica::VolumetricPushConstants, sunColor) == 96);
	CHECK(offsetof(brassica::VolumetricPushConstants, gridDimensions) == 112);
	CHECK(offsetof(brassica::VolumetricPushConstants, clipParams) == 128);
	CHECK(offsetof(brassica::VolumetricPushConstants, gridParams) == 144);
	CHECK(offsetof(brassica::VolumetricPushConstants, lodOffsets0_3) == 160);
	CHECK(offsetof(brassica::VolumetricPushConstants, lodOffsets4_7) == 176);

	CHECK(push.gridDimensions.x == doctest::Approx(160.0f));
	CHECK(push.gridDimensions.y == doctest::Approx(90.0f));
	CHECK(push.gridDimensions.z == doctest::Approx(64.0f));
}

TEST_CASE("Volumetric Lighting Compute Shaders Compilation") {
	brassica::ComputeShader injShader;
	bool injLoaded = injShader.LoadFromFile("shaders/volumetric/injection.comp");
	CHECK(injLoaded);
	if (injLoaded) {
		std::string injSource = injShader.GetSource();
		CHECK(injSource.find("#version 460") != std::string::npos);
		CHECK(injSource.find("injectionGrid") != std::string::npos);
		CHECK(injSource.find("VolumetricPushConstants") != std::string::npos);
		CHECK(injSource.find("topLevelAS") != std::string::npos);
	}

	brassica::ComputeShader integShader;
	bool integLoaded = integShader.LoadFromFile("shaders/volumetric/integration.comp");
	CHECK(integLoaded);
	if (integLoaded) {
		std::string integSource = integShader.GetSource();
		CHECK(integSource.find("#version 460") != std::string::npos);
		CHECK(integSource.find("injectionGrid") != std::string::npos);
		CHECK(integSource.find("integratedGrid") != std::string::npos);
		CHECK(integSource.find("VolumetricPushConstants") != std::string::npos);
	}
}

TEST_CASE("VolumetricLightingPass FrameGraph Pass Registration") {
	FrameGraph           fg;
	FrameGraphBlackboard blackboard;

	brassica::VolumetricLightingPass pass(nullptr, nullptr);

	brassica::VolumetricPushConstants push{};
	auto data = pass.RegisterPass(
		fg,
		blackboard,
		nullptr,
		0,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		push
	);

	CHECK(blackboard.has<brassica::VolumetricLightingData>());
	const auto& bbData = blackboard.get<brassica::VolumetricLightingData>();
	CHECK(bbData.injectionGrid == data.injectionGrid);
	CHECK(bbData.integratedGrid == data.integratedGrid);
}
