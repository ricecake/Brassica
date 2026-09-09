#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstddef>
#include <string>

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/VolumetricLightingPass.hpp"
#include "Shader.hpp"

TEST_CASE("VolumetricPushConstants Struct Layout and Size") {
	CHECK(sizeof(brassica::VolumetricPushConstants) == 224);

	brassica::VolumetricPushConstants push{};
	CHECK(offsetof(brassica::VolumetricPushConstants, invViewProj) == 0);
	CHECK(offsetof(brassica::VolumetricPushConstants, prevViewProj) == 64);
	CHECK(offsetof(brassica::VolumetricPushConstants, cameraPos) == 128);
	CHECK(offsetof(brassica::VolumetricPushConstants, sunDir) == 144);
	CHECK(offsetof(brassica::VolumetricPushConstants, sunColor) == 160);
	CHECK(offsetof(brassica::VolumetricPushConstants, params0) == 176);
	CHECK(offsetof(brassica::VolumetricPushConstants, params1) == 192);
	CHECK(offsetof(brassica::VolumetricPushConstants, cascadeDistances) == 208);

	CHECK(push.cascadeDistances.x == doctest::Approx(20.0f));
	CHECK(push.cascadeDistances.y == doctest::Approx(60.0f));
	CHECK(push.cascadeDistances.z == doctest::Approx(200.0f));
	CHECK(push.cascadeDistances.w == doctest::Approx(1000.0f));
}

TEST_CASE("Volumetric Lighting Compute Shaders Compilation") {
	brassica::ComputeShader injShader;
	bool injLoaded = injShader.LoadFromFile("shaders/volumetric/injection.comp");
	CHECK(injLoaded);
	if (injLoaded) {
		std::string injSource = injShader.GetSource();
		CHECK(injSource.find("#version 460") != std::string::npos);
		CHECK(injSource.find("injectionGrid") != std::string::npos);
		CHECK(injSource.find("uHistoryTexture") != std::string::npos);
		CHECK(injSource.find("VolumetricPushConstants") != std::string::npos);
		CHECK(injSource.find("topLevelAS") != std::string::npos);
	}

	brassica::ComputeShader integShader;
	bool integLoaded = integShader.LoadFromFile("shaders/volumetric/integration.comp");
	CHECK(integLoaded);
	if (integLoaded) {
		std::string integSource = integShader.GetSource();
		CHECK(integSource.find("#version 460") != std::string::npos);
		CHECK(integSource.find("inInjection") != std::string::npos);
		CHECK(integSource.find("outScattering") != std::string::npos);
		CHECK(integSource.find("outHistory") != std::string::npos);
		CHECK(integSource.find("Hillis-Steele") != std::string::npos);
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
