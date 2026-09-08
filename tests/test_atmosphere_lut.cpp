#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstddef>
#include <string>

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/AtmosphereLUTPass.hpp"
#include "Shader.hpp"
#include "types/AtmospherePushConstants.hpp"

TEST_CASE("AtmospherePushConstants Struct Layout and Size") {
	CHECK(sizeof(brassica::AtmospherePushConstants) == 80);

	brassica::AtmospherePushConstants push{};
	CHECK(offsetof(brassica::AtmospherePushConstants, rayleighScatteringBase) == 0);
	CHECK(offsetof(brassica::AtmospherePushConstants, rayleighScaleHeight) == 12);
	CHECK(offsetof(brassica::AtmospherePushConstants, ozoneAbsorptionBase) == 16);
	CHECK(offsetof(brassica::AtmospherePushConstants, mieScaleHeight) == 28);
	CHECK(offsetof(brassica::AtmospherePushConstants, hazeColor) == 32);
	CHECK(offsetof(brassica::AtmospherePushConstants, mieScatteringBase) == 44);
	CHECK(offsetof(brassica::AtmospherePushConstants, mieExtinctionBase) == 48);
	CHECK(offsetof(brassica::AtmospherePushConstants, rayleighScale) == 52);
	CHECK(offsetof(brassica::AtmospherePushConstants, mieScale) == 56);
	CHECK(offsetof(brassica::AtmospherePushConstants, mieAnisotropy) == 60);
	CHECK(offsetof(brassica::AtmospherePushConstants, atmosphereHeight) == 64);
	CHECK(offsetof(brassica::AtmospherePushConstants, hazeDensity) == 68);
	CHECK(offsetof(brassica::AtmospherePushConstants, hazeHeight) == 72);

	CHECK(push.rayleighScatteringBase.x == doctest::Approx(5.802e-3f));
	CHECK(push.rayleighScaleHeight == doctest::Approx(8.0f));
	CHECK(push.mieScaleHeight == doctest::Approx(1.2f));
	CHECK(push.atmosphereHeight == doctest::Approx(100.0f));
}

TEST_CASE("Atmosphere Shaders Compilation") {
	brassica::ComputeShader transShader;
	bool transLoaded = transShader.LoadFromFile("shaders/atmosphere/transmittance_lut.comp");
	CHECK(transLoaded);
	if (transLoaded) {
		std::string transSource = transShader.GetSource();
		CHECK(transSource.find("#version 460") != std::string::npos);
		CHECK(transSource.find("outTransmittance") != std::string::npos);
		CHECK(transSource.find("AtmospherePushConstants") != std::string::npos);
	}

	brassica::ComputeShader multiShader;
	bool multiLoaded = multiShader.LoadFromFile("shaders/atmosphere/multiscattering_lut.comp");
	CHECK(multiLoaded);
	if (multiLoaded) {
		std::string multiSource = multiShader.GetSource();
		CHECK(multiSource.find("#version 460") != std::string::npos);
		CHECK(multiSource.find("outMultiScattering") != std::string::npos);
		CHECK(multiSource.find("u_transmittanceLUT") != std::string::npos);
	}
}

TEST_CASE("AtmosphereLUTPass FrameGraph Pass Registration") {
	FrameGraph           fg;
	FrameGraphBlackboard blackboard;

	// Mock AtmosphereLUTPass registration using nullptr device
	brassica::AtmosphereLUTPass pass(nullptr, nullptr);

	brassica::AtmospherePushConstants push{};
	auto data = pass.RegisterPass(fg, blackboard, 0, push);

	CHECK(blackboard.has<brassica::AtmosphereLUTData>());
	const auto& bbData = blackboard.get<brassica::AtmosphereLUTData>();
	CHECK(bbData.transmittanceLUT == data.transmittanceLUT);
	CHECK(bbData.multiScatteringLUT == data.multiScatteringLUT);
}
