#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cstddef>
#include <string>

#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "render/PipelineLibrary.hpp"
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

// offsetof checks for the two node-specific wrapper structs live in AtmosphereLUTNode.hpp itself
// (static_assert, so a drift fails to compile everywhere rather than only here) -- this test is
// just the GPU-independent "does sizeof/offsetof math work the way the shaders assume" sanity
// check the struct-layout test above already established the pattern for.
TEST_CASE(
	"TransmittanceLUTPushConstants/MultiScatteringLUTPushConstants place their bindless "
	"index fields where the shaders expect"
) {
	CHECK(offsetof(brassica::TransmittanceLUTPushConstants, outIndex) == 80);
	CHECK(offsetof(brassica::MultiScatteringLUTPushConstants, outIndex) == 80);
	CHECK(offsetof(brassica::MultiScatteringLUTPushConstants, transmittanceIndex) == 84);
}

TEST_CASE("Atmosphere Shaders Compilation") {
	brassica::ComputeShader transShader;
	bool                    transLoaded = transShader.LoadFromFile("shaders/atmosphere/transmittance_lut.comp");
	CHECK(transLoaded);
	if (transLoaded) {
		std::string transSource = transShader.GetSource();
		CHECK(transSource.find("#version 460") != std::string::npos);
		CHECK(transSource.find("uImagesRGBA32F") != std::string::npos);
		CHECK(transSource.find("TransmittancePushConstants") != std::string::npos);
	}

	brassica::ComputeShader multiShader;
	bool                    multiLoaded = multiShader.LoadFromFile("shaders/atmosphere/multiscattering_lut.comp");
	CHECK(multiLoaded);
	if (multiLoaded) {
		std::string multiSource = multiShader.GetSource();
		CHECK(multiSource.find("#version 460") != std::string::npos);
		CHECK(multiSource.find("uImagesRGBA32F") != std::string::npos);
		CHECK(multiSource.find("SAMPLE_LINEAR") != std::string::npos);
	}
}

// Real TransmittanceLUTNode/MultiScatteringLUTNode through the real PhysicalExecutionBackend --
// genuine coverage of the regeneration throttle (AtmosphereRegenerationState::ShouldRegenerate/
// MarkRegenerated, replacing AtmosphereLUTPass's identically-behaved original) the old
// fg-dependent version of this test never had (it only checked that RegisterPass populated the
// blackboard, using a null device that never actually built anything).
// Gated on a real headless device since this exercises real compute dispatches; see
// tests/test_headless.cpp for the skip pattern.
TEST_CASE("AtmosphereLUT nodes regenerate only when push constants actually change") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device device = engine.GetDevice();

	{
		brassica::ComputeShader transShader;
		brassica::ComputeShader multiShader;
		REQUIRE(transShader.CompileComputeFromFile(device, "shaders/atmosphere/transmittance_lut.comp"));
		REQUIRE(multiShader.CompileComputeFromFile(device, "shaders/atmosphere/multiscattering_lut.comp"));

		brassica::render::PipelineLibrary         pipelineLibrary(device, nullptr);
		brassica::AtmosphereRegenerationState     throttle{};
		brassica::graph::PhysicalResourceRegistry registry(device, engine.GetAllocator());
		brassica::graph::PhysicalExecutionBackend backend(registry);

		vk::CommandPool pool = device.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				0,
			}
		);
		vk::CommandBuffer vkCmd = device
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();
		vk::Queue queue = device.getQueue(0, 0);

		// Runs one "frame" of just the two LUT nodes and returns whether TransmittanceLUTNode
		// reported itself active (i.e. whether it actually regenerated) -- the direct signal for
		// the throttle, rather than inferring it from texture identity: ProvisionTexture reuses
		// the same PhysicalTexture object whenever the desc matches regardless of whether the
		// producing node ran, and these two LUTs' descs (fixed 256x64/32x32 R32G32B32A32Sfloat)
		// never change, so identity alone can't distinguish "regenerated" from "throttled".
		auto runFrame = [&](const brassica::AtmospherePushConstants& push) {
			brassica::graph::Graph g;
			g.Register<brassica::TransmittanceLUTNode>(brassica::TransmittanceLUTNode{
				.pipelineLibrary = &pipelineLibrary,
				.shader = &transShader,
				.throttle = &throttle,
				.atmosphere = push,
			});
			g.Register<brassica::MultiScatteringLUTNode>(brassica::MultiScatteringLUTNode{
				.pipelineLibrary = &pipelineLibrary,
				.shader = &multiShader,
				.throttle = &throttle,
				.atmosphere = push,
			});

			brassica::graph::FrameContext ctx{.width = 256, .height = 64};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			brassica::graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			backend.Execute(g, ctx, cmd, true);
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			queue.submit(submitInfo);
			queue.waitIdle();

			return g.Recipes()[0].isActive;
		};

		brassica::AtmospherePushConstants push1{};
		CHECK(runFrame(push1)); // first call: nothing generated yet, must regenerate

		REQUIRE(registry.GetTexture<brassica::TransmittanceLUT>() != nullptr);
		REQUIRE(registry.GetTexture<brassica::MultiScatteringLUT>() != nullptr);

		// Second frame, identical push constants: the throttle must report inactive.
		CHECK_FALSE(runFrame(push1));

		// Third frame, changed push constants: must regenerate again.
		brassica::AtmospherePushConstants push2 = push1;
		push2.mieAnisotropy = push1.mieAnisotropy + 0.1f;
		CHECK(runFrame(push2));

		// And immediately throttles again once caught up to the new parameters.
		CHECK_FALSE(runFrame(push2));

		pipelineLibrary.Reset();
		transShader.Destroy(device);
		multiShader.Destroy(device);
		device.destroyCommandPool(pool);
	}

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	engine.Cleanup();
}
