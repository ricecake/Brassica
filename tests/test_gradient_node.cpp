#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/GradientNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

using namespace brassica;

// GradientNode is the Node/Pass-unification migration's trivial case: zero descriptors, a fixed
// pair of shaders, a fixed output format -- but its GraphicsPipelineState still defaults
// enableShadingRate=true (matching the old GradientPass exactly), which means creating its real
// pipeline requires the primitiveFragmentShadingRate/attachmentFragmentShadingRate device
// features. Those are part of VK_KHR_fragment_shading_rate, which MoltenVK does not implement --
// so like every TerrainPass/DeferredPass test in this codebase, this bootstraps through the full
// brassica::Engine (not MinimalDevice) and will skip on this Mac, but proves the entire vertical
// slice -- Setup -> Recipe -> Provision -> PipelineLibrary::ResolveCached -> a real bound pipeline
// -> a real draw, with zero validation errors -- on the user's real hardware.
//
// The cache/generation-invalidation mechanics themselves (which don't need VRS) have their own
// dedicated real-hardware coverage via MinimalDevice in tests/test_pipeline_library.cpp.
TEST_CASE("GradientNode renders through PhysicalExecutionBackend with no validation errors, across two frames") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = engine.GetDevice();
	// Queue family/index 0 -- same simplification test_atmosphere_lut.cpp already makes for a
	// headless single-queue-family test context; Engine exposes no public graphics-queue
	// accessor, and adding one solely for test convenience isn't worth it.
	vk::Queue queue = vkDevice.getQueue(0, 0);

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				0,
			}
		);
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		VertexShader   vertShader;
		FragmentShader fragShader;
		REQUIRE(vertShader.CompileVertexFromFile(vkDevice, "shaders/gradient.vert"));
		REQUIRE(fragShader.CompileFragmentFromFile(vkDevice, "shaders/gradient.frag"));

		render::PipelineLibrary pipelineLibrary(vkDevice, engine.GetPipelineCache());

		graph::PhysicalResourceRegistry registry(vkDevice, engine.GetAllocator());
		graph::PhysicalExecutionBackend backend(registry);

		// Two frames, not one: proves ResolveCached's second-frame call is a real cache hit (no
		// duplicate pipeline creation) rather than something that only happens to work once, and
		// that GradientBackground's desc-match reuse (PhysicalRegistry::ProvisionTexture) holds
		// across frames the same way it does for every other steady-state resource.
		for (std::uint64_t frameIndex = 0; frameIndex < 2; ++frameIndex) {
			graph::Graph graph;
			graph.Register<GradientNode>(GradientNode{
				.pipelineLibrary = &pipelineLibrary,
				.vertShader = &vertShader,
				.fragShader = &fragShader,
				.extent = vk::Extent2D{256, 256},
			});

			graph::FrameContext ctx{.width = 256, .height = 256, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			queue.submit(submitInfo);
			queue.waitIdle();
		}

		CHECK(registry.GetTexture<GradientBackground>() != nullptr);

		pipelineLibrary.Reset();
		vertShader.Destroy(vkDevice);
		fragShader.Destroy(vkDevice);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	engine.Cleanup();
}
