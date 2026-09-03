#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/GradientPass.hpp"
#include "passes/MeshCubePass.hpp"
#include "passes/RenderResources.hpp"

using namespace brassica;

TEST_CASE("RenderGraph Blackboard resource passing") {
	FrameGraph           fg;
	FrameGraphBlackboard blackboard;

	FrameGraphTexture mockTex;
	vk::Extent2D      extent{1280, 720};
	vk::Format        format = vk::Format::eB8G8R8A8Unorm;

	FrameGraphResource swapchainRes = fg.import("SwapchainImage", {extent, format}, std::move(mockTex));
	CHECK(fg.isValid(swapchainRes));

	blackboard.add<SwapchainData>() = SwapchainData{.target = swapchainRes};
	CHECK(blackboard.has<SwapchainData>());
	CHECK(blackboard.get<SwapchainData>().target == swapchainRes);

	// Mock pass 1 (similar to GradientPass registration logic)
	const auto& gradientData = blackboard.get<SwapchainData>();
	const auto& mockGradientPass = fg.addCallbackPass<GradientPassData>(
		"MockGradientPass",
		[&](FrameGraph::Builder& builder, GradientPassData& data) {
			data.target = builder.write(gradientData.target);
			builder.setSideEffect();
		},
		[](const GradientPassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<GradientPassData>() = mockGradientPass;

	CHECK(blackboard.has<GradientPassData>());
	CHECK(blackboard.get<GradientPassData>().target != swapchainRes); // Builder.write creates new version handle

	// Mock pass 2 (similar to MeshCubePass registration logic reading GradientPassData from blackboard)
	FrameGraphResource inputResource;
	if (const auto* gData = blackboard.try_get<GradientPassData>()) {
		inputResource = gData->target;
	} else {
		inputResource = blackboard.get<SwapchainData>().target;
	}
	CHECK(inputResource == blackboard.get<GradientPassData>().target);

	const auto& mockMeshCubePass = fg.addCallbackPass<MeshCubePassData>(
		"MockMeshCubePass",
		[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
			data.target = builder.write(inputResource);
			builder.setSideEffect();
		},
		[](const MeshCubePassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<MeshCubePassData>() = mockMeshCubePass;

	CHECK(blackboard.has<MeshCubePassData>());

	// Compile framegraph
	fg.compile();
}
