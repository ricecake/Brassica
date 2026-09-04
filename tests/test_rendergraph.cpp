#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/ComputeTestPass.hpp"
#include "passes/DeferredPass.hpp"
#include "passes/GradientPass.hpp"
#include "passes/MeshCubePass.hpp"
#include "passes/PassResource.hpp"
#include "passes/RenderResources.hpp"

using namespace brassica;

TEST_CASE("RenderGraph Blackboard resource passing and generic pass subtypes") {
	FrameGraph           fg;
	FrameGraphBlackboard blackboard;

	FrameGraphTexture2D mockTex;
	vk::Extent2D        extent{1280, 720};
	vk::Format          format = vk::Format::eB8G8R8A8Unorm;

	FrameGraphResource swapchainRes = fg.import("SwapchainImage", {extent, format}, std::move(mockTex));
	CHECK(fg.isValid(swapchainRes));

	blackboard.add<SwapchainData>() = SwapchainData{.target = swapchainRes};
	CHECK(blackboard.has<SwapchainData>());
	CHECK(blackboard.get<SwapchainData>().target == swapchainRes);

	// Pass 1: GradientPass (renders background)
	const auto& mockGradientPass = fg.addCallbackPass<GradientPassData>(
		"MockGradientPass",
		[&](FrameGraph::Builder& builder, GradientPassData& data) {
			data.target = builder.create<FrameGraphTexture2D>(
				"GradientBackground",
				FrameGraphTexture2D::Desc{
					.extent = extent,
					.format = vk::Format::eR8G8B8A8Unorm,
					.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
				}
			);
			data.target = builder.write(data.target, static_cast<uint32_t>(TextureUsage::ColorAttachment));
			builder.setSideEffect();
		},
		[](const GradientPassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<GradientPassData>() = mockGradientPass;

	CHECK(blackboard.has<GradientPassData>());
	CHECK(fg.isValid(blackboard.get<GradientPassData>().target));

	// Pass 2: MeshCubePass (G-Buffer MRT Pass)
	const auto& mockMeshCubePass = fg.addCallbackPass<MeshCubePassData>(
		"MockMeshCubePass",
		[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
			data.positionTarget = builder.create<FrameGraphTexture2D>(
				"GBuffer_Position",
				FrameGraphTexture2D::Desc{.extent = extent, .format = vk::Format::eR16G16B16A16Sfloat}
			);
			data.positionTarget = builder.write(data.positionTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

			data.normalTarget = builder.create<FrameGraphTexture2D>(
				"GBuffer_Normal",
				FrameGraphTexture2D::Desc{.extent = extent, .format = vk::Format::eR16G16B16A16Sfloat}
			);
			data.normalTarget = builder.write(data.normalTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

			data.albedoTarget = builder.create<FrameGraphTexture2D>(
				"GBuffer_Albedo",
				FrameGraphTexture2D::Desc{.extent = extent, .format = vk::Format::eR8G8B8A8Unorm}
			);
			data.albedoTarget = builder.write(data.albedoTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

			data.depthTarget = builder.create<FrameGraphTexture2D>(
				"GBuffer_Depth",
				FrameGraphTexture2D::Desc{.extent = extent, .format = vk::Format::eD32Sfloat}
			);
			data.depthTarget = builder.write(data.depthTarget, static_cast<uint32_t>(TextureUsage::DepthStencilAttachment));

			builder.setSideEffect();
		},
		[](const MeshCubePassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<MeshCubePassData>() = mockMeshCubePass;
	blackboard.add<GBufferData>() = GBufferData{
		.positionTarget = mockMeshCubePass.positionTarget,
		.normalTarget = mockMeshCubePass.normalTarget,
		.albedoTarget = mockMeshCubePass.albedoTarget,
		.depthTarget = mockMeshCubePass.depthTarget
	};

	CHECK(blackboard.has<MeshCubePassData>());
	CHECK(blackboard.has<GBufferData>());

	// Pass 3: DeferredPass (Deferred Lighting Pass reading G-Buffer + Background -> Swapchain)
	const auto& gbufferData = blackboard.get<GBufferData>();
	const auto& gradData = blackboard.get<GradientPassData>();
	const auto& scData = blackboard.get<SwapchainData>();

	const auto& mockDeferredPass = fg.addCallbackPass<DeferredPassData>(
		"MockDeferredPass",
		[&](FrameGraph::Builder& builder, DeferredPassData& data) {
			builder.read(gbufferData.positionTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
			builder.read(gbufferData.normalTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
			builder.read(gbufferData.albedoTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
			builder.read(gradData.target, static_cast<uint32_t>(TextureUsage::SampledShaderRead));

			data.target = builder.write(scData.target, static_cast<uint32_t>(TextureUsage::ColorAttachment));
			builder.setSideEffect();
		},
		[](const DeferredPassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<DeferredPassData>() = mockDeferredPass;

	CHECK(blackboard.has<DeferredPassData>());
	CHECK(fg.isValid(blackboard.get<DeferredPassData>().target));

	// Pass 4: Compute Pass using FrameGraphTexture3D and Indirect SSBO
	const auto& mockComputePass = fg.addCallbackPass<ComputeTestPassData>(
		"MockComputeTestPass",
		[&](FrameGraph::Builder& builder, ComputeTestPassData& data) {
			data.outputTexture3D = builder.create<FrameGraphTexture3D>(
				"Volume3D",
				FrameGraphTexture3D::Desc{.extent = {16, 16, 16}, .format = vk::Format::eR8G8B8A8Unorm}
			);
			data.outputTexture3D = builder.write(data.outputTexture3D, static_cast<uint32_t>(TextureUsage::StorageWrite));

			data.outputSSBO = builder.create<FrameGraphSSBO>(
				"IndirectBufferSSBO",
				FrameGraphSSBO::Desc{.size = 1024}
			);
			data.outputSSBO = builder.write(data.outputSSBO, static_cast<uint32_t>(BufferUsage::StorageWrite | BufferUsage::Indirect));
			builder.setSideEffect();
		},
		[](const ComputeTestPassData&, FrameGraphPassResources&, void*) {}
	);
	blackboard.add<ComputeTestPassData>() = mockComputePass;

	CHECK(blackboard.has<ComputeTestPassData>());
	CHECK(fg.isValid(blackboard.get<ComputeTestPassData>().outputTexture3D));
	CHECK(fg.isValid(blackboard.get<ComputeTestPassData>().outputSSBO));

	// Compile framegraph
	fg.compile();
}
