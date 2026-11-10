#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "doctest/doctest.h"

#include "MinimalDevice.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

using namespace brassica;

namespace {

	// Real files on disk (not CompileFromSource) specifically because Shader::Recompile
	// requires a non-empty filePath -- CompileFromSource never sets one, so it can't be used to
	// simulate the hot-reload path this file's generation-invalidation tests need. Mirrors
	// test_shader_watcher.cpp's own temp-directory pattern.
	struct TempShaderFiles {
		std::filesystem::path dir = std::filesystem::current_path() / "test_pipeline_library_dir";
		std::filesystem::path vertPath = dir / "trivial.vert";
		std::filesystem::path fragPath = dir / "trivial.frag";

		TempShaderFiles() {
			std::filesystem::create_directories(dir);
			WriteVert();
			WriteFrag();
		}

		~TempShaderFiles() { std::filesystem::remove_all(dir); }

		void WriteVert() const {
			std::ofstream f(vertPath);
			f << "#version 450\n"
				 "void main() {\n"
				 "    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
				 "}\n";
		}

		void WriteFrag() const {
			std::ofstream f(fragPath);
			f << "#version 450\n"
				 "layout(location = 0) out vec4 outColor;\n"
				 "void main() {\n"
				 "    outColor = vec4(1.0);\n"
				 "}\n";
		}
	};

	// A fragment shader that actually *uses* a push constant -- TempShaderFiles's trivial.frag
	// deliberately doesn't touch one, so a request with the wrong (or missing)
	// pushConstantRanges against *that* shader would never trigger a real validation error: the
	// validation layer only complains when the shader's SPIR-V actually declares a push-constant
	// block the pipeline layout doesn't cover for that stage. This is what let a real bug
	// (DeferredNode::Execute building a GraphicsPipelineRequest with no pushConstantRanges at
	// all, despite deferred.frag using one) compile and pass every existing case here silently.
	struct PushConstantShaderFile {
		std::filesystem::path dir = std::filesystem::current_path() / "test_pipeline_library_pc_dir";
		std::filesystem::path fragPath = dir / "uses_push_constant.frag";

		PushConstantShaderFile() {
			std::filesystem::create_directories(dir);
			std::ofstream f(fragPath);
			f << "#version 450\n"
				 "layout(push_constant) uniform PC { vec4 color; } pc;\n"
				 "layout(location = 0) out vec4 outColor;\n"
				 "void main() {\n"
				 "    outColor = pc.color;\n"
				 "}\n";
		}

		~PushConstantShaderFile() { std::filesystem::remove_all(dir); }
	};

} // namespace

TEST_CASE("PipelineLibrary::Resolve builds a fresh, distinct pipeline on every call (unchanged, uncached contract)") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(device.GetDevice(), files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(device.GetDevice(), files.fragPath.string()));

	{
		render::PipelineLibrary library(device.GetDevice(), nullptr);

		std::array<GraphicsShader*, 2>  stages{&vert, &frag};
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineRequest request{
			.stages = stages,
			.state = render::GraphicsPipelineState{.enableShadingRate = false}, // MinimalDevice has no VRS
			.colorFormats = colorFormats,
		};

		render::ResolvedPipeline first = library.Resolve(request);
		render::ResolvedPipeline second = library.Resolve(request);
		REQUIRE(first.pipeline);
		REQUIRE(second.pipeline);
		CHECK(static_cast<VkPipeline>(first.pipeline) != static_cast<VkPipeline>(second.pipeline));

		// Resolve's contract is "caller owns it" -- unlike ResolveCached, PipelineLibrary never
		// tracked these, so it's on this test to clean them up.
		device.GetDevice().destroyPipeline(first.pipeline);
		device.GetDevice().destroyPipelineLayout(first.layout);
		device.GetDevice().destroyPipeline(second.pipeline);
		device.GetDevice().destroyPipelineLayout(second.layout);
	}

	vert.Destroy(device.GetDevice());
	frag.Destroy(device.GetDevice());

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

TEST_CASE("PipelineLibrary::ResolveCached returns the same pipeline for a repeated identical request") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(device.GetDevice(), files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(device.GetDevice(), files.fragPath.string()));

	{
		render::PipelineLibrary library(device.GetDevice(), nullptr);

		std::array<GraphicsShader*, 2>  stages{&vert, &frag};
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineRequest request{
			.stages = stages,
			.state = render::GraphicsPipelineState{.enableShadingRate = false}, // MinimalDevice has no VRS
			.colorFormats = colorFormats,
		};

		render::ResolvedPipeline first = library.ResolveCached(request);
		render::ResolvedPipeline second = library.ResolveCached(request);
		REQUIRE(first.pipeline);
		CHECK(static_cast<VkPipeline>(first.pipeline) == static_cast<VkPipeline>(second.pipeline));
		CHECK(static_cast<VkPipelineLayout>(first.layout) == static_cast<VkPipelineLayout>(second.layout));
	}

	vert.Destroy(device.GetDevice());
	frag.Destroy(device.GetDevice());

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// DeferredNode is the first real consumer of setLayouts/pushConstantRanges in a
// PipelineLibrary::ResolveCached request (GradientNode's is empty on both) -- neither BuildKey's
// nor Build's handling of them had any real coverage before this case.
TEST_CASE("PipelineLibrary::ResolveCached distinguishes requests that differ only in setLayouts/pushConstantRanges") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(vkDevice, files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(vkDevice, files.fragPath.string()));

	{
		vk::DescriptorSetLayout setLayoutA = vkDevice.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{});
		vk::DescriptorSetLayout setLayoutB = vkDevice.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{});

		render::PipelineLibrary library(vkDevice, nullptr);

		std::array<GraphicsShader*, 2>       stages{&vert, &frag};
		std::array<vk::Format, 1>            colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineState        state{.enableShadingRate = false}; // MinimalDevice has no VRS
		std::array<vk::PushConstantRange, 1> pushRange{
			vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, 16}
		};

		std::array<vk::DescriptorSetLayout, 1> setLayoutsA{setLayoutA};
		render::GraphicsPipelineRequest        requestA{
				   .stages = stages,
				   .state = state,
				   .colorFormats = colorFormats,
				   .setLayouts = setLayoutsA,
				   .pushConstantRanges = pushRange,
        };

		std::array<vk::DescriptorSetLayout, 1> setLayoutsB{setLayoutB};
		render::GraphicsPipelineRequest        requestB{
				   .stages = stages,
				   .state = state,
				   .colorFormats = colorFormats,
				   .setLayouts = setLayoutsB, // only this differs from requestA
				   .pushConstantRanges = pushRange,
        };

		render::ResolvedPipeline resolvedA = library.ResolveCached(requestA);
		render::ResolvedPipeline resolvedAAgain = library.ResolveCached(requestA);
		render::ResolvedPipeline resolvedB = library.ResolveCached(requestB);

		REQUIRE(resolvedA.pipeline);
		REQUIRE(resolvedB.pipeline);
		CHECK(static_cast<VkPipeline>(resolvedA.pipeline) == static_cast<VkPipeline>(resolvedAAgain.pipeline));
		CHECK(static_cast<VkPipeline>(resolvedA.pipeline) != static_cast<VkPipeline>(resolvedB.pipeline));
		CHECK(static_cast<VkPipelineLayout>(resolvedA.layout) != static_cast<VkPipelineLayout>(resolvedB.layout));

		library.Reset();
		vkDevice.destroyDescriptorSetLayout(setLayoutA);
		vkDevice.destroyDescriptorSetLayout(setLayoutB);
	}

	vert.Destroy(vkDevice);
	frag.Destroy(vkDevice);

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

TEST_CASE("PipelineLibrary::ResolveCached distinguishes requests that differ only in BlendMode") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(vkDevice, files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(vkDevice, files.fragPath.string()));

	{
		render::PipelineLibrary library(vkDevice, nullptr);

		std::array<GraphicsShader*, 2> stages{&vert, &frag};
		std::array<vk::Format, 1>      colorFormats{vk::Format::eR8G8B8A8Srgb};

		render::GraphicsPipelineRequest opaqueRequest{
			.stages = stages,
			.state = render::GraphicsPipelineState{.blendMode = render::BlendMode::Opaque, .enableShadingRate = false},
			.colorFormats = colorFormats,
		};
		render::GraphicsPipelineRequest alphaRequest{
			.stages = stages,
			.state =
				render::GraphicsPipelineState{.blendMode = render::BlendMode::AlphaBlend, .enableShadingRate = false},
			.colorFormats = colorFormats,
		};

		render::ResolvedPipeline opaqueResolved = library.ResolveCached(opaqueRequest);
		render::ResolvedPipeline opaqueAgain = library.ResolveCached(opaqueRequest);
		render::ResolvedPipeline alphaResolved = library.ResolveCached(alphaRequest);

		REQUIRE(opaqueResolved.pipeline);
		REQUIRE(alphaResolved.pipeline);
		CHECK(static_cast<VkPipeline>(opaqueResolved.pipeline) == static_cast<VkPipeline>(opaqueAgain.pipeline));
		CHECK(static_cast<VkPipeline>(opaqueResolved.pipeline) != static_cast<VkPipeline>(alphaResolved.pipeline));

		library.Reset();
	}

	vert.Destroy(vkDevice);
	frag.Destroy(vkDevice);

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Regression test for the exact bug DeferredNode::Execute shipped with: a shader that declares
// and reads a push-constant block, resolved through a request that never sets
// pushConstantRanges at all. Real pipeline creation, real validation layers -- this is the case
// none of the other tests in this file could have caught, since their fixture shaders never
// actually use a push constant.
TEST_CASE(
	"PipelineLibrary flags a real validation error when a request omits pushConstantRanges a shader actually "
	"uses, and is clean once they're supplied"
) {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	TempShaderFiles        vertFile;
	PushConstantShaderFile fragFile;
	VertexShader           vert;
	FragmentShader         frag;
	REQUIRE(vert.CompileVertexFromFile(vkDevice, vertFile.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(vkDevice, fragFile.fragPath.string()));

	{
		render::PipelineLibrary library(vkDevice, nullptr);

		std::array<GraphicsShader*, 2> stages{&vert, &frag};
		std::array<vk::Format, 1>      colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineState  state{.enableShadingRate = false}; // MinimalDevice has no VRS

		// Omitted entirely -- request.pushConstantRanges defaults empty, exactly what
		// DeferredNode::Execute shipped with before this was caught on real hardware.
		render::GraphicsPipelineRequest badRequest{.stages = stages, .state = state, .colorFormats = colorFormats};
		render::ResolvedPipeline        badResolved = library.Resolve(badRequest);
		// The layout/pipeline still get created (createGraphicsPipeline doesn't fail the call
		// outright, it's a validation-layer diagnostic, not a VkResult error) -- the missing
		// range is what CHECK(GetValidationErrorCount() > 0) below actually catches.
		CHECK(badResolved.pipeline);
		if (badResolved.pipeline) {
			vkDevice.destroyPipeline(badResolved.pipeline);
		}
		if (badResolved.layout) {
			vkDevice.destroyPipelineLayout(badResolved.layout);
		}

		CHECK(device.GetValidationErrorCount() > 0);
		MESSAGE("(expected) validation errors above are the bug this test exists to catch");

		// GetValidationErrorCount is cumulative for the device's whole lifetime, not per-call --
		// snapshot it here so the check below is "the good request added zero new errors", not
		// "zero errors ever", which the bad request above already made impossible.
		std::uint32_t errorsBeforeGoodRequest = device.GetValidationErrorCount();

		std::array<vk::PushConstantRange, 1> pushConstantRanges{
			vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(float) * 4}
		};
		render::GraphicsPipelineRequest goodRequest{
			.stages = stages,
			.state = state,
			.colorFormats = colorFormats,
			.pushConstantRanges = pushConstantRanges,
		};
		render::ResolvedPipeline goodResolved = library.Resolve(goodRequest);
		REQUIRE(goodResolved.pipeline);
		vkDevice.destroyPipeline(goodResolved.pipeline);
		vkDevice.destroyPipelineLayout(goodResolved.layout);

		CHECK(device.GetValidationErrorCount() == errorsBeforeGoodRequest);
	}

	vert.Destroy(vkDevice);
	frag.Destroy(vkDevice);
}

TEST_CASE("PipelineLibrary::ResolveCached rebuilds after a shader hot-reloads, without the caller changing anything") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(device.GetDevice(), files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(device.GetDevice(), files.fragPath.string()));

	{
		render::PipelineLibrary library(device.GetDevice(), nullptr);

		std::array<GraphicsShader*, 2>  stages{&vert, &frag};
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineRequest request{
			.stages = stages,
			.state = render::GraphicsPipelineState{.enableShadingRate = false}, // MinimalDevice has no VRS
			.colorFormats = colorFormats,
		};

		render::ResolvedPipeline before = library.ResolveCached(request);
		REQUIRE(before.pipeline);

		// Simulates the exact effect ShaderWatcher::ProcessPendingReloads has on a changed file
		// -- recompiling in place bumps Shader::GetGeneration(), which is baked into
		// PipelineLibrary's cache key, so the *same* request now misses.
		std::uint64_t generationBefore = frag.GetGeneration();
		REQUIRE(frag.Recompile(device.GetDevice()));
		CHECK(frag.GetGeneration() != generationBefore);

		render::ResolvedPipeline after = library.ResolveCached(request);
		REQUIRE(after.pipeline);
		CHECK(static_cast<VkPipeline>(before.pipeline) != static_cast<VkPipeline>(after.pipeline));

		// Re-resolving again (no further reload) is a cache hit against the *new* entry.
		render::ResolvedPipeline again = library.ResolveCached(request);
		CHECK(static_cast<VkPipeline>(after.pipeline) == static_cast<VkPipeline>(again.pipeline));
	}

	vert.Destroy(device.GetDevice());
	frag.Destroy(device.GetDevice());

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

TEST_CASE("PipelineLibrary::Reset destroys every cached pipeline/layout with no validation errors") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	TempShaderFiles files;
	VertexShader    vert;
	FragmentShader  frag;
	REQUIRE(vert.CompileVertexFromFile(device.GetDevice(), files.vertPath.string()));
	REQUIRE(frag.CompileFragmentFromFile(device.GetDevice(), files.fragPath.string()));

	{
		render::PipelineLibrary library(device.GetDevice(), nullptr);

		std::array<GraphicsShader*, 2>  stages{&vert, &frag};
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Srgb};
		render::GraphicsPipelineRequest request{
			.stages = stages,
			.state = render::GraphicsPipelineState{.enableShadingRate = false}, // MinimalDevice has no VRS
			.colorFormats = colorFormats,
		};

		render::ResolvedPipeline resolved = library.ResolveCached(request);
		REQUIRE(resolved.pipeline);

		// Explicit, not just the destructor -- mirrors Engine::Cleanup()'s call, which must run
		// while the device is still alive.
		library.Reset();

		// A cache-cleared library resolving the identical request builds again rather than
		// returning something already destroyed.
		render::ResolvedPipeline rebuilt = library.ResolveCached(request);
		REQUIRE(rebuilt.pipeline);

		library.Reset();
	}

	vert.Destroy(device.GetDevice());
	frag.Destroy(device.GetDevice());

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Empirical proof for GBufferDepth's eD32Sfloat -> eD32SfloatS8Uint move (render::StencilMode
// needs a real stencil aspect to write into) before anything depends on it. Two things this
// specifically has to prove about a *combined* depth-stencil image, neither covered by the
// AsAttachment proof in test_physical_backend.cpp (which deliberately used a plain eD32Sfloat
// target): (1) a combined-aspect view can be used as pDepthAttachment alone, with no
// pStencilAttachment set, the exact shape DynamicRenderingWrapper::Begin already builds; (2) a
// *separate*, depth-only view of that same image -- PhysicalTexture::GetSampledView()'s whole
// reason for existing -- can be written into a sampled-image descriptor and actually sampled, on
// real hardware, with the depth value round-tripping correctly.
TEST_CASE(
	"A combined eD32SfloatS8Uint image can be depth-attached via its combined view and sampled "
	"via a separate depth-only view, with no validation errors"
) {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device   vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();

	constexpr vk::Format kFormat = vk::Format::eD32SfloatS8Uint;

	{
		// Mirrors GBufferDepth::kUsage (include/resources/core.hpp) exactly.
		vk::ImageCreateInfo imageInfo{};
		imageInfo.setImageType(vk::ImageType::e2D)
			.setFormat(kFormat)
			.setExtent(vk::Extent3D{1, 1, 1})
			.setMipLevels(1)
			.setArrayLayers(1)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setTiling(vk::ImageTiling::eOptimal)
			.setUsage(
				vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			)
			.setInitialLayout(vk::ImageLayout::eUndefined);
		VkImageCreateInfo       rawImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
		VmaAllocationCreateInfo imageAllocInfo{};
		imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		VkImage       rawImage{};
		VmaAllocation imageAlloc{};
		REQUIRE(vmaCreateImage(allocator, &rawImageInfo, &imageAllocInfo, &rawImage, &imageAlloc, nullptr) == VK_SUCCESS);
		vk::Image image{rawImage};

		// Combined-aspect view -- what DynamicRenderingWrapper::Begin uses for pDepthAttachment
		// (PhysicalTexture::m_view).
		vk::ImageViewCreateInfo combinedViewInfo{};
		combinedViewInfo.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(kFormat)
			.setSubresourceRange(
				vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}
			);
		vk::ImageView combinedView = vkDevice.createImageView(combinedViewInfo);

		// Depth-only view -- what PhysicalTexture::GetSampledView() returns for this format
		// (PhysicalTexture::m_sampledView).
		vk::ImageViewCreateInfo depthOnlyViewInfo{};
		depthOnlyViewInfo.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(kFormat)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
		vk::ImageView depthOnlyView = vkDevice.createImageView(depthOnlyViewInfo);

		static constexpr const char* kVertSource = R"(
			#version 450
			void main() {
				vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
				gl_Position = vec4(pos, 0.0, 1.0);
			}
		)";
		static constexpr const char* kWriterFragSource = R"(
			#version 450
			void main() {
				gl_FragDepth = 0.6;
			}
		)";
		static constexpr const char* kSamplerFragSource = R"(
			#version 450
			layout(set = 0, binding = 0) uniform sampler2D uDepthSampler;
			layout(location = 0) out vec4 outColor;
			void main() {
				outColor = vec4(texture(uDepthSampler, vec2(0.5, 0.5)).r, 0.0, 0.0, 1.0);
			}
		)";

		VertexShader   vert;
		FragmentShader writerFrag;
		FragmentShader samplerFrag;
		REQUIRE(vert.CompileFromSource(vkDevice, kVertSource, shaderc_glsl_vertex_shader, "depthstencil_test_vert"));
		REQUIRE(
			writerFrag.CompileFromSource(
				vkDevice,
				kWriterFragSource,
				shaderc_glsl_fragment_shader,
				"depthstencil_test_writer_frag"
			)
		);
		REQUIRE(
			samplerFrag.CompileFromSource(
				vkDevice,
				kSamplerFragSource,
				shaderc_glsl_fragment_shader,
				"depthstencil_test_sampler_frag"
			)
		);

		render::PipelineLibrary library(vkDevice, nullptr);

		std::array<GraphicsShader*, 2>  writerStages{&vert, &writerFrag};
		render::GraphicsPipelineRequest writerRequest{
			.stages = writerStages,
			.state = render::GraphicsPipelineState{
				.cullMode = vk::CullModeFlagBits::eNone,
				.depthTest = true,
				.depthWrite = true,
				.depthCompareOp = vk::CompareOp::eAlways,
			},
			.colorFormats = {},
			.depthFormat = kFormat,
		};
		render::ResolvedPipeline writerResolved = library.Resolve(writerRequest);
		REQUIRE(writerResolved.pipeline);

		// One-binding sampled-image descriptor set, mirroring test_physical_backend.cpp's
		// minimal single-binding storage-image fixture, for a sampler2D instead.
		vk::DescriptorSetLayoutBinding samplerBinding{};
		samplerBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eFragment);
		vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo{};
		samplerLayoutInfo.setBindings(samplerBinding);
		vk::DescriptorSetLayout samplerSetLayout = vkDevice.createDescriptorSetLayout(samplerLayoutInfo);

		vk::DescriptorPoolSize       samplerPoolSize{vk::DescriptorType::eCombinedImageSampler, 1};
		vk::DescriptorPoolCreateInfo samplerPoolInfo{};
		samplerPoolInfo.setPoolSizes(samplerPoolSize).setMaxSets(1);
		vk::DescriptorPool samplerPool = vkDevice.createDescriptorPool(samplerPoolInfo);

		vk::DescriptorSetAllocateInfo samplerSetAllocInfo{};
		samplerSetAllocInfo.setDescriptorPool(samplerPool).setSetLayouts(samplerSetLayout);
		vk::DescriptorSet samplerSet = vkDevice.allocateDescriptorSets(samplerSetAllocInfo).front();

		vk::SamplerCreateInfo samplerInfo{};
		vk::Sampler            sampler = vkDevice.createSampler(samplerInfo);

		vk::DescriptorImageInfo samplerImageInfo{};
		samplerImageInfo.setSampler(sampler)
			.setImageView(depthOnlyView)
			.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.setDstSet(samplerSet)
			.setDstBinding(0)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setImageInfo(samplerImageInfo);
		vkDevice.updateDescriptorSets(samplerWrite, {});

		std::array<GraphicsShader*, 2>       samplerStages{&vert, &samplerFrag};
		std::array<vk::Format, 1>            samplerColorFormats{vk::Format::eR8G8B8A8Unorm};
		std::array<vk::DescriptorSetLayout, 1> samplerSetLayouts{samplerSetLayout};
		render::GraphicsPipelineRequest       samplerRequest{
			.stages = samplerStages,
			.state = render::GraphicsPipelineState{.cullMode = vk::CullModeFlagBits::eNone},
			.colorFormats = samplerColorFormats,
			.setLayouts = samplerSetLayouts,
		};
		render::ResolvedPipeline samplerResolved = library.Resolve(samplerRequest);
		REQUIRE(samplerResolved.pipeline);

		vk::ImageCreateInfo colorImageInfo{};
		colorImageInfo.setImageType(vk::ImageType::e2D)
			.setFormat(vk::Format::eR8G8B8A8Unorm)
			.setExtent(vk::Extent3D{1, 1, 1})
			.setMipLevels(1)
			.setArrayLayers(1)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setTiling(vk::ImageTiling::eOptimal)
			.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc)
			.setInitialLayout(vk::ImageLayout::eUndefined);
		VkImageCreateInfo       rawColorImageInfo = static_cast<VkImageCreateInfo>(colorImageInfo);
		VmaAllocationCreateInfo colorImageAllocInfo{};
		colorImageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		VkImage       rawColorImage{};
		VmaAllocation colorImageAlloc{};
		REQUIRE(
			vmaCreateImage(allocator, &rawColorImageInfo, &colorImageAllocInfo, &rawColorImage, &colorImageAlloc, nullptr) ==
			VK_SUCCESS
		);
		vk::Image image2{rawColorImage};
		vk::ImageViewCreateInfo colorViewInfo{};
		colorViewInfo.setImage(image2)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(vk::Format::eR8G8B8A8Unorm)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		vk::ImageView colorView = vkDevice.createImageView(colorViewInfo);

		vk::CommandPool cmdPool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);
		vk::CommandBuffer cmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, 1})
				.front();
		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		// -- Pass 1: write depth=0.6 via the combined view as pDepthAttachment alone (no
		// pStencilAttachment) -- exactly DynamicRenderingWrapper::Begin's shape for a plain Read-
		// or-Write depth access on a combined-format resource.
		vk::ImageMemoryBarrier2 toDepthAttachment{};
		toDepthAttachment.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
			.setSrcAccessMask({})
			.setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests)
			.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
			.setImage(image)
			.setSubresourceRange(
				vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}
			);
		vk::DependencyInfo toDepthAttachmentDep{};
		toDepthAttachmentDep.setImageMemoryBarriers(toDepthAttachment);
		cmd.pipelineBarrier2(toDepthAttachmentDep);

		vk::RenderingAttachmentInfo depthAttachmentInfo{};
		depthAttachmentInfo.setImageView(combinedView)
			.setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearDepthStencilValue{1.0f, 0});
		vk::RenderingInfo writerRenderingInfo{};
		writerRenderingInfo.setRenderArea(vk::Rect2D{{0, 0}, {1, 1}})
			.setLayerCount(1)
			.setPDepthAttachment(&depthAttachmentInfo);
		cmd.beginRendering(writerRenderingInfo);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, writerResolved.pipeline);
		cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
		cmd.setScissor(0, vk::Rect2D{{0, 0}, {1, 1}});
		cmd.draw(3, 1, 0, 0);
		cmd.endRendering();

		// -- Pass 2: sample the depth-only view, output to a color attachment.
		vk::ImageMemoryBarrier2 toShaderRead{};
		toShaderRead.setSrcStageMask(vk::PipelineStageFlagBits2::eLateFragmentTests)
			.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
			.setOldLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setImage(image)
			.setSubresourceRange(
				vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}
			);
		vk::ImageMemoryBarrier2 colorToAttachment{};
		colorToAttachment.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
			.setSrcAccessMask({})
			.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
			.setImage(image2)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		std::array<vk::ImageMemoryBarrier2, 2> toPass2{toShaderRead, colorToAttachment};
		vk::DependencyInfo                     toPass2Dep{};
		toPass2Dep.setImageMemoryBarriers(toPass2);
		cmd.pipelineBarrier2(toPass2Dep);

		vk::RenderingAttachmentInfo colorAttachmentInfo{};
		colorAttachmentInfo.setImageView(colorView)
			.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f));
		vk::RenderingInfo samplerRenderingInfo{};
		samplerRenderingInfo.setRenderArea(vk::Rect2D{{0, 0}, {1, 1}})
			.setLayerCount(1)
			.setColorAttachments(colorAttachmentInfo);
		cmd.beginRendering(samplerRenderingInfo);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, samplerResolved.pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, samplerResolved.layout, 0, samplerSet, {});
		cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
		cmd.setScissor(0, vk::Rect2D{{0, 0}, {1, 1}});
		cmd.draw(3, 1, 0, 0);
		cmd.endRendering();

		vk::BufferCreateInfo readbackInfo{};
		readbackInfo.setSize(4).setUsage(vk::BufferUsageFlagBits::eTransferDst);
		VkBufferCreateInfo      rawReadbackInfo = static_cast<VkBufferCreateInfo>(readbackInfo);
		VmaAllocationCreateInfo readbackAllocInfo{};
		readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		readbackAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          rawReadback{};
		VmaAllocation     readbackAlloc{};
		VmaAllocationInfo readbackAllocResultInfo{};
		REQUIRE(
			vmaCreateBuffer(allocator, &rawReadbackInfo, &readbackAllocInfo, &rawReadback, &readbackAlloc, &readbackAllocResultInfo) ==
			VK_SUCCESS
		);

		vk::ImageMemoryBarrier2 colorToTransferSrc{};
		colorToTransferSrc.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
			.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setImage(image2)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		vk::DependencyInfo colorToTransferSrcDep{};
		colorToTransferSrcDep.setImageMemoryBarriers(colorToTransferSrc);
		cmd.pipelineBarrier2(colorToTransferSrcDep);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
			.setImageExtent(vk::Extent3D{1, 1, 1});
		cmd.copyImageToBuffer(image2, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer{rawReadback}, copyRegion);

		cmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(cmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		vmaInvalidateAllocation(allocator, readbackAlloc, 0, VK_WHOLE_SIZE);
		const auto* pixel = static_cast<const std::uint8_t*>(readbackAllocResultInfo.pMappedData);
		// 0.6 * 255 = 153, exactly representable in R8Unorm with no rounding ambiguity at this
		// value (153/255 == 0.6 to within float precision).
		CHECK(static_cast<int>(pixel[0]) == 153);

		vmaDestroyBuffer(allocator, rawReadback, readbackAlloc);
		vkDevice.destroyCommandPool(cmdPool);
		vkDevice.destroyPipeline(writerResolved.pipeline);
		vkDevice.destroyPipelineLayout(writerResolved.layout);
		vkDevice.destroyPipeline(samplerResolved.pipeline);
		vkDevice.destroyPipelineLayout(samplerResolved.layout);
		vkDevice.destroySampler(sampler);
		vkDevice.destroyDescriptorPool(samplerPool);
		vkDevice.destroyDescriptorSetLayout(samplerSetLayout);
		vkDevice.destroyImageView(colorView);
		vmaDestroyImage(allocator, static_cast<VkImage>(image2), colorImageAlloc);
		vkDevice.destroyImageView(combinedView);
		vkDevice.destroyImageView(depthOnlyView);
		vmaDestroyImage(allocator, static_cast<VkImage>(image), imageAlloc);
		vert.Destroy(vkDevice);
		writerFrag.Destroy(vkDevice);
		samplerFrag.Destroy(vkDevice);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Empirical proof for BlendMode::WeightedOit before anything production depends on it: two
// overlapping fragments, drawn as two separate draw calls into a 1x1 accum + 1x1 revealage
// target, must accumulate via real fixed-function blend hardware to the exact McGuire WBOIT
// values -- not just "a pipeline with 2 attachments gets created without erroring."
//
// Draw 1: color (1,0,0), weight 0.5. Draw 2: color (0,1,0), weight 0.25. Accum starts cleared to
// 0 and uses ONE/ONE additive blend on both color and alpha, so accum.rgb = sum(color_i*weight_i)
// and accum.a = sum(weight_i). Revealage starts cleared to 1 (nothing drawn yet = fully
// revealed) and uses srcFactor=ZERO, dstFactor=ONE_MINUS_SRC_{COLOR,ALPHA}, so it ends at
// product(1 - weight_i) per channel. Expected: accum = (0.5, 0.25, 0, 0.75), revealage = 0.375
// (all four channels, since the shader writes weight uniformly across rgba).
TEST_CASE(
	"BlendMode::WeightedOit produces the exact McGuire accum/revealage values from real "
	"fixed-function blend hardware across two draw calls"
) {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device   vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();

	constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;

	{
		static constexpr const char* kVertSource = R"(
			#version 450
			void main() {
				vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
				gl_Position = vec4(pos, 0.0, 1.0);
			}
		)";
		static constexpr const char* kFragSource = R"(
			#version 450
			layout(push_constant) uniform PC { vec4 color; float weight; } pc;
			layout(location = 0) out vec4 outAccum;
			layout(location = 1) out vec4 outRevealage;
			void main() {
				outAccum = vec4(pc.color.rgb * pc.weight, pc.weight);
				outRevealage = vec4(pc.weight);
			}
		)";

		VertexShader   vert;
		FragmentShader frag;
		REQUIRE(vert.CompileFromSource(vkDevice, kVertSource, shaderc_glsl_vertex_shader, "wboit_test_vert"));
		REQUIRE(frag.CompileFromSource(vkDevice, kFragSource, shaderc_glsl_fragment_shader, "wboit_test_frag"));

		auto makeAttachmentImage = [&](vk::Image& outImage, VmaAllocation& outAlloc, vk::ImageView& outView) {
			vk::ImageCreateInfo imageInfo{};
			imageInfo.setImageType(vk::ImageType::e2D)
				.setFormat(kFormat)
				.setExtent(vk::Extent3D{1, 1, 1})
				.setMipLevels(1)
				.setArrayLayers(1)
				.setSamples(vk::SampleCountFlagBits::e1)
				.setTiling(vk::ImageTiling::eOptimal)
				.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc)
				.setInitialLayout(vk::ImageLayout::eUndefined);
			VkImageCreateInfo       rawImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			VkImage rawImage{};
			REQUIRE(vmaCreateImage(allocator, &rawImageInfo, &allocInfo, &rawImage, &outAlloc, nullptr) == VK_SUCCESS);
			outImage = vk::Image{rawImage};

			vk::ImageViewCreateInfo viewInfo{};
			viewInfo.setImage(outImage)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(kFormat)
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			outView = vkDevice.createImageView(viewInfo);
		};

		vk::Image     accumImage, revealageImage;
		VmaAllocation accumAlloc{}, revealageAlloc{};
		vk::ImageView accumView, revealageView;
		makeAttachmentImage(accumImage, accumAlloc, accumView);
		makeAttachmentImage(revealageImage, revealageAlloc, revealageView);

		render::PipelineLibrary library(vkDevice, nullptr);

		struct WboitPushConstants {
			float color[4];
			float weight;
		};

		std::array<GraphicsShader*, 2> stages{&vert, &frag};
		std::array<vk::Format, 2>      colorFormats{kFormat, kFormat};
		std::array<vk::PushConstantRange, 1> pushRanges{
			vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(WboitPushConstants)}
		};
		render::GraphicsPipelineRequest request{
			.stages = stages,
			.state = render::GraphicsPipelineState{
				// eNone, not the struct default eBack -- this is a fullscreen triangle built
				// from gl_VertexIndex with no particular winding guarantee, same convention as
				// every other fullscreen pass in this codebase (CloudCompositeNode, DeferredNode).
				.cullMode = vk::CullModeFlagBits::eNone,
				.blendMode = render::BlendMode::WeightedOit,
				.enableShadingRate = false, // MinimalDevice has no VRS
			},
			.colorFormats = colorFormats,
			.pushConstantRanges = pushRanges,
		};
		render::ResolvedPipeline resolved = library.Resolve(request);
		REQUIRE(resolved.pipeline);

		vk::CommandPool cmdPool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);
		vk::CommandBuffer cmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, 1})
				.front();
		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		std::array<vk::ImageMemoryBarrier2, 2> toColorAttachment{};
		for (std::size_t i = 0; i < 2; ++i) {
			toColorAttachment[i]
				.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
				.setSrcAccessMask({})
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eUndefined)
				.setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setImage(i == 0 ? accumImage : revealageImage)
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		}
		vk::DependencyInfo toColorAttachmentDep{};
		toColorAttachmentDep.setImageMemoryBarriers(toColorAttachment);
		cmd.pipelineBarrier2(toColorAttachmentDep);

		vk::RenderingAttachmentInfo accumAttachment{};
		accumAttachment.setImageView(accumView)
			.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f));
		vk::RenderingAttachmentInfo revealageAttachment{};
		revealageAttachment.setImageView(revealageView)
			.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setClearValue(vk::ClearColorValue(1.0f, 1.0f, 1.0f, 1.0f));
		std::array<vk::RenderingAttachmentInfo, 2> colorAttachments{accumAttachment, revealageAttachment};

		vk::RenderingInfo renderingInfo{};
		renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, {1, 1}}).setLayerCount(1).setColorAttachments(colorAttachments);
		cmd.beginRendering(renderingInfo);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
		cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
		cmd.setScissor(0, vk::Rect2D{{0, 0}, {1, 1}});

		WboitPushConstants push1{.color = {1.0f, 0.0f, 0.0f, 1.0f}, .weight = 0.5f};
		cmd.pushConstants(
			resolved.layout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			sizeof(WboitPushConstants),
			&push1
		);
		cmd.draw(3, 1, 0, 0);

		WboitPushConstants push2{.color = {0.0f, 1.0f, 0.0f, 1.0f}, .weight = 0.25f};
		cmd.pushConstants(
			resolved.layout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			sizeof(WboitPushConstants),
			&push2
		);
		cmd.draw(3, 1, 0, 0);

		cmd.endRendering();

		auto makeReadbackBuffer = [&](VkBuffer& outBuffer, VmaAllocation& outAlloc, VmaAllocationInfo& outAllocInfo) {
			vk::BufferCreateInfo bufferInfo{};
			bufferInfo.setSize(4 * sizeof(std::uint16_t)).setUsage(vk::BufferUsageFlagBits::eTransferDst);
			VkBufferCreateInfo       rawBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
			VmaAllocationCreateInfo  allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			REQUIRE(
				vmaCreateBuffer(allocator, &rawBufferInfo, &allocInfo, &outBuffer, &outAlloc, &outAllocInfo) ==
				VK_SUCCESS
			);
		};

		VkBuffer          accumReadback{}, revealageReadback{};
		VmaAllocation     accumReadbackAlloc{}, revealageReadbackAlloc{};
		VmaAllocationInfo accumReadbackInfo{}, revealageReadbackInfo{};
		makeReadbackBuffer(accumReadback, accumReadbackAlloc, accumReadbackInfo);
		makeReadbackBuffer(revealageReadback, revealageReadbackAlloc, revealageReadbackInfo);

		std::array<vk::ImageMemoryBarrier2, 2> toTransferSrc{};
		for (std::size_t i = 0; i < 2; ++i) {
			toTransferSrc[i]
				.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setImage(i == 0 ? accumImage : revealageImage)
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
		}
		vk::DependencyInfo toTransferSrcDep{};
		toTransferSrcDep.setImageMemoryBarriers(toTransferSrc);
		cmd.pipelineBarrier2(toTransferSrcDep);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
			.setImageExtent(vk::Extent3D{1, 1, 1});
		cmd.copyImageToBuffer(accumImage, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer{accumReadback}, copyRegion);
		cmd.copyImageToBuffer(
			revealageImage,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::Buffer{revealageReadback},
			copyRegion
		);

		cmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(cmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		vmaInvalidateAllocation(allocator, accumReadbackAlloc, 0, VK_WHOLE_SIZE);
		vmaInvalidateAllocation(allocator, revealageReadbackAlloc, 0, VK_WHOLE_SIZE);

		// Self-contained IEEE-754 half -> float32 bit conversion (not relying on any particular
		// math library's internal/unstable API) -- every value this test produces (0, 0.25,
		// 0.375, 0.5, 0.75) is an exactly-representable normal half, so the subnormal/inf/nan
		// branches below are just for correctness, not exercised by this test's actual values.
		auto halfToFloat = [](std::uint16_t h) -> float {
			std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
			std::uint32_t exponent = (h & 0x7C00u) >> 10;
			std::uint32_t mantissa = h & 0x03FFu;
			std::uint32_t bits;
			if (exponent == 0) {
				if (mantissa == 0) {
					bits = sign;
				} else {
					std::int32_t e = 1;
					while ((mantissa & 0x0400u) == 0) {
						mantissa <<= 1;
						--e;
					}
					mantissa &= 0x03FFu;
					bits = sign | (static_cast<std::uint32_t>(e + (127 - 15)) << 23) | (mantissa << 13);
				}
			} else if (exponent == 0x1F) {
				bits = sign | 0x7F800000u | (mantissa << 13);
			} else {
				bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
			}
			float f;
			std::memcpy(&f, &bits, sizeof(f));
			return f;
		};

		auto readHalf4 = [&](const void* mapped) {
			std::array<std::uint16_t, 4> half{};
			std::memcpy(half.data(), mapped, sizeof(half));
			std::array<float, 4> out{};
			for (std::size_t i = 0; i < 4; ++i) {
				out[i] = halfToFloat(half[i]);
			}
			return out;
		};

		std::array<float, 4> accumValues = readHalf4(accumReadbackInfo.pMappedData);
		std::array<float, 4> revealageValues = readHalf4(revealageReadbackInfo.pMappedData);

		constexpr float kEpsilon = 0.01f;
		CHECK(accumValues[0] == doctest::Approx(0.5f).epsilon(kEpsilon));
		CHECK(accumValues[1] == doctest::Approx(0.25f).epsilon(kEpsilon));
		CHECK(accumValues[2] == doctest::Approx(0.0f).epsilon(kEpsilon));
		CHECK(accumValues[3] == doctest::Approx(0.75f).epsilon(kEpsilon));
		for (float v : revealageValues) {
			CHECK(v == doctest::Approx(0.375f).epsilon(kEpsilon));
		}

		vmaDestroyBuffer(allocator, accumReadback, accumReadbackAlloc);
		vmaDestroyBuffer(allocator, revealageReadback, revealageReadbackAlloc);
		vkDevice.destroyCommandPool(cmdPool);
		vkDevice.destroyPipeline(resolved.pipeline);
		vkDevice.destroyPipelineLayout(resolved.layout);
		vkDevice.destroyImageView(accumView);
		vkDevice.destroyImageView(revealageView);
		vmaDestroyImage(allocator, static_cast<VkImage>(accumImage), accumAlloc);
		vmaDestroyImage(allocator, static_cast<VkImage>(revealageImage), revealageAlloc);
		vert.Destroy(vkDevice);
		frag.Destroy(vkDevice);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
