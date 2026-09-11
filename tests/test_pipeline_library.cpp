#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
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
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Unorm};
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
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Unorm};
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
		std::array<vk::Format, 1>            colorFormats{vk::Format::eR8G8B8A8Unorm};
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
		std::array<vk::Format, 1>      colorFormats{vk::Format::eR8G8B8A8Unorm};
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
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Unorm};
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
		std::array<vk::Format, 1>       colorFormats{vk::Format::eR8G8B8A8Unorm};
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
