#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Shader.hpp"
#include "passes/RenderResources.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "vulkan/vulkan.hpp"

TEST_CASE("Vulkan-Hpp RenderResources Types") {
	brassica::FrameGraphTexture::Desc desc;
	desc.extent = vk::Extent2D{1920, 1080};
	desc.format = vk::Format::eR8G8B8A8Unorm;

	CHECK(desc.extent.width == 1920);
	CHECK(desc.extent.height == 1080);
	CHECK(desc.format == vk::Format::eR8G8B8A8Unorm);

	vk::Image fakeImage{reinterpret_cast<VkImage>(0x12345)};
	vk::ImageView fakeView{reinterpret_cast<VkImageView>(0x6789A)};

	brassica::FrameGraphTexture texture(fakeImage, fakeView);
	CHECK(texture.image == fakeImage);
	CHECK(texture.imageView == fakeView);
}

TEST_CASE("FrameUBO Struct Size and Alignment") {
	CHECK(sizeof(brassica::FrameUBO) == 16);
	CHECK(alignof(brassica::FrameUBO) == 16);
}

TEST_CASE("Shader Stage Flag Bits") {
	brassica::VertexShader vertShader;
	CHECK(vertShader.GetStageFlag() == vk::ShaderStageFlagBits::eVertex);

	brassica::FragmentShader fragShader;
	CHECK(fragShader.GetStageFlag() == vk::ShaderStageFlagBits::eFragment);

	brassica::ComputeShader computeShader;

	brassica::MeshShader meshShader;
	CHECK(meshShader.GetStageFlag() == vk::ShaderStageFlagBits::eMeshEXT);

	brassica::TaskShader taskShader;
	CHECK(taskShader.GetStageFlag() == vk::ShaderStageFlagBits::eTaskEXT);

	auto vertStageInfo = vertShader.GetStageCreateInfo();
	CHECK(vertStageInfo.stage == vk::ShaderStageFlagBits::eVertex);
	CHECK(std::string(vertStageInfo.pName) == "main");

	auto fragStageInfo = fragShader.GetStageCreateInfo();
	CHECK(fragStageInfo.stage == vk::ShaderStageFlagBits::eFragment);
	CHECK(std::string(fragStageInfo.pName) == "main");

	auto computeStageInfo = computeShader.GetStageCreateInfo();
	CHECK(computeStageInfo.stage == vk::ShaderStageFlagBits::eCompute);
	CHECK(std::string(computeStageInfo.pName) == "main");

	auto meshStageInfo = meshShader.GetStageCreateInfo();
	CHECK(meshStageInfo.stage == vk::ShaderStageFlagBits::eMeshEXT);
	CHECK(std::string(meshStageInfo.pName) == "main");

	auto taskStageInfo = taskShader.GetStageCreateInfo();
	CHECK(taskStageInfo.stage == vk::ShaderStageFlagBits::eTaskEXT);
	CHECK(std::string(taskStageInfo.pName) == "main");
}

TEST_CASE("GLSL 4.6 Shader Compilation with Shaderc targeting Vulkan 1.4") {
	std::string vertSource = R"(#version 460
layout(location = 0) out vec2 outUV;
void main() {
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
)";

	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	options.SetOptimizationLevel(shaderc_optimization_level_performance);
	options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
	options.SetTargetSpirv(shaderc_spirv_version_1_6);

	auto result = compiler.CompileGlslToSpv(
		vertSource,
		shaderc_glsl_vertex_shader,
		"TestVertexShader",
		options
	);

	CHECK(result.GetCompilationStatus() == shaderc_compilation_status_success);
	std::vector<uint32_t> spirv(result.cbegin(), result.cend());
	CHECK(!spirv.empty());
}

TEST_CASE("GLSL 4.6 Mesh Shader Compilation with Shaderc") {
	std::string meshSource = R"(#version 460
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	uint frameIndex;
	uint globalSeed;
	uint frameRandom;
} ubo;

void main() {
	SetMeshOutputsEXT(3, 1);
	gl_MeshVerticesEXT[0].gl_Position = vec4(-0.5, -0.5, 0.0, 1.0);
	gl_MeshVerticesEXT[1].gl_Position = vec4( 0.5, -0.5, 0.0, 1.0);
	gl_MeshVerticesEXT[2].gl_Position = vec4( 0.0,  0.5, 0.0, 1.0);
	gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
)";

	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	options.SetOptimizationLevel(shaderc_optimization_level_performance);
	options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
	options.SetTargetSpirv(shaderc_spirv_version_1_6);

	auto result = compiler.CompileGlslToSpv(
		meshSource,
		shaderc_glsl_mesh_shader,
		"TestMeshShader",
		options
	);

	CHECK(result.GetCompilationStatus() == shaderc_compilation_status_success);
	std::vector<uint32_t> spirv(result.cbegin(), result.cend());
	CHECK(!spirv.empty());
}
