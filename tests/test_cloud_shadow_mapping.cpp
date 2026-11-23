#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "doctest/doctest.h"

#include "MinimalDevice.hpp"
#include "Shader.hpp"

using namespace brassica;

namespace {

	constexpr float kDim = 1024.0f;

	float Mod(float x, float y) { return x - y * std::floor(x / y); }

	// C++ mirror of helpers/cloud_shadow_mapping.glsl, kept in lockstep so this half of the test
	// runs even where no Vulkan device is available (every MinimalDevice-gated test in this repo
	// skips its GPU section on this Mac -- see MinimalDevice.hpp's header comment). The GPU
	// section below re-proves the same properties through the real shader, which is the part that
	// actually matters: this mirror only guards against the two implementations drifting apart.
	float WriteLabelX(float vox, float centerX, float texelSize) {
		float gridCenter = centerX / texelSize;
		float windowStart = gridCenter - kDim * 0.5f;
		float target = vox + 0.5f;
		float cellForVox = windowStart + Mod(target - windowStart, kDim);
		return cellForVox * texelSize;
	}

	float ReadUVNumerator(float labelX, float texelSize) {
		float gridPos = labelX / texelSize;
		return Mod(gridPos, kDim);
	}

	// Mirror of cloudShadowLabel's X component (p.xz - max(0,p.y)/raySunY * sunDir.xz). Only the
	// X component is mirrored/tested -- same reasoning as the round-trip test above using only
	// vox.x: the formula treats X and Z identically and independently, so testing one axis
	// covers the other by construction.
	float LabelX(float px, float py, float sunDirX, float raySunY) {
		return px - (std::max(0.0f, py) / raySunY) * sunDirX;
	}

	// Writes a temp .comp file so Shader::CompileComputeFromFile exercises the real
	// #include-resolving pipeline (helpers/cloud_shadow_mapping.glsl), the same TempShaderFiles
	// pattern test_pipeline_library.cpp uses and for the same reason: CompileFromSource has no
	// filePath, and more importantly here, this test specifically wants the production include
	// resolver, not shaderc's own.
	struct TempShaderFile {
		std::filesystem::path dir = std::filesystem::current_path() / "test_cloud_shadow_mapping_dir";
		std::filesystem::path path = dir / "verify_mapping.comp";

		TempShaderFile() {
			std::filesystem::create_directories(dir);
			std::ofstream f(path);
			f << R"(
#version 450
#include "helpers/cloud_shadow_mapping.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ResultBuffer {
	float results[];
};

layout(push_constant) uniform PC {
	float centerX;
	float centerZ;
	float shadowMapExtent;
	uint  baseOffset;
} pc;

void main() {
	uint vox = gl_GlobalInvocationID.x;
	if (vox >= 1024u) return;

	vec2 centerXZ = vec2(pc.centerX, pc.centerZ);
	vec2 label = cloudShadowWriteLabel(ivec2(int(vox), 0), centerXZ, pc.shadowMapExtent);
	vec2 uv = cloudShadowUV(label, pc.shadowMapExtent);

	// Round-trip identity: the label the writer computes for this exact texel must read back,
	// through the reader's camera-independent formula, to this exact texel's center -- for any
	// centerXZ. This is the fix for the camera-motion ghosting: the reader has no centerXZ
	// parameter at all, so once this holds, a texel's meaning can never drift with the camera.
	float readBack = uv.x * CLOUD_SHADOW_MAP_DIM;
	float expected = float(vox) + 0.5;
	results[pc.baseOffset + vox] = abs(readBack - expected);
}
)";
		}

		~TempShaderFile() { std::filesystem::remove_all(dir); }
	};

	struct PushConstants {
		float         centerX;
		float         centerZ;
		float         shadowMapExtent;
		std::uint32_t baseOffset;
	};

	// Regression coverage for the bug sgreenhusted found on real hardware: cloudShadowLabel
	// originally used raw p.y with no clamp, so a point below the y=0 plane (underwater, a low
	// valley) got the full ray-angle correction scaled by 1/raySunY -- ~20x p.y at raySunY's
	// grazing-angle floor of 0.05. A few metres of depth change could swing the lookup across
	// unrelated, wrapped-around texels: jagged shadow edges with no real cloud behind them, and
	// spurious transmittance variation feeding false god rays in the froxel pass. This dispatch
	// computes cloudShadowLabel for one fixed XZ/sunDir/raySunY across a range of y values
	// (`baseOffset` selects which) and writes label.x out directly (not an error metric, since
	// there's no "expected" closed form here beyond the clamp itself) for the C++ side to check.
	struct LabelClampShaderFile {
		std::filesystem::path dir = std::filesystem::current_path() / "test_cloud_shadow_label_dir";
		std::filesystem::path path = dir / "verify_label_clamp.comp";

		LabelClampShaderFile() {
			std::filesystem::create_directories(dir);
			std::ofstream f(path);
			f << R"(
#version 450
#include "helpers/cloud_shadow_mapping.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ResultBuffer {
	float results[];
};

layout(push_constant) uniform PC {
	float px;
	float sunDirX;
	float raySunY;
	uint  baseOffset;
} pc;

void main() {
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= 64u) return;

	// py sweeps from -630 to 0 in steps of 10 -- all at or below the plane, where the label
	// must stay pinned to px regardless of depth.
	float py = -630.0 + float(idx) * 10.0;
	vec3 sunDir = vec3(pc.sunDirX, sqrt(max(0.0, 1.0 - pc.sunDirX * pc.sunDirX)), 0.0);
	vec2 label = cloudShadowLabel(vec3(pc.px, py, 0.0), sunDir, pc.raySunY);
	results[pc.baseOffset + idx] = label.x;
}
)";
		}

		~LabelClampShaderFile() { std::filesystem::remove_all(dir); }
	};

	struct LabelClampPushConstants {
		float         px;
		float         sunDirX;
		float         raySunY;
		std::uint32_t baseOffset;
	};

} // namespace

TEST_CASE("cloudShadowWriteLabel/cloudShadowUV round-trip, C++ mirror (always runs, no GPU needed)") {
	const float shadowMapExtent = 320000.0f;
	const float texelSize = shadowMapExtent / kDim;

	// A spread of camera centers, including some far outside [0, extent) and some that aren't
	// multiples of the texel size -- centerXZ needs no snapping for this to hold.
	const std::vector<float> centers{0.0f, 1.0f, -1.0f, 157.3f, -157.3f, 50000.0f, -1234567.0f, 999999.5f};

	for (float center : centers) {
		for (int vox = 0; vox < 1024; vox += 37) { // sampled, not exhaustive -- 1024 is cheap either way, kept short on purpose
			float label = WriteLabelX(static_cast<float>(vox), center, texelSize);
			float readBack = ReadUVNumerator(label, texelSize);
			CHECK(std::abs(readBack - (static_cast<float>(vox) + 0.5f)) < 1e-3f);
		}
	}
}

TEST_CASE("cloudShadowLabel pins to p.xz at or below the y=0 plane, C++ mirror (always runs, no GPU needed)") {
	// A grazing sun (raySunY at its 0.05 floor) and a range of depths including a fairly deep
	// one (-500m) -- exactly the combination that used to swing the lookup by tens of texels
	// before the clamp.
	const float px = 12345.0f;
	const float sunDirX = 0.95f;
	const float raySunY = 0.05f;

	const float atPlane = LabelX(px, 0.0f, sunDirX, raySunY);
	CHECK(std::abs(atPlane - px) < 1e-4f);

	for (float py : {-1.0f, -16.0f, -100.0f, -500.0f, -5000.0f}) {
		float label = LabelX(px, py, sunDirX, raySunY);
		CHECK(std::abs(label - px) < 1e-4f); // must stay pinned to px, not drift with depth
		CHECK(std::abs(label - atPlane) < 1e-4f);
	}

	// Above the plane, the clamp must NOT engage -- this should still be the full, unclamped
	// ray correction (sanity check that the fix didn't accidentally clamp both directions).
	float above = LabelX(px, 100.0f, sunDirX, raySunY);
	CHECK(std::abs(above - (px - (100.0f / raySunY) * sunDirX)) < 1e-3f);
	CHECK(std::abs(above - px) > 1.0f); // genuinely shifted, not pinned
}

TEST_CASE("cloudShadowLabel pins to p.xz at or below the y=0 plane, real shader on real hardware") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	LabelClampShaderFile shaderFile;
	ComputeShader         shader;
	REQUIRE(shader.CompileComputeFromFile(device.GetDevice(), shaderFile.path.string()));

	vk::Device   vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();

	constexpr std::size_t    kCount = 64;
	const vk::DeviceSize      byteSize = kCount * sizeof(float);

	VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo bufferAllocInfo{};
	bufferAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	VkBuffer      resultBuffer{};
	VmaAllocation resultAllocation{};
	REQUIRE(
		vmaCreateBuffer(allocator, &bufferInfo, &bufferAllocInfo, &resultBuffer, &resultAllocation, nullptr) ==
		VK_SUCCESS
	);

	vk::DescriptorSetLayoutBinding binding{};
	binding.setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(
		vk::ShaderStageFlagBits::eCompute
	);
	vk::DescriptorSetLayoutCreateInfo setLayoutInfo{};
	setLayoutInfo.setBindings(binding);
	vk::DescriptorSetLayout setLayout = vkDevice.createDescriptorSetLayout(setLayoutInfo);

	vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eStorageBuffer, 1};
	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.setPoolSizes(poolSize).setMaxSets(1);
	vk::DescriptorPool pool = vkDevice.createDescriptorPool(poolInfo);

	vk::DescriptorSetAllocateInfo setAllocInfo{};
	setAllocInfo.setDescriptorPool(pool).setSetLayouts(setLayout);
	vk::DescriptorSet set = vkDevice.allocateDescriptorSets(setAllocInfo).front();

	vk::DescriptorBufferInfo bufInfo{vk::Buffer(resultBuffer), 0, byteSize};
	vk::WriteDescriptorSet   write{};
	write.setDstSet(set).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bufInfo
	);
	vkDevice.updateDescriptorSets(write, {});

	vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(LabelClampPushConstants)};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
	vk::PipelineLayout pipelineLayout = vkDevice.createPipelineLayout(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.setStage(shader.GetStageCreateInfo()).setLayout(pipelineLayout);
	vk::Pipeline pipeline = vkDevice.createComputePipelines(vk::PipelineCache{}, pipelineInfo).value.front();

	vk::CommandPool cmdPool = vkDevice.createCommandPool(
		vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
	);
	vk::CommandBuffer vkCmd =
		vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, 1})
			.front();

	vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, set, {});

	LabelClampPushConstants pc{12345.0f, 0.95f, 0.05f, 0u};
	vkCmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
	vkCmd.dispatch(1, 1, 1); // 64 invocations, one per py sample

	vk::MemoryBarrier2 barrier{};
	barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
		.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
	vk::DependencyInfo depInfo{};
	depInfo.setMemoryBarriers(barrier);
	vkCmd.pipelineBarrier2(depInfo);

	VkBufferCreateInfo readbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	readbackInfo.size = byteSize;
	readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo readbackAllocInfo{};
	readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	readbackAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBuffer          readbackBuffer{};
	VmaAllocation     readbackAllocation{};
	VmaAllocationInfo readbackAllocResultInfo{};
	REQUIRE(
		vmaCreateBuffer(
			allocator,
			&readbackInfo,
			&readbackAllocInfo,
			&readbackBuffer,
			&readbackAllocation,
			&readbackAllocResultInfo
		) == VK_SUCCESS
	);

	vkCmd.copyBuffer(vk::Buffer(resultBuffer), vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, byteSize});
	vkCmd.end();

	vk::CommandBufferSubmitInfo cmdInfo{};
	cmdInfo.setCommandBuffer(vkCmd);
	vk::SubmitInfo2 submit{};
	submit.setCommandBufferInfos(cmdInfo);
	device.GetQueue().submit2(submit, nullptr);
	vkDevice.waitIdle();

	vmaInvalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	std::vector<float> results(kCount);
	std::memcpy(results.data(), readbackAllocResultInfo.pMappedData, byteSize);

	// idx 63 (py=0, at the plane) is the reference; idx 0..62 (py from -630 to -10) must all
	// match it exactly -- this is the real-hardware proof that depth no longer perturbs the
	// lookup once at or below the plane.
	float atPlane = results[63];
	for (std::size_t i = 0; i < kCount; ++i) {
		CHECK(std::abs(results[i] - atPlane) < 1e-2f);
	}

	vmaDestroyBuffer(allocator, readbackBuffer, readbackAllocation);
	vkDevice.destroyCommandPool(cmdPool);
	vkDevice.destroyPipeline(pipeline);
	vkDevice.destroyPipelineLayout(pipelineLayout);
	vkDevice.destroyDescriptorPool(pool);
	vkDevice.destroyDescriptorSetLayout(setLayout);
	vmaDestroyBuffer(allocator, resultBuffer, resultAllocation);
	shader.Destroy(vkDevice);

	CHECK(device.GetValidationErrorCount() == 0);
}

TEST_CASE("cloudShadowWriteLabel/cloudShadowUV round-trip, real shader on real hardware") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	TempShaderFile  shaderFile;
	ComputeShader   shader;
	REQUIRE(shader.CompileComputeFromFile(device.GetDevice(), shaderFile.path.string()));

	vk::Device    vkDevice = device.GetDevice();
	VmaAllocator  allocator = device.GetAllocator();

	const std::vector<float> centers{0.0f, 1.0f, -1.0f, 157.3f, -157.3f, 50000.0f, -1234567.0f, 999999.5f};
	const std::size_t        totalFloats = centers.size() * 1024;
	const vk::DeviceSize      byteSize = totalFloats * sizeof(float);

	VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo bufferAllocInfo{};
	bufferAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	VkBuffer      resultBuffer{};
	VmaAllocation resultAllocation{};
	REQUIRE(
		vmaCreateBuffer(allocator, &bufferInfo, &bufferAllocInfo, &resultBuffer, &resultAllocation, nullptr) ==
		VK_SUCCESS
	);

	vk::DescriptorSetLayoutBinding binding{};
	binding.setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(
		vk::ShaderStageFlagBits::eCompute
	);
	vk::DescriptorSetLayoutCreateInfo setLayoutInfo{};
	setLayoutInfo.setBindings(binding);
	vk::DescriptorSetLayout setLayout = vkDevice.createDescriptorSetLayout(setLayoutInfo);

	vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer, 1};
	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.setPoolSizes(poolSize).setMaxSets(1);
	vk::DescriptorPool pool = vkDevice.createDescriptorPool(poolInfo);

	vk::DescriptorSetAllocateInfo setAllocInfo{};
	setAllocInfo.setDescriptorPool(pool).setSetLayouts(setLayout);
	vk::DescriptorSet set = vkDevice.allocateDescriptorSets(setAllocInfo).front();

	vk::DescriptorBufferInfo bufInfo{vk::Buffer(resultBuffer), 0, byteSize};
	vk::WriteDescriptorSet    write{};
	write.setDstSet(set).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bufInfo
	);
	vkDevice.updateDescriptorSets(write, {});

	vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
	vk::PipelineLayout pipelineLayout = vkDevice.createPipelineLayout(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.setStage(shader.GetStageCreateInfo()).setLayout(pipelineLayout);
	vk::Pipeline pipeline = vkDevice.createComputePipelines(vk::PipelineCache{}, pipelineInfo).value.front();

	vk::CommandPool cmdPool = vkDevice.createCommandPool(
		vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
	);
	vk::CommandBuffer vkCmd =
		vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, 1})
			.front();

	vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, set, {});

	for (std::size_t i = 0; i < centers.size(); ++i) {
		PushConstants pc{centers[i], -centers[i] * 0.37f, 320000.0f, static_cast<std::uint32_t>(i * 1024)};
		vkCmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &pc);
		vkCmd.dispatch(16, 1, 1); // 16 * local_size_x(64) = 1024 texels
	}

	vk::MemoryBarrier2 barrier{};
	barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
		.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
	vk::DependencyInfo depInfo{};
	depInfo.setMemoryBarriers(barrier);
	vkCmd.pipelineBarrier2(depInfo);

	VkBufferCreateInfo readbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	readbackInfo.size = byteSize;
	readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo readbackAllocInfo{};
	readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	readbackAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBuffer          readbackBuffer{};
	VmaAllocation     readbackAllocation{};
	VmaAllocationInfo readbackAllocResultInfo{};
	REQUIRE(
		vmaCreateBuffer(
			allocator,
			&readbackInfo,
			&readbackAllocInfo,
			&readbackBuffer,
			&readbackAllocation,
			&readbackAllocResultInfo
		) == VK_SUCCESS
	);

	vkCmd.copyBuffer(vk::Buffer(resultBuffer), vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, byteSize});
	vkCmd.end();

	vk::CommandBufferSubmitInfo cmdInfo{};
	cmdInfo.setCommandBuffer(vkCmd);
	vk::SubmitInfo2 submit{};
	submit.setCommandBufferInfos(cmdInfo);
	device.GetQueue().submit2(submit, nullptr);
	vkDevice.waitIdle();

	vmaInvalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	std::vector<float> results(totalFloats);
	std::memcpy(results.data(), readbackAllocResultInfo.pMappedData, byteSize);

	float maxErr = 0.0f;
	for (float err : results) {
		maxErr = std::max(maxErr, err);
	}
	CHECK(maxErr < 1e-2f);

	vmaDestroyBuffer(allocator, readbackBuffer, readbackAllocation);
	vkDevice.destroyCommandPool(cmdPool);
	vkDevice.destroyPipeline(pipeline);
	vkDevice.destroyPipelineLayout(pipelineLayout);
	vkDevice.destroyDescriptorPool(pool);
	vkDevice.destroyDescriptorSetLayout(setLayout);
	vmaDestroyBuffer(allocator, resultBuffer, resultAllocation);
	shader.Destroy(vkDevice);

	CHECK(device.GetValidationErrorCount() == 0);
}
