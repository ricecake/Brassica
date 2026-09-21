#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "doctest/doctest.h"

#include "MinimalDevice.hpp"
#include "Shader.hpp"
#include "render/PipelineLibrary.hpp"

// evaluate_terrain_normal (shaders/common.glsl) is suspected of a sign error: sgreenhusted
// reported terrain lighting looking like the light is coming from the wrong direction on slopes.
// Deriving the standard height-field normal by hand (Tx x Tz for a surface (x, H(x,z), z)) gives
// normalize(-dH/dx, C, -dH/dz); the function currently builds normalize(grad.x, 2*eps, grad.z)
// with grad.x/grad.z proportional to +dH/dx, +dH/dz (the tetrahedron-gradient identity sum_i
// e_i (e_i . grad f) = 4*grad f for these 4 unit vectors, verified by hand), i.e. missing the
// negation on both horizontal components.
//
// This test proves that empirically against the real function (not a hand-derived symbolic
// re-implementation) by comparing evaluate_terrain_normal's output to an independent
// central-difference estimate of dH/dx, dH/dz computed via evaluate_terrain directly, dispatched
// through shaders/terrain_normal_probe.comp on a real device via MinimalDevice -- this needs no
// mesh shader/ray query/AS, so it runs for real on this Mac's MoltenVK, not just a stub.
//
// This is written to assert the CORRECT convention, so it is expected to FAIL against
// evaluate_terrain_normal before the sign fix lands, and PASS after -- see the terrain_gen.comp
// call site (`normal = evaluate_terrain_normal(...)`) for where this feeds real per-vertex
// terrain normals and, from there, the deferred lighting N.L term.

namespace {

	struct ProbePushConstants {
		float spatial_scale;
		float min_height;
		float max_height;
		float ridge_weight;
		float biome_bleed;
		float phase;
		float warp_strength;
		float eps;
	};

} // namespace

TEST_CASE("evaluate_terrain_normal's horizontal tilt matches the real height gradient's sign") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	// Non-lattice-aligned points (avoids symmetric noise-gradient zeros) spread widely enough that
	// at least most of them land on a real slope rather than a local flat spot.
	const std::vector<std::array<float, 3>> testPoints = {
		{4.37f, 0.0f, -3.29f},
		{-6.73f, 0.0f, 2.58f},
		{-2.11f, 0.0f, 1.29f},
		{5.03f, 0.0f, 4.87f},
		{-3.6f, 0.0f, -1.9f},
		{8.15f, 0.0f, -6.33f},
		{-0.93f, 0.0f, 6.02f},
		{3.14f, 0.0f, 2.71f},
	};
	const std::uint32_t numPoints = static_cast<std::uint32_t>(testPoints.size());

	{
		std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
		for (std::uint32_t i = 0; i < 3; ++i) {
			bindings[i]
				.setBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute);
		}
		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		vk::DescriptorSetLayout setLayout = vkDevice.createDescriptorSetLayout(layoutInfo);

		vk::DescriptorPoolSize      poolSize{vk::DescriptorType::eStorageBuffer, 3};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setMaxSets(1);
		vk::DescriptorPool pool = vkDevice.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo setAllocInfo{};
		setAllocInfo.setDescriptorPool(pool);
		setAllocInfo.setSetLayouts(setLayout);
		vk::DescriptorSet set = vkDevice.allocateDescriptorSets(setAllocInfo).front();

		auto makeBuffer = [&](VkDeviceSize size) {
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = size;
			bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VkBuffer          buffer = VK_NULL_HANDLE;
			VmaAllocation     allocation = nullptr;
			VmaAllocationInfo resultInfo{};
			vmaCreateBuffer(device.GetAllocator(), &bufferInfo, &allocInfo, &buffer, &allocation, &resultInfo);
			return std::tuple{buffer, allocation, resultInfo};
		};

		const VkDeviceSize bufSize = static_cast<VkDeviceSize>(numPoints) * sizeof(float) * 4;
		auto [pointsBuffer, pointsAlloc, pointsInfo] = makeBuffer(bufSize);
		auto [normalsBuffer, normalsAlloc, normalsInfo] = makeBuffer(bufSize);
		auto [gradientsBuffer, gradientsAlloc, gradientsInfo] = makeBuffer(bufSize);

		REQUIRE(pointsInfo.pMappedData != nullptr);
		auto* pointsData = static_cast<float*>(pointsInfo.pMappedData);
		for (std::uint32_t i = 0; i < numPoints; ++i) {
			pointsData[i * 4 + 0] = testPoints[i][0];
			pointsData[i * 4 + 1] = testPoints[i][1];
			pointsData[i * 4 + 2] = testPoints[i][2];
			pointsData[i * 4 + 3] = 0.0f;
		}

		std::array<vk::DescriptorBufferInfo, 3> bufferInfos{
			vk::DescriptorBufferInfo{pointsBuffer, 0, bufSize},
			vk::DescriptorBufferInfo{normalsBuffer, 0, bufSize},
			vk::DescriptorBufferInfo{gradientsBuffer, 0, bufSize},
		};
		std::array<vk::WriteDescriptorSet, 3> writes{};
		for (std::uint32_t i = 0; i < 3; ++i) {
			writes[i]
				.setDstSet(set)
				.setDstBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferInfos[i]);
		}
		vkDevice.updateDescriptorSets(writes, nullptr);

		brassica::ComputeShader shader;
		REQUIRE(shader.CompileComputeFromFile(vkDevice, "shaders/terrain_normal_probe.comp"));

		vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ProbePushConstants)};

		brassica::render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);
		brassica::render::ResolvedPipeline resolved = pipelineLibrary.Resolve(
			brassica::render::ComputePipelineRequest{
				.shader = &shader,
				.setLayouts = std::span(&setLayout, 1),
				.pushConstantRanges = std::span(&pcRange, 1),
			}
		);
		REQUIRE(static_cast<bool>(resolved.pipeline));

		vk::CommandPool pool2 = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);
		vk::CommandBuffer cmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool2, vk::CommandBufferLevel::ePrimary, 1})
				.front();

		ProbePushConstants push{
			.spatial_scale = 1.0f,
			.min_height = -50.0f,
			.max_height = 1000.0f,
			.ridge_weight = 0.14f,
			.biome_bleed = 0.5f,
			.phase = 0.0f,
			// 0 here, not the production call site's 0.75: the anisotropic domain warp
			// (p_warped = p + (S*p)*warp_strength inside evaluate_terrain) makes the height field
			// locally anisotropic enough that an axis-aligned central difference and the
			// tetrahedron's diagonal-offset gradient can genuinely disagree in sign at a handful of
			// points, independent of evaluate_terrain_normal's own sign convention -- confirmed by
			// hand while building this test, not a hypothesis. Keeping warp_strength at 0 isolates
			// the specific invariant this test checks (grad.x/grad.z's sign relative to the real
			// height gradient) from that separate, expected property of the warp technique.
			.warp_strength = 0.0f,
			.eps = 0.005f,
		};

		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, set, {});
		cmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		cmd.dispatch(numPoints, 1, 1);
		cmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(cmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		auto* normalsData = static_cast<float*>(normalsInfo.pMappedData);
		auto* gradientsData = static_cast<float*>(gradientsInfo.pMappedData);
		REQUIRE(normalsData != nullptr);
		REQUIRE(gradientsData != nullptr);

		int checksPerformed = 0;
		for (std::uint32_t i = 0; i < numPoints; ++i) {
			float nx = normalsData[i * 4 + 0];
			float ny = normalsData[i * 4 + 1];
			float nz = normalsData[i * 4 + 2];
			float dHdx = gradientsData[i * 4 + 0];
			float dHdz = gradientsData[i * 4 + 1];

			CHECK(ny > 0.0f); // up-facing regardless of horizontal tilt direction

			// Correct up-facing height-field normal is normalize(-dH/dx, C, -dH/dz): the normal's
			// horizontal component must have the OPPOSITE sign of the height gradient along that axis.
			if (std::abs(dHdx) > 1e-4f) {
				INFO("point ", i, " dHdx=", dHdx, " nx=", nx);
				CHECK((nx > 0.0f) != (dHdx > 0.0f));
				++checksPerformed;
			}
			if (std::abs(dHdz) > 1e-4f) {
				INFO("point ", i, " dHdz=", dHdz, " nz=", nz);
				CHECK((nz > 0.0f) != (dHdz > 0.0f));
				++checksPerformed;
			}
		}
		// Guards against every point silently landing on a degenerate flat spot, which would make
		// the CHECKs above vacuously pass without exercising anything.
		REQUIRE(checksPerformed >= static_cast<int>(numPoints));

		vkDevice.destroyPipeline(resolved.pipeline);
		vkDevice.destroyPipelineLayout(resolved.layout);
		shader.Destroy(vkDevice);
		vkDevice.destroyCommandPool(pool2);
		vmaDestroyBuffer(device.GetAllocator(), pointsBuffer, pointsAlloc);
		vmaDestroyBuffer(device.GetAllocator(), normalsBuffer, normalsAlloc);
		vmaDestroyBuffer(device.GetAllocator(), gradientsBuffer, gradientsAlloc);
		vkDevice.destroyDescriptorPool(pool);
		vkDevice.destroyDescriptorSetLayout(setLayout);
	}

	CHECK(device.GetValidationErrorCount() == 0);
}
