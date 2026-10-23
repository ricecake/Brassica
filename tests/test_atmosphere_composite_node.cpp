#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/VolumetricFroxelGenNode.hpp"
#include "passes/VolumetricRenderNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	// Stands in for TerrainNode + DeferredNode + CloudRenderNode: writes the G-buffer inputs
	// VolumetricRenderNode reads and the HdrColor target it modifies.
	struct FakeSceneProducer {
		float     albedoAlpha = 0.0f;
		glm::vec3 relPos{0.0f};

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Create<GBufferPosition>()
				.Desc(graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat))
				.Clear({relPos.x, relPos.y, relPos.z, 0.0f});
			r.Create<GBufferAlbedo>()
				.Desc(graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Srgb))
				.Clear({0.5f, 0.5f, 0.5f, albedoAlpha});
			r.Create<GBufferNormal>().Desc(graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat));
			r.Create<GBufferMaterial>().Desc(graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm));
			r.Create<GBufferDepth>().Desc(graph::DepthBufferDesc(ctx.width, ctx.height));
			r.Create<HdrColor>()
				.Desc(graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat))
				.Clear({1.0f, 1.0f, 1.0f, 1.0f});
		}

		void Execute(graph::NodeContext&) {}
	};

	struct AtmosphereBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};
		void*                   frameUboMapped{nullptr};
		vk::Buffer              atmosphereUboBuffer{};
		VmaAllocation           atmosphereUboAllocation{};
		void*                   atmosphereUboMapped{nullptr};

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                        sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	AtmosphereBindlessSet CreateAtmosphereBindlessSet(vk::Device device, VmaAllocator allocator) {
		AtmosphereBindlessSet result;

		std::array<vk::DescriptorSetLayoutBinding, 5> uboBindings{};
		uboBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		uboBindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		uboBindings[2]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eStorageBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		uboBindings[3]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eStorageBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		uboBindings[4]
			.setBinding(4)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(uboBindings);
		result.frameLayout = device.createDescriptorSetLayout(frameLayoutInfo);

		std::array<vk::DescriptorPoolSize, 2> framePoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 3},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2},
		};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSizes);
		framePoolInfo.setMaxSets(1);
		result.framePool = device.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(result.framePool);
		frameAllocInfo.setSetLayouts(result.frameLayout);
		vk::DescriptorSet frameSet = device.allocateDescriptorSets(frameAllocInfo).front();

		auto createUboBuffer = [&](VkDeviceSize size, vk::Buffer& buf, VmaAllocation& alloc) {
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = size;
			bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VkBuffer          rawBuffer = VK_NULL_HANDLE;
			VmaAllocationInfo allocResultInfo{};
			vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &rawBuffer, &alloc, &allocResultInfo);
			buf = rawBuffer;
			return allocResultInfo.pMappedData;
		};

		FrameUBO frameUbo{};
		frameUbo.cameraPosition = glm::vec4(0.0f, -10.0f, 0.0f, 1.0f);
		result.frameUboMapped = createUboBuffer(sizeof(FrameUBO), result.frameUboBuffer, result.frameUboAllocation);
		if (result.frameUboMapped) {
			std::memcpy(result.frameUboMapped, &frameUbo, sizeof(FrameUBO));
		}

		AtmospherePushConstants atmosphere{};
		result.atmosphereUboMapped =
			createUboBuffer(sizeof(AtmospherePushConstants), result.atmosphereUboBuffer, result.atmosphereUboAllocation);
		if (result.atmosphereUboMapped) {
			std::memcpy(result.atmosphereUboMapped, &atmosphere, sizeof(AtmospherePushConstants));
		}

		std::array<vk::DescriptorBufferInfo, 2> bufferDescs{
			vk::DescriptorBufferInfo{result.frameUboBuffer, 0, sizeof(FrameUBO)},
			vk::DescriptorBufferInfo{result.atmosphereUboBuffer, 0, sizeof(AtmospherePushConstants)},
		};
		std::array<vk::WriteDescriptorSet, 2> uboWrites{};
		uboWrites[0]
			.setDstSet(frameSet)
			.setDstBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setBufferInfo(bufferDescs[0]);
		uboWrites[1]
			.setDstSet(frameSet)
			.setDstBinding(4)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setBufferInfo(bufferDescs[1]);
		device.updateDescriptorSets(uboWrites, nullptr);

		std::array<vk::DescriptorSetLayoutBinding, 4> layoutBindings{};
		layoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[1]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[2]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eStorageImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[3]
			.setBinding(5)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 4> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(layoutBindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 4> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 16},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 16},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 16},
		};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setMaxSets(1);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		result.pool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(result.pool);
		allocInfo.setSetLayouts(result.layout);
		vk::DescriptorSet set = device.allocateDescriptorSets(allocInfo).front();

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		result.sampler = device.createSampler(samplerInfo);

		std::array<vk::DescriptorImageInfo, 4> samplerImageInfos{};
		for (auto& info : samplerImageInfos) {
			info.setSampler(result.sampler);
		}
		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.setDstSet(set);
		samplerWrite.setDstBinding(2);
		samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfos);
		device.updateDescriptorSets(samplerWrite, {});

		result.bindings.set = set;
		result.bindings.layout = result.layout;
		result.bindings.sampledImage2DBinding = 0;
		result.bindings.samplerBinding = 2;
		result.bindings.storageImageBinding = 3;
		result.bindings.sampledImage3DBinding = 5;
		result.bindings.frameSet = frameSet;
		result.bindings.frameSetLayout = result.frameLayout;
		return result;
	}

	void DestroyAtmosphereBindlessSet(vk::Device device, VmaAllocator allocator, AtmosphereBindlessSet& s) {
		device.destroySampler(s.sampler);
		device.destroyDescriptorPool(s.pool);
		device.destroyDescriptorSetLayout(s.layout);
		if (s.frameUboBuffer && s.frameUboAllocation) {
			vmaDestroyBuffer(allocator, s.frameUboBuffer, s.frameUboAllocation);
		}
		if (s.atmosphereUboBuffer && s.atmosphereUboAllocation) {
			vmaDestroyBuffer(allocator, s.atmosphereUboBuffer, s.atmosphereUboAllocation);
		}
		device.destroyDescriptorPool(s.framePool);
		device.destroyDescriptorSetLayout(s.frameLayout);
	}

	float HalfToFloat(std::uint16_t h) {
		std::uint32_t sign = (h & 0x8000u) << 16u;
		std::uint32_t exponent = (h >> 10u) & 0x1Fu;
		std::uint32_t mantissa = h & 0x3FFu;

		std::uint32_t bits;
		if (exponent == 0u) {
			if (mantissa == 0u) {
				bits = sign;
			} else {
				int e = -1;
				do {
					mantissa <<= 1u;
					++e;
				} while ((mantissa & 0x400u) == 0u);
				mantissa &= 0x3FFu;
				bits = sign | ((static_cast<std::uint32_t>(112 - e) << 23u)) | (mantissa << 13u);
			}
		} else if (exponent == 0x1Fu) {
			bits = sign | 0x7F800000u | (mantissa << 13u);
		} else {
			bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
		}

		float result;
		std::memcpy(&result, &bits, sizeof(float));
		return result;
	}

} // namespace

TEST_CASE("Unified VolumetricRenderNode executes cleanly in render graph with no validation errors") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				device.GetQueueFamily(),
			}
		);
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

		render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		AtmosphereBindlessSet           bindlessSet = CreateAtmosphereBindlessSet(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		graph::PhysicalExecutionBackend backend(registry);

		TransmittanceLUTNode transNode;
		transNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary, .physicalRegistry = &registry});
		MultiScatteringLUTNode multiNode;
		multiNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary, .physicalRegistry = &registry});
		SkyViewLUTNode skyNode;
		skyNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary, .physicalRegistry = &registry});

		VolumetricFroxelGenNode froxelGenNode;
		froxelGenNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary, .physicalRegistry = &registry});

		VolumetricRenderNode volumetricRenderNode;
		volumetricRenderNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary, .physicalRegistry = &registry});
		Shader::ClearConstants();

		graph::Graph graph;
		graph.Register<FakeSceneProducer>(FakeSceneProducer{});
		graph.RegisterRef(transNode);
		graph.RegisterRef(multiNode);
		graph.RegisterRef(skyNode);
		graph.RegisterRef(froxelGenNode);
		graph.RegisterRef(volumetricRenderNode);

		graph::FrameContext ctx{.width = 64, .height = 64};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
		vkCmd.end();

		const auto& schedule = graph.GetSchedule();
		REQUIRE(schedule.stages.size() >= 2);

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		pipelineLibrary.Reset();
		volumetricRenderNode.Destroy(vkDevice);
		froxelGenNode.Destroy(vkDevice);
		skyNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		DestroyAtmosphereBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
