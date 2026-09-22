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
#include "passes/AtmosphereCompositeNode.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	// Stands in for TerrainNode + DeferredNode: writes the G-buffer inputs AtmosphereCompositeNode
	// reads and the HdrColor target it modifies, with nothing in its own Execute -- this test is
	// about the new node's own pipeline/barrier/descriptor correctness, not about rendering a real
	// scene first. Albedo stays cleared to alpha=0 (no opaque surface anywhere), so every pixel
	// takes composite.frag's no-surface branch -- with the camera placed underwater (see FrameUBO
	// setup below), that's the underwater view-ray-reconstruction + evaluateAtmosphere path, real
	// exercise of the new math, not just an early discard.
	struct FakeSceneProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>,
			graph::Create<HdrColor>>;

		// Defaults (albedo.a=0 everywhere) match the original no-surface-anywhere fixture. Setting
		// albedoAlpha=1 + relPos simulates "there's opaque terrain at this camera-relative
		// position" without needing TerrainNode/mesh shaders at all.
		float     albedoAlpha = 0.0f;
		glm::vec3 relPos{0.0f};

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
					.clearColor = {relPos.x, relPos.y, relPos.z, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
					.clearColor = {0.5f, 0.5f, 0.5f, albedoAlpha},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					// Known white baseline -- lets the GPU readback test below compare "left alone
					// (discarded, above water)" against "actually fogged (submerged)" against an
					// exact starting value, instead of whatever an unspecified clear would give.
					.clearColor = {1.0f, 1.0f, 1.0f, 1.0f},
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

	// Real frame set (set 0, now 5 bindings: FrameUBO/LightingUBO/LightsBuffer/ClusterGridBuffer/
	// AtmosphereUBO) + bindless set (set 1), mirroring test_water_node.cpp's WaterBindlessSet
	// exactly, extended with binding 4 for the new AtmosphereUBO -- composite.frag's included
	// atmosphere/common.glsl statically declares and reads it, so an unbound/undefined descriptor
	// there would be a real validation gap, same reasoning as WaterBindlessSet's FrameUBO.
	struct AtmosphereBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};
		void*                   frameUboMapped{nullptr}; // lets a caller update cameraPosition between "frames"
		vk::Buffer              atmosphereUboBuffer{};
		VmaAllocation           atmosphereUboAllocation{};
		void*                   atmosphereUboMapped{nullptr}; // lets a caller update tuning values (e.g. skyConvergenceStrength) between "frames"

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                        sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	AtmosphereBindlessSet CreateAtmosphereBindlessSet(vk::Device device, VmaAllocator allocator) {
		AtmosphereBindlessSet result;

		// -- Frame set (set 0) --
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

		// Camera 10m below sea level (AtmospherePushConstants' default waterLevel is 0), identity
		// view/proj matrices (FrameUBO's own defaults) -- composite.frag's ray-reconstruction math
		// only needs a well-defined, non-degenerate direction, not a physically accurate frustum.
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

		// -- Bindless set (set 1): sampled 2D + storage 2D (LUT compute output) + sampler catalog --
		std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{};
		layoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[1]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(4) // SAMPLE_LINEAR (index 1) is used throughout the LUT/composite
			                       // shaders, unlike WaterNode's fixture which only needed index 0
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[2]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eStorageImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 3> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(layoutBindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 3> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 16},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 16},
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

		// Same sampler object at every index the shaders statically reference (SAMPLE_NEAREST=0,
		// SAMPLE_LINEAR=1) -- this fixture doesn't care about real filtering behavior, only that
		// every referenced index has a real, defined descriptor.
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

	// IEEE 754 half -> float. No existing helper for this in the codebase; small enough to just
	// write directly rather than pull in a dependency for one conversion.
	float HalfToFloat(std::uint16_t h) {
		std::uint32_t sign = (h & 0x8000u) << 16u;
		std::uint32_t exponent = (h >> 10u) & 0x1Fu;
		std::uint32_t mantissa = h & 0x3FFu;

		std::uint32_t bits;
		if (exponent == 0u) {
			if (mantissa == 0u) {
				bits = sign;
			} else {
				// Subnormal half -> normalized float.
				int e = -1;
				do {
					mantissa <<= 1u;
					++e;
				} while ((mantissa & 0x400u) == 0u);
				mantissa &= 0x3FFu;
				bits = sign | ((static_cast<std::uint32_t>(112 - e) << 23u)) | (mantissa << 13u);
			}
		} else if (exponent == 0x1Fu) {
			bits = sign | 0x7F800000u | (mantissa << 13u); // Inf/NaN
		} else {
			bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
		}

		float result;
		std::memcpy(&result, &bits, sizeof(float));
		return result;
	}

} // namespace

TEST_CASE(
	"AtmosphereCompositeNode fogs the scene via real underwater aerial perspective, at the correct phase, with no "
	"validation errors"
) {
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
		transNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		MultiScatteringLUTNode multiNode;
		multiNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		SkyViewLUTNode skyNode;
		skyNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});

		AtmosphereCompositeNode compositeNode;
		compositeNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		Shader::ClearConstants();

		graph::Graph graph;
		graph.Register<FakeSceneProducer>(FakeSceneProducer{});
		graph.RegisterRef(transNode);
		graph.RegisterRef(multiNode);
		graph.RegisterRef(skyNode);
		graph.RegisterRef(compositeNode);

		graph::FrameContext ctx{.width = 64, .height = 64};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
		vkCmd.end();

		// FakeSceneProducer (Default) -> Transmittance -> MultiScattering -> SkyView (Compute, no
		// ordering constraint against Default here since nothing in this graph reads their output
		// except AtmosphereCompositeNode) -> AtmosphereCompositeNode (SubPhase::Atmosphere, strictly
		// after Default). Exact stage count depends on how the LUT chain schedules relative to the
		// producer; what matters is the composite node lands in a later stage than the producer.
		const auto& schedule = graph.GetSchedule();
		REQUIRE(schedule.stages.size() >= 2);
		CHECK(schedule.stages.back().nodes.size() == 1);

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		pipelineLibrary.Reset();
		compositeNode.Destroy(vkDevice);
		skyNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		DestroyAtmosphereBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Stage 1 (test_atmosphere_ubo.cpp) proved the water coefficients are numerically right in
// isolation; the case above proved the real pipeline runs clean. This is the remaining gap: does
// the real, compiled, GPU-executed shader actually produce that color shift on real pixels,
// not just "validates." Renders the same fake scene (HdrColor cleared to known white) twice --
// camera above water, then submerged -- and reads HdrColor back via a real
// vkCmdCopyImageToBuffer, converting to a mapped host buffer this Mac's MoltenVK can read
// directly (no mesh shader/ray query needed, same as the case above).
TEST_CASE("AtmosphereCompositeNode produces a real wavelength-dependent color shift underwater, read back from the GPU") {
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
		transNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		MultiScatteringLUTNode multiNode;
		multiNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		SkyViewLUTNode skyNode;
		skyNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});

		AtmosphereCompositeNode compositeNode;
		compositeNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		Shader::ClearConstants();
		// Default sunDir (0,1,0) points straight at what most of this test's "looking up out of
		// the water" reconstructed rays also point at -- a worst-case resonance with the
		// strongly-forward-peaked Mie phase function (mieAnisotropy=0.8), not a realistic viewing
		// condition. A sideways sun keeps the test away from that degenerate spike.
		compositeNode.push.sunDir = glm::normalize(glm::vec3(0.8f, 0.1f, 0.2f));

		constexpr std::uint32_t kWidth = 8;
		constexpr std::uint32_t kHeight = 8;
		constexpr VkDeviceSize  kBytesPerPixel = 8; // R16G16B16A16Sfloat
		VkDeviceSize            readbackSize = static_cast<VkDeviceSize>(kWidth) * kHeight * kBytesPerPixel;

		VkBufferCreateInfo readbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		readbackInfo.size = readbackSize;
		readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo readbackAllocInfo{};
		readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		readbackAllocInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          readbackBuffer = VK_NULL_HANDLE;
		VmaAllocation     readbackAllocation = nullptr;
		VmaAllocationInfo readbackResultInfo{};
		vmaCreateBuffer(
			device.GetAllocator(),
			&readbackInfo,
			&readbackAllocInfo,
			&readbackBuffer,
			&readbackAllocation,
			&readbackResultInfo
		);
		REQUIRE(readbackResultInfo.pMappedData != nullptr);

		auto renderAndReadBack = [&](float cameraY) {
			REQUIRE(bindlessSet.frameUboMapped != nullptr);
			FrameUBO frameUbo{};
			frameUbo.cameraPosition = glm::vec4(0.0f, cameraY, 0.0f, 1.0f);
			std::memcpy(bindlessSet.frameUboMapped, &frameUbo, sizeof(FrameUBO));

			vk::CommandBuffer vkCmd = vkDevice
										  .allocateCommandBuffers(
											  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
										  )
										  .front();

			graph::Graph graph;
			graph.Register<FakeSceneProducer>(FakeSceneProducer{});
			graph.RegisterRef(transNode);
			graph.RegisterRef(multiNode);
			graph.RegisterRef(skyNode);
			graph.RegisterRef(compositeNode);

			graph::FrameContext ctx{.width = kWidth, .height = kHeight};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			backend.Execute(graph, ctx, cmd, false);

			auto hdrColorTex = registry.GetTexture<HdrColor>();
			REQUIRE(hdrColorTex != nullptr);

			// PhysicalTexture's tracked layout (BarrierTranslator's side of the world) has no idea
			// this manual barrier happened -- transition back to whatever it was before the copy,
			// so the next renderAndReadBack call's real Vulkan state still matches what the
			// registry believes going into its own render pass.
			vk::ImageLayout preCopyLayout = hdrColorTex->GetCurrentLayout();

			vk::ImageMemoryBarrier2 toTransfer{};
			toTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setOldLayout(preCopyLayout)
				.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setImage(hdrColorTex->GetImage())
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo depInfo{};
			depInfo.setImageMemoryBarriers(toTransfer);
			vkCmd.pipelineBarrier2(depInfo);

			vk::BufferImageCopy copyRegion{};
			copyRegion.setBufferOffset(0)
				.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
				.setImageExtent(vk::Extent3D{kWidth, kHeight, 1});
			vkCmd.copyImageToBuffer(hdrColorTex->GetImage(), vk::ImageLayout::eTransferSrcOptimal, readbackBuffer, copyRegion);

			vk::ImageMemoryBarrier2 restoreLayout{};
			restoreLayout.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setNewLayout(preCopyLayout)
				.setImage(hdrColorTex->GetImage())
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo restoreDepInfo{};
			restoreDepInfo.setImageMemoryBarriers(restoreLayout);
			vkCmd.pipelineBarrier2(restoreDepInfo);

			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();

			vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, readbackSize);

			std::vector<glm::vec4> pixels(static_cast<std::size_t>(kWidth) * kHeight);
			const auto*            half = static_cast<const std::uint16_t*>(readbackResultInfo.pMappedData);
			for (std::size_t i = 0; i < pixels.size(); ++i) {
				pixels[i] = glm::vec4(
					HalfToFloat(half[i * 4 + 0]),
					HalfToFloat(half[i * 4 + 1]),
					HalfToFloat(half[i * 4 + 2]),
					HalfToFloat(half[i * 4 + 3])
				);
			}

			vkDevice.freeCommandBuffers(pool, vkCmd);
			return pixels;
		};

		std::vector<glm::vec4> aboveWater = renderAndReadBack(50.0f);
		std::vector<glm::vec4> submerged = renderAndReadBack(-10.0f);

		// Above water, every pixel takes composite.frag's "no surface, above water -> discard"
		// branch (real surfaces are never present in this fake scene): HdrColor must be left
		// exactly at its known white clear value, untouched.
		for (const auto& p : aboveWater) {
			CHECK(p.r == doctest::Approx(1.0f).epsilon(0.01));
			CHECK(p.g == doctest::Approx(1.0f).epsilon(0.01));
			CHECK(p.b == doctest::Approx(1.0f).epsilon(0.01));
		}

		// Submerged, every pixel marches through a real (if per-pixel-varying) stretch of water --
		// at least 10m looking straight up, up to the 200m cap otherwise -- starting from the same
		// white baseline. Real Beer-Lambert attenuation through *any* meaningful water depth
		// should show red attenuated far more than blue: exactly the color shift, not just
		// darkening, sgreenhusted asked for.
		int realColorShiftCount = 0;
		for (const auto& p : submerged) {
			if (p.r < 0.9f) { // skip only if a pixel somehow saw ~zero water path
				CHECK(p.r < p.b);
				++realColorShiftCount;
			}
		}
		REQUIRE(realColorShiftCount >= static_cast<int>(submerged.size()) / 2);

		vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
		pipelineLibrary.Reset();
		compositeNode.Destroy(vkDevice);
		skyNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		DestroyAtmosphereBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// sgreenhusted reported (real hardware, full scene): above water, terrain and sky are blindingly
// bright/washed out, and lowering tonemap exposure all the way doesn't fix terrain/sky (only
// water darkens). The other two test cases in this file never exercised this: both use
// albedoAlpha=0 everywhere (no opaque surface), so composite.frag's *other* branch (no-surface,
// underwater) is all that's been proven. This is the has-surface branch, above water, at
// increasing camera-to-surface distances -- including the ~20-30km "distant mountain" case that
// is the actual point of the feature -- checking for NaN/Inf and runaway magnitude, not just "it
// validates."
TEST_CASE("AtmosphereCompositeNode fogging real terrain above water stays finite and bounded at increasing distance") {
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
		transNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		MultiScatteringLUTNode multiNode;
		multiNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		SkyViewLUTNode skyNode;
		skyNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});

		AtmosphereCompositeNode compositeNode;
		compositeNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		Shader::ClearConstants();
		// A typical daytime sun -- mostly overhead, not the deliberately-avoided-resonance angle
		// the other GPU test uses. If a realistic sun angle is itself what triggers the reported
		// bug, this needs to catch it, not dodge it.
		compositeNode.push.sunDir = glm::normalize(glm::vec3(0.3f, 0.7f, 0.3f));

		constexpr std::uint32_t kWidth = 4;
		constexpr std::uint32_t kHeight = 4;
		constexpr VkDeviceSize  kBytesPerPixel = 8; // R16G16B16A16Sfloat
		VkDeviceSize            readbackSize = static_cast<VkDeviceSize>(kWidth) * kHeight * kBytesPerPixel;

		VkBufferCreateInfo readbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		readbackInfo.size = readbackSize;
		readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo readbackAllocInfo{};
		readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		readbackAllocInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          readbackBuffer = VK_NULL_HANDLE;
		VmaAllocation     readbackAllocation = nullptr;
		VmaAllocationInfo readbackResultInfo{};
		vmaCreateBuffer(
			device.GetAllocator(),
			&readbackInfo,
			&readbackAllocInfo,
			&readbackBuffer,
			&readbackAllocation,
			&readbackResultInfo
		);
		REQUIRE(readbackResultInfo.pMappedData != nullptr);

		// Camera 50m above sea level, looking at opaque terrain `distanceMeters` away, straight
		// ahead (-Z, matching this fixture's identity view matrix). relPos is camera-relative, as
		// the real G-buffer stores it (deferred.frag: pos = relPos + uCameraPosition.xyz).
		auto renderAndReadBack = [&](float distanceMeters) {
			FrameUBO frameUbo{};
			frameUbo.cameraPosition = glm::vec4(0.0f, 50.0f, 0.0f, 1.0f);
			std::memcpy(bindlessSet.frameUboMapped, &frameUbo, sizeof(FrameUBO));

			vk::CommandBuffer vkCmd = vkDevice
										  .allocateCommandBuffers(
											  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
										  )
										  .front();

			graph::Graph graph;
			graph.Register<FakeSceneProducer>(
				FakeSceneProducer{.albedoAlpha = 1.0f, .relPos = glm::vec3(0.0f, 0.0f, -distanceMeters)}
			);
			graph.RegisterRef(transNode);
			graph.RegisterRef(multiNode);
			graph.RegisterRef(skyNode);
			graph.RegisterRef(compositeNode);

			graph::FrameContext ctx{.width = kWidth, .height = kHeight};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			backend.Execute(graph, ctx, cmd, false);

			auto hdrColorTex = registry.GetTexture<HdrColor>();
			REQUIRE(hdrColorTex != nullptr);

			vk::ImageLayout         preCopyLayout = hdrColorTex->GetCurrentLayout();
			vk::ImageMemoryBarrier2 toTransfer{};
			toTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setOldLayout(preCopyLayout)
				.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setImage(hdrColorTex->GetImage())
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo depInfo{};
			depInfo.setImageMemoryBarriers(toTransfer);
			vkCmd.pipelineBarrier2(depInfo);

			vk::BufferImageCopy copyRegion{};
			copyRegion.setBufferOffset(0)
				.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
				.setImageExtent(vk::Extent3D{kWidth, kHeight, 1});
			vkCmd.copyImageToBuffer(hdrColorTex->GetImage(), vk::ImageLayout::eTransferSrcOptimal, readbackBuffer, copyRegion);

			vk::ImageMemoryBarrier2 restoreLayout{};
			restoreLayout.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setNewLayout(preCopyLayout)
				.setImage(hdrColorTex->GetImage())
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo restoreDepInfo{};
			restoreDepInfo.setImageMemoryBarriers(restoreLayout);
			vkCmd.pipelineBarrier2(restoreDepInfo);

			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();

			vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, readbackSize);

			const auto* half = static_cast<const std::uint16_t*>(readbackResultInfo.pMappedData);
			std::size_t centerIndex = (kHeight / 2) * kWidth + (kWidth / 2);
			glm::vec4   centerPixel(
				HalfToFloat(half[centerIndex * 4 + 0]),
				HalfToFloat(half[centerIndex * 4 + 1]),
				HalfToFloat(half[centerIndex * 4 + 2]),
				HalfToFloat(half[centerIndex * 4 + 3])
			);

			vkDevice.freeCommandBuffers(pool, vkCmd);
			return centerPixel;
		};

		glm::vec4 close = renderAndReadBack(100.0f);
		glm::vec4 medium = renderAndReadBack(5000.0f);
		glm::vec4 far = renderAndReadBack(25000.0f);

		MESSAGE("close (100m):   r=", close.r, " g=", close.g, " b=", close.b);
		MESSAGE("medium (5km):   r=", medium.r, " g=", medium.g, " b=", medium.b);
		MESSAGE("far (25km):     r=", far.r, " g=", far.g, " b=", far.b);

		for (const glm::vec4* p : {&close, &medium, &far}) {
			CHECK(std::isfinite(p->r));
			CHECK(std::isfinite(p->g));
			CHECK(std::isfinite(p->b));
			// Scene albedo was cleared to 0.5; even full-strength direct sun (intensity ~10) plus a
			// generous margin for in-scattering should not push a single pixel's radiance into the
			// hundreds. This is deliberately loose -- it's a "not blindingly bright" bound, not a
			// tight physical prediction.
			CHECK(p->r < 50.0f);
			CHECK(p->g < 50.0f);
			CHECK(p->b < 50.0f);
		}

		// Sky-color convergence check: `far` above was rendered with the default
		// skyConvergenceStrength=1.0 (AtmospherePushConstants' real default). Re-rendering the same
		// 25km case with it forced to 0.0 isolates exactly this blend's contribution -- if the
		// SkyViewLUT wiring were dead (e.g. a descriptor never actually bound, or skyViewIndex never
		// set), these two would come out identical. The direction matters too, not just "differs":
		// convergence is supposed to fix the reported-backwards color balance by pulling distant fog
		// toward the sky's own (bluer) hue, so far's blue fraction of total radiance should come out
		// higher than farNoConvergence's.
		AtmospherePushConstants noConvergence{};
		noConvergence.skyConvergenceStrength = 0.0f;
		std::memcpy(bindlessSet.atmosphereUboMapped, &noConvergence, sizeof(AtmospherePushConstants));
		vmaFlushAllocation(device.GetAllocator(), bindlessSet.atmosphereUboAllocation, 0, sizeof(AtmospherePushConstants));
		glm::vec4 farNoConvergence = renderAndReadBack(25000.0f);
		MESSAGE(
			"far, no convergence (25km): r=",
			farNoConvergence.r,
			" g=",
			farNoConvergence.g,
			" b=",
			farNoConvergence.b
		);

		float farBlueFraction = far.b / std::max(1e-4f, far.r + far.g + far.b);
		float noConvergenceBlueFraction =
			farNoConvergence.b / std::max(1e-4f, farNoConvergence.r + farNoConvergence.g + farNoConvergence.b);
		CHECK(farBlueFraction > noConvergenceBlueFraction);

		vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
		pipelineLibrary.Reset();
		compositeNode.Destroy(vkDevice);
		skyNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		DestroyAtmosphereBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
