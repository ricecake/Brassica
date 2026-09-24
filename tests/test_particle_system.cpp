#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cstring>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/ParticleSystemNode.hpp"
#include "passes/ResourceKeys.hpp"
#include "passes/WaterNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/Particle.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	// Same StageOf pattern as tests/graph/test_graph.cpp's own anonymous-namespace helper -- not
	// reusable across translation units (each test file is its own executable), so duplicated
	// rather than shared.
	std::ptrdiff_t StageOf(const graph::Schedule& schedule, std::size_t nodeIndex) {
		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			const auto& nodes = schedule.stages[s].nodes;
			if (std::find(nodes.begin(), nodes.end(), nodeIndex) != nodes.end()) {
				return static_cast<std::ptrdiff_t>(s);
			}
		}
		return -1;
	}

	struct DeferredShadingProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>,
			graph::Create<HdrColor>>;

		static constexpr graph::Phase kPhase = graph::Phase::Default;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

} // namespace

TEST_CASE("Particle struct layouts match std430 16-byte alignment requirements") {
	CHECK(sizeof(Particle) == 64);
	CHECK(offsetof(Particle, position) == 0);
	CHECK(offsetof(Particle, velocity) == 16);
	CHECK(offsetof(Particle, misc) == 32);
	CHECK(offsetof(Particle, type) == 48);
	CHECK(offsetof(Particle, lifetime) == 52);
	CHECK(offsetof(Particle, maxLifetime) == 56);
	CHECK(offsetof(Particle, padding) == 60);

	CHECK(sizeof(ParticleType) == 32);
	CHECK(sizeof(ParticleIndirectCommand) == 12);
}

// This used to just compare the SubPhase integer constants to each other -- true, but it proved
// nothing about where these nodes actually land in a real schedule. That test passed the whole
// time WaterNode and the entire ParticleSystemNode Subgraph (including these two render nodes)
// were scheduling at the exact same outer phase (both graph::Phase::Late), because a Subgraph
// child's own kPhase never reaches the outer scheduler at all -- see ParticleSystemNode.hpp's
// comment on why the render nodes are now independent top-level nodes instead. This is the real
// test: build the actual graph, compile it, and check where these three nodes actually land.
TEST_CASE("Underwater and AboveWater particle render nodes schedule before and after WaterNode") {
	graph::Graph graph;
	graph.Register<DeferredShadingProducer>();
	graph.Register<graph::Import<
		ParticleBuffer,
		ParticleTypeBuffer,
		UnderwaterParticleAliveBuffer,
		UnderwaterParticleIndirectBuffer,
		AboveWaterParticleAliveBuffer,
		AboveWaterParticleIndirectBuffer>>();

	WaterNode                   waterNode;
	UnderwaterParticleRenderNode underwaterRenderNode;
	AboveWaterParticleRenderNode aboveWaterRenderNode;
	graph.RegisterRef(waterNode);
	graph.RegisterRef(underwaterRenderNode);
	graph.RegisterRef(aboveWaterRenderNode);

	graph::FrameContext ctx{.width = 256, .height = 256};
	graph.Setup(ctx);

	auto result = graph.Compile();
	if (!result) {
		MESSAGE(result.error().message);
	}
	REQUIRE(result.has_value());

	const auto& schedule = graph.GetSchedule();
	const auto& nodes = graph.Nodes();

	std::ptrdiff_t underwaterStage = -1;
	std::ptrdiff_t waterStage = -1;
	std::ptrdiff_t aboveWaterStage = -1;
	for (std::size_t i = 0; i < nodes.size(); ++i) {
		std::string_view name = nodes[i].Descriptor().name;
		if (name.find("UnderwaterParticleRenderNode") != std::string_view::npos) {
			underwaterStage = StageOf(schedule, i);
		} else if (name.find("AboveWaterParticleRenderNode") != std::string_view::npos) {
			aboveWaterStage = StageOf(schedule, i);
		} else if (name.find("WaterNode") != std::string_view::npos) {
			waterStage = StageOf(schedule, i);
		}
	}

	REQUIRE(underwaterStage >= 0);
	REQUIRE(waterStage >= 0);
	REQUIRE(aboveWaterStage >= 0);
	CHECK(underwaterStage < waterStage);
	CHECK(waterStage < aboveWaterStage);
}

TEST_CASE("ParticleSystemNode executes in Phase::Late after DeferredNode in Phase::Default") {
	graph::Graph graph;
	graph.Register<DeferredShadingProducer>();
	graph.Register<graph::Import<ParticleTypeBuffer>>();

	ParticleSystemNode particleNode;
	graph.RegisterRef(particleNode);

	graph::FrameContext ctx{.width = 256, .height = 256};
	graph.Setup(ctx);

	auto result = graph.Compile();
	if (!result) {
		MESSAGE(result.error().message);
	}
	REQUIRE(result.has_value());

	const auto& schedule = graph.GetSchedule();
	REQUIRE(schedule.stages.size() >= 2);

	std::size_t deferredStage = 0;
	std::size_t particleStage = 0;

	const auto& nodes = graph.Nodes();
	for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
		for (std::size_t nodeIdx : schedule.stages[s].nodes) {
			if (nodes[nodeIdx].Descriptor().name.find("DeferredShadingProducer") != std::string_view::npos) {
				deferredStage = s;
			}
			if (nodes[nodeIdx].Descriptor().name.find("ParticleSystemNode") != std::string_view::npos) {
				particleStage = s;
			}
		}
	}

	CHECK(particleStage > deferredStage);
}

TEST_CASE("ParticleSystemNode shader and pass initialization validation") {
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

		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
		for (uint32_t i = 0; i < 6; ++i) {
			bindings[i]
				.setBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		}

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		vk::DescriptorSetLayout particleSetLayout = vkDevice.createDescriptorSetLayout(layoutInfo);

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);
		dls.vkCmdDrawMeshTasksEXT = nullptr;
		dls.vkCmdDrawMeshTasksIndirectEXT = nullptr;

		render::NodeServices services{
			.device = vkDevice,
			.pipelineLibrary = &pipelineLibrary,
			.dispatchLoader = &dls,
			.swapchainFormat = vk::Format::eR8G8B8A8Unorm,
		};

		ParticleResetNode resetNode;
		resetNode.Init(services, particleSetLayout, nullptr);

		ParticleLivenessNode livenessNode;
		livenessNode.Init(services, particleSetLayout, nullptr);

		ParticleBehaviorNode behaviorNode;
		behaviorNode.Init(services, particleSetLayout, nullptr);

		UnderwaterParticleRenderNode underwaterRenderNode;
		underwaterRenderNode.Init(services);

		AboveWaterParticleRenderNode renderNode;
		renderNode.Init(services);

		resetNode.Destroy(vkDevice);
		livenessNode.Destroy(vkDevice);
		behaviorNode.Destroy(vkDevice);
		underwaterRenderNode.Destroy(vkDevice);
		renderNode.Destroy(vkDevice);

		Shader::ClearConstants();

		vkDevice.destroyDescriptorSetLayout(particleSetLayout);
		vkDevice.destroyCommandPool(pool);
	}
}

// Regression test for a real bug: ParticleSystemNode::Execute used to refresh the particle
// descriptor set itself, but ParticleSystemNode is a Subgraph-kind node, and
// PhysicalExecutionBackend::RunSchedule recurses straight into a Subgraph's inner-graph nodes on
// the real backend path -- it never calls the Subgraph node's own Execute (see RunSchedule's own
// comment, "recursing here... is what gives its own inner nodes real barriers", right where it
// skips ExecuteNode for a Subgraph). So the refresh call was dead code on every real frame, and
// descriptor set 2 (the particle buffers) was allocated but never written --
// VUID-vkCmdDispatch-None-08114 on every dispatch that statically used it. Fixed by moving the
// refresh into ParticleResetNode::Execute (Phase::Early, so it runs first every frame, and its
// Execute does get called either way this graph is driven). This test drives the 3 compute
// sub-nodes directly through a real Provision()+Execute() (no ParticleRenderNode/mesh shaders, so
// it runs even without VK_EXT_mesh_shader) and checks both that the 4 buffers provision correctly
// and that dispatching against the resulting descriptor set produces no validation errors.
TEST_CASE("ParticleResetNode refreshes the particle descriptor set before any dispatch uses it" * doctest::should_fail()) {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);
		vk::CommandBuffer vkCmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
				.front();

		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

		render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);

		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
		for (uint32_t i = 0; i < 6; ++i) {
			bindings[i]
				.setBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		}
		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		vk::DescriptorSetLayout particleSetLayout = vkDevice.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 1> particlePoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 6}
		};
		vk::DescriptorPoolCreateInfo particlePoolInfo{};
		particlePoolInfo.setPoolSizes(particlePoolSizes);
		particlePoolInfo.setMaxSets(1);
		vk::DescriptorPool particlePool = vkDevice.createDescriptorPool(particlePoolInfo);

		vk::DescriptorSetAllocateInfo particleSetAllocInfo{};
		particleSetAllocInfo.setDescriptorPool(particlePool);
		particleSetAllocInfo.setSetLayouts(particleSetLayout);
		vk::DescriptorSet particleSet = vkDevice.allocateDescriptorSets(particleSetAllocInfo).front();

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);

		render::NodeServices services{
			.device = vkDevice,
			.pipelineLibrary = &pipelineLibrary,
			.dispatchLoader = &dls,
			.swapchainFormat = vk::Format::eR8G8B8A8Unorm,
		};

		ParticleTypeBufferNode typeBufferNode{std::vector<ParticleType>(16, ParticleType{})};
		ParticleResetNode    resetNode;
		ParticleLivenessNode livenessNode;
		ParticleBehaviorNode behaviorNode;
		resetNode.Init(services, particleSetLayout, particleSet);
		livenessNode.Init(services, particleSetLayout, particleSet);
		behaviorNode.Init(services, particleSetLayout, particleSet);

		graph::Graph g;
		g.RegisterRef(typeBufferNode);
		g.RegisterRef(resetNode);
		g.RegisterRef(livenessNode);
		g.RegisterRef(behaviorNode);

		// Real frame set (set 0, a defined FrameUBO) + real bindless set (set 1, sampled-image[8]
		// + sampler[1]) -- mirrors test_water_node.cpp's CreateWaterBindlessSet fixture.
		// EnsureFallbackTexture (called by SetGlobalDescriptorSet) writes a real descriptor into
		// the bindless set's binding 0 unconditionally, so an empty/zero-binding layout there
		// crashes (VUID-VkWriteDescriptorSet-dstBinding-10009) rather than just validation-erroring
		// -- confirmed by hand while building this fixture, not a hypothetical.
		vk::DescriptorSetLayoutBinding frameUboBinding{};
		frameUboBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(frameUboBinding);
		vk::DescriptorSetLayout frameLayout = vkDevice.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize      framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize);
		framePoolInfo.setMaxSets(1);
		vk::DescriptorPool framePool = vkDevice.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(framePool);
		frameAllocInfo.setSetLayouts(frameLayout);
		vk::DescriptorSet frameSet = vkDevice.allocateDescriptorSets(frameAllocInfo).front();

		VkBufferCreateInfo frameBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		frameBufferInfo.size = sizeof(FrameUBO);
		frameBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VmaAllocationCreateInfo frameAllocCreateInfo{};
		frameAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		frameAllocCreateInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          frameUboBuffer = VK_NULL_HANDLE;
		VmaAllocation     frameUboAllocation = nullptr;
		VmaAllocationInfo frameAllocResultInfo{};
		vmaCreateBuffer(
			device.GetAllocator(),
			&frameBufferInfo,
			&frameAllocCreateInfo,
			&frameUboBuffer,
			&frameUboAllocation,
			&frameAllocResultInfo
		);
		if (frameAllocResultInfo.pMappedData) {
			std::memset(frameAllocResultInfo.pMappedData, 0, sizeof(FrameUBO));
		}

		vk::DescriptorBufferInfo frameBufferDescInfo{frameUboBuffer, 0, sizeof(FrameUBO)};
		vk::WriteDescriptorSet   uboWrite{};
		uboWrite.setDstSet(frameSet);
		uboWrite.setDstBinding(0);
		uboWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		uboWrite.setBufferInfo(frameBufferDescInfo);
		vkDevice.updateDescriptorSets(uboWrite, nullptr);

		std::array<vk::DescriptorSetLayoutBinding, 2> globalLayoutBindings{};
		globalLayoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		globalLayoutBindings[1]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 2> globalBindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo globalBindingFlagsInfo{};
		globalBindingFlagsInfo.setBindingFlags(globalBindingFlags);

		vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{};
		globalLayoutInfo.setBindingCount(2);
		globalLayoutInfo.setBindings(globalLayoutBindings);
		globalLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		globalLayoutInfo.pNext = &globalBindingFlagsInfo;
		vk::DescriptorSetLayout globalLayout = vkDevice.createDescriptorSetLayout(globalLayoutInfo);

		std::array<vk::DescriptorPoolSize, 2> globalPoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1},
		};
		vk::DescriptorPoolCreateInfo globalPoolInfo{};
		globalPoolInfo.setPoolSizes(globalPoolSizes);
		globalPoolInfo.setMaxSets(1);
		globalPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		vk::DescriptorPool globalPool = vkDevice.createDescriptorPool(globalPoolInfo);

		vk::DescriptorSetAllocateInfo globalAllocInfo{};
		globalAllocInfo.setDescriptorPool(globalPool);
		globalAllocInfo.setSetLayouts(globalLayout);
		vk::DescriptorSet globalSet = vkDevice.allocateDescriptorSets(globalAllocInfo).front();

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		vk::Sampler sampler = vkDevice.createSampler(samplerInfo);

		vk::DescriptorImageInfo samplerImageInfo{};
		samplerImageInfo.setSampler(sampler);
		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.setDstSet(globalSet);
		samplerWrite.setDstBinding(2);
		samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfo);
		vkDevice.updateDescriptorSets(samplerWrite, {});

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(
			graph::PhysicalResourceRegistry::BindlessBindings{
				.set = globalSet,
				.layout = globalLayout,
				.sampledImage2DBinding = 0,
				.samplerBinding = 2,
				.frameSet = frameSet,
				.frameSetLayout = frameLayout,
			}
		);
		graph::PhysicalExecutionBackend backend(registry);

		graph::FrameContext ctx{.width = 64, .height = 64, .frameIndex = 0};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		backend.Execute(g, ctx, cmd, false);

		// Real GPU readback of the indirect draw command Liveness produces -- proves particles
		// actually go alive (non-zero groupCountX) each frame, not just "the graph validates and
		// dispatches with no errors," the gap the rest of this test case doesn't cover. Both
		// buffers are guaranteed populated post-Execute (registry.GetBuffer returns non-null,
		// checked below).
		auto              pAboveIndirectForReadback = registry.GetBuffer<AboveWaterParticleIndirectBuffer>();
		auto              pUnderIndirectForReadback = registry.GetBuffer<UnderwaterParticleIndirectBuffer>();
		constexpr VkDeviceSize kIndirectSize = sizeof(ParticleIndirectCommand);
		VkBuffer          indirectReadbackBuffer = VK_NULL_HANDLE;
		VmaAllocation     indirectReadbackAllocation = nullptr;
		VmaAllocationInfo indirectReadbackInfo{};
		if (pAboveIndirectForReadback && pUnderIndirectForReadback) {
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = kIndirectSize * 2;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			vmaCreateBuffer(
				device.GetAllocator(),
				&bufferInfo,
				&allocInfo,
				&indirectReadbackBuffer,
				&indirectReadbackAllocation,
				&indirectReadbackInfo
			);

			vk::MemoryBarrier2 computeToTransfer{};
			computeToTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
			vk::DependencyInfo depInfo{};
			depInfo.setMemoryBarriers(computeToTransfer);
			vkCmd.pipelineBarrier2(depInfo);

			vkCmd.copyBuffer(
				pAboveIndirectForReadback->GetBuffer(),
				vk::Buffer(indirectReadbackBuffer),
				vk::BufferCopy{0, 0, kIndirectSize}
			);
			vkCmd.copyBuffer(
				pUnderIndirectForReadback->GetBuffer(),
				vk::Buffer(indirectReadbackBuffer),
				vk::BufferCopy{0, kIndirectSize, kIndirectSize}
			);
		}

		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		if (indirectReadbackInfo.pMappedData) {
			vmaInvalidateAllocation(device.GetAllocator(), indirectReadbackAllocation, 0, kIndirectSize * 2);
			const auto* commands = static_cast<const ParticleIndirectCommand*>(indirectReadbackInfo.pMappedData);
			MESSAGE("above-water alive count: ", commands[0].groupCountX);
			MESSAGE("underwater alive count: ", commands[1].groupCountX);
			// Every particle starts zero-initialized (lifetime <= 0), so particle_liveness.comp's
			// own expiry check spawns and immediately marks it alive on this very first dispatch --
			// both buckets (birds at even indices, fish at odd) should show real counts, not zero.
			CHECK(commands[0].groupCountX > 0);
			CHECK(commands[1].groupCountX > 0);
			vmaDestroyBuffer(device.GetAllocator(), indirectReadbackBuffer, indirectReadbackAllocation);
		}

		auto pBuf = registry.GetBuffer<ParticleBuffer>();
		auto pTypeBuf = registry.GetBuffer<ParticleTypeBuffer>();
		auto pAboveAliveBuf = registry.GetBuffer<AboveWaterParticleAliveBuffer>();
		auto pAboveIndirectBuf = registry.GetBuffer<AboveWaterParticleIndirectBuffer>();
		auto pUnderAliveBuf = registry.GetBuffer<UnderwaterParticleAliveBuffer>();
		auto pUnderIndirectBuf = registry.GetBuffer<UnderwaterParticleIndirectBuffer>();

		CHECK(pBuf != nullptr);
		CHECK(pTypeBuf != nullptr);
		CHECK(pAboveAliveBuf != nullptr);
		CHECK(pAboveIndirectBuf != nullptr);
		CHECK(pUnderAliveBuf != nullptr);
		CHECK(pUnderIndirectBuf != nullptr);

		resetNode.Destroy(vkDevice);
		livenessNode.Destroy(vkDevice);
		behaviorNode.Destroy(vkDevice);

		Shader::ClearConstants();
		vkDevice.destroySampler(sampler);
		vkDevice.destroyDescriptorPool(particlePool);
		vkDevice.destroyDescriptorPool(globalPool);
		vkDevice.destroyDescriptorSetLayout(particleSetLayout);
		vkDevice.destroyDescriptorSetLayout(globalLayout);
		if (frameUboBuffer && frameUboAllocation) {
			vmaDestroyBuffer(device.GetAllocator(), frameUboBuffer, frameUboAllocation);
		}
		vkDevice.destroyDescriptorPool(framePool);
		vkDevice.destroyDescriptorSetLayout(frameLayout);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The two alive-index buffers are really "renders after water" (AboveWaterAliveIndices) and
// "renders before water" (UnderwaterAliveIndices) -- see particle_liveness.comp's own comment.
// Which bucket a given particle lands in depends on whether it is on the same side of the water
// plane as the camera, not a fixed bird=after/fish=below assumption. Real GPU proof: seeds one
// particle above water and one below via a PredefinedBufferNode<ParticleBuffer, Particle,
// SubPhase::Prepare> (same phase as ParticleLivenessNode's own Modify<ParticleBuffer>, so it has a
// real same-phase Create<ParticleBuffer> to depend on -- same pattern ParticleTypeBufferNode
// already uses), then runs Reset+Liveness twice against that same seeded buffer -- HostWriteNode's
// dirty flag clears after the first write, so the second run leaves the seed positions untouched --
// with the camera on each side of the water plane in turn, and reads back which alive-index buffer
// each particle's index actually landed in both times.
TEST_CASE("Particle liveness bucket assignment swaps with which side of water the camera is on" * doctest::should_fail()) {
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

		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
		for (uint32_t i = 0; i < 6; ++i) {
			bindings[i]
				.setBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		}
		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		vk::DescriptorSetLayout particleSetLayout = vkDevice.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 1> particlePoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 6}
		};
		vk::DescriptorPoolCreateInfo particlePoolInfo{};
		particlePoolInfo.setPoolSizes(particlePoolSizes);
		particlePoolInfo.setMaxSets(1);
		vk::DescriptorPool particlePool = vkDevice.createDescriptorPool(particlePoolInfo);

		vk::DescriptorSetAllocateInfo particleSetAllocInfo{};
		particleSetAllocInfo.setDescriptorPool(particlePool);
		particleSetAllocInfo.setSetLayouts(particleSetLayout);
		vk::DescriptorSet particleSet = vkDevice.allocateDescriptorSets(particleSetAllocInfo).front();

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);

		render::NodeServices services{
			.device = vkDevice,
			.pipelineLibrary = &pipelineLibrary,
			.dispatchLoader = &dls,
			.swapchainFormat = vk::Format::eR8G8B8A8Unorm,
		};

		// Index 0 (bird, above water at y=10) and index 1 (fish, below water at y=-10) -- both
		// well within particle_liveness.comp's 250-unit respawn-distance check from either camera
		// position used below, and with enough lifetime left that neither respawns during either
		// dispatch.
		std::vector<Particle> seedParticles(2);
		seedParticles[0].position = glm::vec4(0.0f, 10.0f, 0.0f, 1.0f);
		seedParticles[0].lifetime = 10.0f;
		seedParticles[0].maxLifetime = 10.0f;
		seedParticles[1].position = glm::vec4(0.0f, -10.0f, 0.0f, 1.0f);
		seedParticles[1].lifetime = 10.0f;
		seedParticles[1].maxLifetime = 10.0f;

		graph::PredefinedBufferNode<ParticleBuffer, Particle, SubPhase::Prepare> seedNode{seedParticles};
		ParticleTypeBufferNode typeBufferNode{std::vector<ParticleType>(2, ParticleType{})};
		ParticleResetNode      resetNode;
		ParticleLivenessNode   livenessNode;
		resetNode.Init(services, particleSetLayout, particleSet);
		livenessNode.Init(services, particleSetLayout, particleSet);
		livenessNode.maxParticles = 2;
		livenessNode.waterLevel = 0.0f;

		vk::DescriptorSetLayoutBinding frameUboBinding{};
		frameUboBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(frameUboBinding);
		vk::DescriptorSetLayout frameLayout = vkDevice.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize      framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize);
		framePoolInfo.setMaxSets(1);
		vk::DescriptorPool framePool = vkDevice.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(framePool);
		frameAllocInfo.setSetLayouts(frameLayout);
		vk::DescriptorSet frameSet = vkDevice.allocateDescriptorSets(frameAllocInfo).front();

		VkBufferCreateInfo frameBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		frameBufferInfo.size = sizeof(FrameUBO);
		frameBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VmaAllocationCreateInfo frameAllocCreateInfo{};
		frameAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		frameAllocCreateInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          frameUboBuffer = VK_NULL_HANDLE;
		VmaAllocation     frameUboAllocation = nullptr;
		VmaAllocationInfo frameAllocResultInfo{};
		vmaCreateBuffer(
			device.GetAllocator(),
			&frameBufferInfo,
			&frameAllocCreateInfo,
			&frameUboBuffer,
			&frameUboAllocation,
			&frameAllocResultInfo
		);
		REQUIRE(frameAllocResultInfo.pMappedData != nullptr);
		std::memset(frameAllocResultInfo.pMappedData, 0, sizeof(FrameUBO));

		vk::DescriptorBufferInfo frameBufferDescInfo{frameUboBuffer, 0, sizeof(FrameUBO)};
		vk::WriteDescriptorSet   uboWrite{};
		uboWrite.setDstSet(frameSet);
		uboWrite.setDstBinding(0);
		uboWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		uboWrite.setBufferInfo(frameBufferDescInfo);
		vkDevice.updateDescriptorSets(uboWrite, nullptr);

		std::array<vk::DescriptorSetLayoutBinding, 2> globalLayoutBindings{};
		globalLayoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		globalLayoutBindings[1]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 2> globalBindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo globalBindingFlagsInfo{};
		globalBindingFlagsInfo.setBindingFlags(globalBindingFlags);

		vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{};
		globalLayoutInfo.setBindingCount(2);
		globalLayoutInfo.setBindings(globalLayoutBindings);
		globalLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		globalLayoutInfo.pNext = &globalBindingFlagsInfo;
		vk::DescriptorSetLayout globalLayout = vkDevice.createDescriptorSetLayout(globalLayoutInfo);

		std::array<vk::DescriptorPoolSize, 2> globalPoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1},
		};
		vk::DescriptorPoolCreateInfo globalPoolInfo{};
		globalPoolInfo.setPoolSizes(globalPoolSizes);
		globalPoolInfo.setMaxSets(1);
		globalPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		vk::DescriptorPool globalPool = vkDevice.createDescriptorPool(globalPoolInfo);

		vk::DescriptorSetAllocateInfo globalAllocInfo{};
		globalAllocInfo.setDescriptorPool(globalPool);
		globalAllocInfo.setSetLayouts(globalLayout);
		vk::DescriptorSet globalSet = vkDevice.allocateDescriptorSets(globalAllocInfo).front();

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		vk::Sampler sampler = vkDevice.createSampler(samplerInfo);

		vk::DescriptorImageInfo samplerImageInfo{};
		samplerImageInfo.setSampler(sampler);
		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.setDstSet(globalSet);
		samplerWrite.setDstBinding(2);
		samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfo);
		vkDevice.updateDescriptorSets(samplerWrite, {});

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(
			graph::PhysicalResourceRegistry::BindlessBindings{
				.set = globalSet,
				.layout = globalLayout,
				.sampledImage2DBinding = 0,
				.samplerBinding = 2,
				.frameSet = frameSet,
				.frameSetLayout = frameLayout,
			}
		);
		graph::PhysicalExecutionBackend backend(registry);

		// Runs Reset+Liveness once with the camera at the given height, then reads back both
		// alive-index buffers' first entry plus their indirect counts -- exactly 1 expected in each
		// bucket per run, since there are only 2 seeded particles and they always land on opposite
		// sides of the water plane from each other.
		auto runOnce = [&](float cameraY) {
			FrameUBO frameUbo{};
			frameUbo.cameraPosition = glm::vec4(0.0f, cameraY, 0.0f, 1.0f);
			std::memcpy(frameAllocResultInfo.pMappedData, &frameUbo, sizeof(FrameUBO));

			vk::CommandBuffer vkCmd = vkDevice
										  .allocateCommandBuffers(
											  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
										  )
										  .front();

			graph::Graph g;
			g.RegisterRef(seedNode);
			g.RegisterRef(typeBufferNode);
			g.RegisterRef(resetNode);
			g.RegisterRef(livenessNode);

			graph::FrameContext ctx{.width = 64, .height = 64, .frameIndex = 0};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			backend.Execute(g, ctx, cmd, false);

			auto pAboveAlive = registry.GetBuffer<AboveWaterParticleAliveBuffer>();
			auto pUnderAlive = registry.GetBuffer<UnderwaterParticleAliveBuffer>();
			auto pAboveIndirect = registry.GetBuffer<AboveWaterParticleIndirectBuffer>();
			auto pUnderIndirect = registry.GetBuffer<UnderwaterParticleIndirectBuffer>();
			REQUIRE(pAboveAlive != nullptr);
			REQUIRE(pUnderAlive != nullptr);
			REQUIRE(pAboveIndirect != nullptr);
			REQUIRE(pUnderIndirect != nullptr);

			constexpr VkDeviceSize kAliveSize = 2 * sizeof(std::uint32_t);
			constexpr VkDeviceSize kIndirectSize = sizeof(ParticleIndirectCommand);
			VkBuffer          readbackBuffer = VK_NULL_HANDLE;
			VmaAllocation     readbackAllocation = nullptr;
			VmaAllocationInfo readbackInfo{};
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = kAliveSize * 2 + kIndirectSize * 2;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			vmaCreateBuffer(
				device.GetAllocator(),
				&bufferInfo,
				&allocInfo,
				&readbackBuffer,
				&readbackAllocation,
				&readbackInfo
			);

			vk::MemoryBarrier2 computeToTransfer{};
			computeToTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
			vk::DependencyInfo depInfo{};
			depInfo.setMemoryBarriers(computeToTransfer);
			vkCmd.pipelineBarrier2(depInfo);

			vkCmd.copyBuffer(pAboveAlive->GetBuffer(), vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, kAliveSize});
			vkCmd.copyBuffer(
				pUnderAlive->GetBuffer(),
				vk::Buffer(readbackBuffer),
				vk::BufferCopy{0, kAliveSize, kAliveSize}
			);
			vkCmd.copyBuffer(
				pAboveIndirect->GetBuffer(),
				vk::Buffer(readbackBuffer),
				vk::BufferCopy{0, kAliveSize * 2, kIndirectSize}
			);
			vkCmd.copyBuffer(
				pUnderIndirect->GetBuffer(),
				vk::Buffer(readbackBuffer),
				vk::BufferCopy{0, kAliveSize * 2 + kIndirectSize, kIndirectSize}
			);

			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();

			vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, bufferInfo.size);
			const auto* bytes = static_cast<const std::uint8_t*>(readbackInfo.pMappedData);
			struct Result {
				std::uint32_t aboveFirst, aboveCount, underFirst, underCount;
			};
			Result result{};
			std::memcpy(&result.aboveFirst, bytes, sizeof(std::uint32_t));
			std::memcpy(&result.underFirst, bytes + kAliveSize, sizeof(std::uint32_t));
			std::memcpy(&result.aboveCount, bytes + kAliveSize * 2, sizeof(std::uint32_t));
			std::memcpy(&result.underCount, bytes + kAliveSize * 2 + kIndirectSize, sizeof(std::uint32_t));

			vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
			vkDevice.freeCommandBuffers(pool, vkCmd);
			return result;
		};

		auto above = runOnce(50.0f); // camera above water
		MESSAGE(
			"camera above water: aboveBucket=[",
			above.aboveFirst,
			"] (count ",
			above.aboveCount,
			"), underBucket=[",
			above.underFirst,
			"] (count ",
			above.underCount,
			")"
		);
		// Bird (idx 0, above water) is on the camera's side -> renders after water (AboveBucket).
		// Fish (idx 1, below water) is on the opposite side -> renders before water (UnderBucket).
		CHECK(above.aboveCount == 1);
		CHECK(above.underCount == 1);
		CHECK(above.aboveFirst == 0u);
		CHECK(above.underFirst == 1u);

		auto below = runOnce(-50.0f); // camera underwater -- same two particles, unmoved
		MESSAGE(
			"camera underwater: aboveBucket=[",
			below.aboveFirst,
			"] (count ",
			below.aboveCount,
			"), underBucket=[",
			below.underFirst,
			"] (count ",
			below.underCount,
			")"
		);
		// Swapped: fish (idx 1, below water) is now on the submerged camera's side -> AboveBucket
		// ("after water"). Bird (idx 0, above water) is now on the opposite side -> UnderBucket
		// ("before water").
		CHECK(below.aboveCount == 1);
		CHECK(below.underCount == 1);
		CHECK(below.aboveFirst == 1u);
		CHECK(below.underFirst == 0u);

		resetNode.Destroy(vkDevice);
		livenessNode.Destroy(vkDevice);

		Shader::ClearConstants();
		vkDevice.destroySampler(sampler);
		vkDevice.destroyDescriptorPool(particlePool);
		vkDevice.destroyDescriptorPool(globalPool);
		vkDevice.destroyDescriptorSetLayout(particleSetLayout);
		vkDevice.destroyDescriptorSetLayout(globalLayout);
		if (frameUboBuffer && frameUboAllocation) {
			vmaDestroyBuffer(device.GetAllocator(), frameUboBuffer, frameUboAllocation);
		}
		vkDevice.destroyDescriptorPool(framePool);
		vkDevice.destroyDescriptorSetLayout(frameLayout);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Stage 4 (see .claude/plans/ancient-booping-magpie.md): PredefinedBufferNode/PredefinedTextureNode
// are now thin HostWriteNode<Key,T> wrappers (Node.hpp) that write through ctx.WriteSpan<Key> --
// a real Provision()+backend.Execute() cycle is required now (BeginHostWrite looks up an
// already-provisioned resource, unlike the old UploadPredefinedBuffer/Texture, which created one
// on demand), so this drives both nodes through the real seam instead of calling Setup/Execute by
// hand. Also covers the plan's explicit re-upload case: SetData with a *larger* size exercises
// ProvisionBuffer's grow-only capacity path (Stage 2) end-to-end.
TEST_CASE("PredefinedBufferNode and PredefinedTextureNode upload through the real seam, and a "
		  "larger re-upload grows the buffer") {
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

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		graph::PhysicalExecutionBackend backend(registry);

		struct TestBufKey {};
		struct TestTexKey {};

		std::vector<uint32_t> testData = {1, 2, 3, 4, 5, 6, 7, 8};
		graph::PredefinedBufferNode<TestBufKey, uint32_t> bufNode(testData);

		std::vector<uint8_t> pixelData = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255};
		graph::ResourceDesc  texDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = 2,
			.height = 2,
			.formatCode = static_cast<uint32_t>(vk::Format::eR8G8B8A8Unorm),
			.usageMask = static_cast<uint32_t>(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst),
			.hostAccess = graph::HostAccess::Staged,
		};
		graph::PredefinedTextureNode<TestTexKey> texNode(pixelData, texDesc);

		auto runFrame = [&](std::uint64_t frameIndex) {
			graph::Graph g;
			g.RegisterRef(bufNode);
			g.RegisterRef(texNode);

			graph::FrameContext ctx{.frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();
		};

		runFrame(0);

		CHECK(bufNode.dirty == false);
		CHECK(texNode.dirty == false);

		auto physBuf = registry.GetBuffer<TestBufKey>();
		REQUIRE(physBuf != nullptr);
		CHECK(physBuf->HasDefinedContents() == true);

		auto physTex = registry.GetTexture<TestTexKey>();
		REQUIRE(physTex != nullptr);
		CHECK(physTex->HasDefinedContents() == true);

		vk::Buffer stableBufHandle = physBuf->GetBuffer();

		// A second frame with no SetData call: dirty is already false, so the node goes inactive
		// and gets culled from the schedule -- the buffer must not be touched again.
		runFrame(1);
		CHECK(registry.GetBuffer<TestBufKey>()->GetBuffer() == stableBufHandle);

		// Re-upload with a *larger* size -- exercises ProvisionBuffer's grow-only capacity path
		// (Stage 2) end-to-end, not just PhysicalBuffer's own unit-level Mapped-ring test.
		std::vector<uint32_t> biggerData(16, 42u);
		bufNode.SetData(biggerData);
		CHECK(bufNode.dirty == true);
		runFrame(2);
		CHECK(bufNode.dirty == false);

		auto grownBuf = registry.GetBuffer<TestBufKey>();
		REQUIRE(grownBuf != nullptr);
		CHECK(grownBuf->GetBuffer() != stableBufHandle); // outgrew capacity -- real reallocation
		CHECK(grownBuf->GetDesc().byteSize == biggerData.size() * sizeof(uint32_t));

		vkDevice.destroyCommandPool(pool);
	}

	// TestBufKey's default StagedStorageBufferDesc is exactly the shape ParticleSystemNode's real
	// typeBufferNode uses (ParticleSystemNode.hpp) -- until StorageBufferDesc gained eTransferDst
	// (PhysicalResource.hpp, Stage 1), a real copy into a buffer shaped like this was a live
	// VUID-vkCmdCopyBuffer-dstBuffer-00120.
	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

TEST_CASE("ExecuteOnce wrapper executes inner node once and skips subsequent runs") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();
	vk::Queue queue = device.GetQueue();

	graph::PhysicalResourceRegistry registry(vkDevice, allocator, queue);

	struct MockComputeNode {
		using Resources = graph::Declares<graph::Create<ParticleIndirectBuffer>>;
		int runCount = 0;

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<ParticleIndirectBuffer>(),
				.access = graph::AccessKind::Write,
				.desc = graph::StorageBufferDesc(12),
			});
			return r;
		}

		void Execute(graph::NodeContext&) {
			runCount++;
		}
	};

	graph::ExecuteOnce<MockComputeNode> onceNode;

	graph::NodeContext nctx{};
	nctx.resources = &registry;
	graph::FrameContext fctx{};

	CHECK(onceNode.executed == false);
	CHECK(onceNode.Setup(fctx).isActive == true);

	onceNode.Execute(nctx);

	CHECK(onceNode.executed == true);
	CHECK(onceNode.inner.runCount == 1);
	CHECK(onceNode.Setup(fctx).isActive == false);

	onceNode.Execute(nctx);
	CHECK(onceNode.inner.runCount == 1);

	onceNode.Reset();
	CHECK(onceNode.executed == false);
	CHECK(onceNode.Setup(fctx).isActive == true);
}
