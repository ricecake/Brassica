#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cstring>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/EntityNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	struct TestBallTag {};

	struct FakeSceneProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>>;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
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
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

	struct BallBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	BallBindlessSet CreateBallBindlessSet(vk::Device device, VmaAllocator allocator) {
		BallBindlessSet result;

		vk::DescriptorSetLayoutBinding uboBinding{};
		uboBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(uboBinding);
		result.frameLayout = device.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize        framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize);
		framePoolInfo.setMaxSets(1);
		result.framePool = device.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(result.framePool);
		frameAllocInfo.setSetLayouts(result.frameLayout);
		vk::DescriptorSet frameSet = device.allocateDescriptorSets(frameAllocInfo).front();

		VkBufferCreateInfo frameBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		frameBufferInfo.size = sizeof(FrameUBO);
		frameBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VmaAllocationCreateInfo frameAllocCreateInfo{};
		frameAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		frameAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          frameBuffer = VK_NULL_HANDLE;
		VmaAllocationInfo frameAllocResultInfo{};
		vmaCreateBuffer(
			allocator,
			&frameBufferInfo,
			&frameAllocCreateInfo,
			&frameBuffer,
			&result.frameUboAllocation,
			&frameAllocResultInfo
		);
		result.frameUboBuffer = frameBuffer;
		if (frameAllocResultInfo.pMappedData) {
			std::memset(frameAllocResultInfo.pMappedData, 0, sizeof(FrameUBO));
		}

		vk::DescriptorBufferInfo frameBufferDescInfo{result.frameUboBuffer, 0, sizeof(FrameUBO)};
		vk::WriteDescriptorSet   uboWrite{};
		uboWrite.setDstSet(frameSet);
		uboWrite.setDstBinding(0);
		uboWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		uboWrite.setBufferInfo(frameBufferDescInfo);
		device.updateDescriptorSets(uboWrite, nullptr);

		vk::DescriptorSetLayoutBinding sampledBinding{};
		sampledBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		vk::DescriptorBindingFlags bindingFlag = vk::DescriptorBindingFlagBits::ePartiallyBound |
			vk::DescriptorBindingFlagBits::eUpdateAfterBind;
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlag);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindingCount(1);
		layoutInfo.setBindings(sampledBinding);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eSampledImage, 8};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setMaxSets(1);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		result.pool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(result.pool);
		allocInfo.setSetLayouts(result.layout);
		vk::DescriptorSet set = device.allocateDescriptorSets(allocInfo).front();

		result.bindings.set = set;
		result.bindings.layout = result.layout;
		result.bindings.sampledImage2DBinding = 0;
		result.bindings.frameSet = frameSet;
		result.bindings.frameSetLayout = result.frameLayout;
		return result;
	}

	void DestroyBallBindlessSet(vk::Device device, VmaAllocator allocator, BallBindlessSet& s) {
		device.destroyDescriptorPool(s.pool);
		device.destroyDescriptorSetLayout(s.layout);
		if (s.frameUboBuffer && s.frameUboAllocation) {
			vmaDestroyBuffer(allocator, s.frameUboBuffer, s.frameUboAllocation);
		}
		device.destroyDescriptorPool(s.framePool);
		device.destroyDescriptorSetLayout(s.frameLayout);
	}

} // namespace

TEST_CASE("EntityNode shader compilation and host-mapped indirect buffer execution") {
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

	TaskShader     task;
	MeshShader     mesh;
	FragmentShader frag;

	CHECK(task.CompileTaskFromFile(vk::Device{}, "shaders/ball.task"));
	CHECK(mesh.CompileMeshFromFile(vk::Device{}, "shaders/ball.mesh"));
	CHECK(frag.CompileFragmentFromFile(vk::Device{}, "shaders/ball.frag"));

	CHECK(!task.GetSPIRV().empty());
	CHECK(!mesh.GetSPIRV().empty());
	CHECK(!frag.GetSPIRV().empty());

	Shader::ClearConstants();
}

TEST_CASE("EntityNode executes in frame graph writing host-mapped indirect command buffer") {
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
		BallBindlessSet                 bindlessSet = CreateBallBindlessSet(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		graph::PhysicalExecutionBackend backend(registry);

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);
		dls.vkCmdDrawMeshTasksEXT = nullptr;
		dls.vkCmdDrawMeshTasksIndirectEXT = nullptr;

		EntityNode<TestBallTag> entityNode;
		entityNode.SetIndirectCommand(MeshTasksIndirectCommand{1, 1, 1});
		entityNode.Init(
			render::NodeServices{
				.device = vkDevice,
				.pipelineLibrary = &pipelineLibrary,
				.dispatchLoader = &dls,
			}
		);
		Shader::ClearConstants();

		graph::Graph graph;
		graph.Register<FakeSceneProducer>();
		graph.RegisterRef(entityNode);

		graph::FrameContext ctx{.width = 256, .height = 256};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
		vkCmd.end();

		// Verify EntityIndirectBuffer was provisioned as host-mapped
		auto physBuf = registry.GetBuffer<EntityIndirectBuffer<TestBallTag>>();
		REQUIRE(physBuf != nullptr);
		CHECK(physBuf->IsHostMapped());

		const auto* cmdData = static_cast<const MeshTasksIndirectCommand*>(physBuf->MappedSlice(0));
		REQUIRE(cmdData != nullptr);
		CHECK(cmdData->groupCountX == 1);
		CHECK(cmdData->groupCountY == 1);
		CHECK(cmdData->groupCountZ == 1);

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		pipelineLibrary.Reset();
		entityNode.Destroy(vkDevice);
		DestroyBallBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
