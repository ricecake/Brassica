#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <array>
#include <cstring>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "MinimalDevice.hpp"
#include "passes/model/ModelInstanceManager.hpp"
#include "passes/model/nodes/ModelInstanceScatterNode.hpp"
#include "passes/model/resources.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"

using namespace brassica;

namespace {

	struct GlobalDescriptors {
		vk::DescriptorPool      framePool;
		vk::DescriptorPool      globalPool;
		vk::DescriptorSetLayout frameLayout;
		vk::DescriptorSetLayout globalLayout;
		vk::DescriptorSet       frameSet;
		vk::DescriptorSet       globalSet;
	};

	// Minimal valid (non-null, correctly-flagged) frame/global descriptor sets -- every pipeline
	// layout in this engine always includes both sets regardless of whether a given shader
	// actually references their bindings (model_instance_scatter.comp doesn't touch either; it's
	// pure buffer_reference via push constants), so BindCompute still needs *something* real
	// here or vkCreatePipelineLayout rejects the null set layouts. Same binding shape as
	// test_particle_system.cpp's own bootstrap, just without the FrameUBO/lights-buffer/sampler
	// writes this shader never reads.
	GlobalDescriptors CreateMinimalGlobalDescriptors(vk::Device vkDevice) {
		GlobalDescriptors gd{};

		vk::DescriptorSetLayoutBinding frameBinding{};
		frameBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(frameBinding);
		gd.frameLayout = vkDevice.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize       framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize);
		framePoolInfo.setMaxSets(1);
		gd.framePool = vkDevice.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(gd.framePool);
		frameAllocInfo.setSetLayouts(gd.frameLayout);
		gd.frameSet = vkDevice.allocateDescriptorSets(frameAllocInfo).front();

		// Binding 6 (storage buffer): PhysicalResourceRegistry::AssignAndWriteBindlessBufferIndex
		// auto-registers *every* eStorageBuffer-usage buffer into the bindless array as soon as
		// it's provisioned, regardless of whether anything actually reads it by bindless index
		// (ModelInstanceBuffer/ModelInstanceWriteBuffer are both buffer_reference-only, but still
		// have eStorageBuffer set via StorageBufferDesc) -- a known class of test-setup gap, see
		// project memory on BallBindlessSet's identical shape.
		std::array<vk::DescriptorSetLayoutBinding, 2> globalBindings{};
		globalBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		globalBindings[1]
			.setBinding(6)
			.setDescriptorType(vk::DescriptorType::eStorageBuffer)
			.setDescriptorCount(32)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		std::array<vk::DescriptorBindingFlags, 2> globalBindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo globalBindingFlagsInfo{};
		globalBindingFlagsInfo.setBindingFlags(globalBindingFlags);
		vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{};
		globalLayoutInfo.setBindings(globalBindings);
		globalLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		globalLayoutInfo.pNext = &globalBindingFlagsInfo;
		gd.globalLayout = vkDevice.createDescriptorSetLayout(globalLayoutInfo);

		std::array<vk::DescriptorPoolSize, 2> globalPoolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 32},
		};
		vk::DescriptorPoolCreateInfo globalPoolInfo{};
		globalPoolInfo.setPoolSizes(globalPoolSizes);
		globalPoolInfo.setMaxSets(1);
		globalPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		gd.globalPool = vkDevice.createDescriptorPool(globalPoolInfo);

		vk::DescriptorSetAllocateInfo globalAllocInfo{};
		globalAllocInfo.setDescriptorPool(gd.globalPool);
		globalAllocInfo.setSetLayouts(gd.globalLayout);
		gd.globalSet = vkDevice.allocateDescriptorSets(globalAllocInfo).front();

		return gd;
	}

	std::vector<ModelInstance> ReadBackInstances(
		testing::MinimalDevice&          device,
		graph::PhysicalResourceRegistry& registry,
		vk::CommandPool                  pool
	) {
		vk::Device vkDevice = device.GetDevice();
		auto       buf = registry.GetBuffer<ModelInstanceBuffer>();
		REQUIRE(buf != nullptr);

		vk::DeviceSize size = static_cast<vk::DeviceSize>(kModelInstanceCapacity) * sizeof(ModelInstance);

		VkBuffer           readbackBuffer = VK_NULL_HANDLE;
		VmaAllocation      readbackAllocation = nullptr;
		VmaAllocationInfo  readbackInfo{};
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = size;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		vmaCreateBuffer(device.GetAllocator(), &bufferInfo, &allocInfo, &readbackBuffer, &readbackAllocation, &readbackInfo);

		vk::CommandBuffer readCmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
				.front();
		readCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		vk::BufferMemoryBarrier2 toTransferRead{
			buf->GetLastStage(),
			buf->GetLastAccess(),
			vk::PipelineStageFlagBits2::eTransfer,
			vk::AccessFlagBits2::eTransferRead,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			buf->GetBuffer(),
			0,
			size,
		};
		vk::DependencyInfo readDep{};
		readDep.setBufferMemoryBarriers(toTransferRead);
		readCmd.pipelineBarrier2(readDep);
		readCmd.copyBuffer(buf->GetBuffer(), vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, size});
		readCmd.end();

		vk::SubmitInfo submit{};
		submit.setCommandBuffers(readCmd);
		device.GetQueue().submit(submit);
		device.GetQueue().waitIdle();

		vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, size);
		std::vector<ModelInstance> result(kModelInstanceCapacity);
		std::memcpy(result.data(), readbackInfo.pMappedData, size);

		vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
		return result;
	}

	void RunScatterFrame(
		testing::MinimalDevice&          device,
		graph::PhysicalResourceRegistry& registry,
		graph::PhysicalExecutionBackend& backend,
		vk::CommandPool                  pool,
		ModelInstanceScatterNode&        scatterNode,
		ModelInstanceManager&            mgr
	) {
		graph::Graph graph;
		mgr.RegisterBufferNodes(graph, registry.Catalog());
		graph.RegisterRef(scatterNode);

		graph::FrameContext ctx{.width = 256, .height = 256};

		vk::CommandBuffer cmd =
			device.GetDevice()
				.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
				.front();
		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer wrapped{static_cast<void*>(static_cast<VkCommandBuffer>(cmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, wrapped, true));
		cmd.end();

		vk::SubmitInfo submit{};
		submit.setCommandBuffers(cmd);
		device.GetQueue().submit(submit);
		device.GetQueue().waitIdle();
	}

} // namespace

// Regression test for the exact lesson-12 trap this design relies on avoiding: ModelInstanceBuffer
// is catalog-converted and kLifetime::Persistent, with a producer (ModelInstanceScatterNode)
// that's active only on frames with pending writes. Frame 1 spawns two instances and reads back
// both the written slots and a sample of untouched ones (must be zero/dead, not garbage). Frame 2
// has nothing pending -- the scatter node goes inactive, and the whole table must read back
// byte-identical to frame 1's end state, not zero-cleared again.
TEST_CASE("ModelInstanceScatterNode applies sparse writes and survives an idle frame unchanged") {
	ServiceLocator locator;
	ServiceLocator::SetInstance(&locator);

	ModelInstanceManager mgr;
	locator.Provide<ModelInstanceManager>(
		std::shared_ptr<ModelInstanceManager>(&mgr, [](ModelInstanceManager*) {})
	);

	ModelInstanceHandle h0 = mgr.Spawn(0, glm::vec3(1.0f, 2.0f, 3.0f), 2.0f);
	ModelInstanceHandle h1 = mgr.Spawn(0, glm::vec3(4.0f, 5.0f, 6.0f), 1.5f);
	REQUIRE(h0 != kInvalidModelInstanceHandle);
	REQUIRE(h1 != kInvalidModelInstanceHandle);
	mgr.SetClip(h0, 3u, 1000u, true);

	testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		ServiceLocator::SetInstance(nullptr);
		return;
	}

	{
		vk::CommandPool pool = device.GetDevice().createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);

		graph::PhysicalResourceRegistry registry(device.GetDevice(), device.GetAllocator());
		graph::PhysicalExecutionBackend backend(registry);

		GlobalDescriptors globalDescriptors = CreateMinimalGlobalDescriptors(device.GetDevice());
		registry.SetGlobalDescriptorSet(
			graph::PhysicalResourceRegistry::BindlessBindings{
				.set = globalDescriptors.globalSet,
				.layout = globalDescriptors.globalLayout,
				.frameSet = globalDescriptors.frameSet,
				.frameSetLayout = globalDescriptors.frameLayout,
			}
		);

		render::PipelineLibrary pipelineLibrary(device.GetDevice(), nullptr);
		render::NodeServices     services{
			.device = device.GetDevice(),
			.pipelineLibrary = &pipelineLibrary,
			.physicalRegistry = &registry,
		};
		ModelInstanceScatterNode scatterNode;
		scatterNode.Init(services);

		RunScatterFrame(device, registry, backend, pool, scatterNode, mgr);

		std::vector<ModelInstance> frame1 = ReadBackInstances(device, registry, pool);
		CHECK(frame1[h0].position == glm::vec3(1.0f, 2.0f, 3.0f));
		CHECK(frame1[h0].scale == doctest::Approx(2.0f));
		CHECK(frame1[h0].clipIndex == 3u);
		CHECK(frame1[h0].startTimeMs == 1000u);
		CHECK((frame1[h0].flags & kModelInstanceAliveBit) != 0u);
		CHECK((frame1[h0].flags & kModelInstanceLoopBit) != 0u);

		CHECK(frame1[h1].position == glm::vec3(4.0f, 5.0f, 6.0f));
		CHECK(frame1[h1].scale == doctest::Approx(1.5f));
		CHECK(frame1[h1].clipIndex == kModelClipRestPose);
		CHECK((frame1[h1].flags & kModelInstanceAliveBit) != 0u);

		for (std::uint32_t slot = 0; slot < kModelInstanceCapacity; ++slot) {
			if (slot == h0 || slot == h1) {
				continue;
			}
			CHECK(frame1[slot].flags == 0u);
			CHECK(frame1[slot].position == glm::vec3(0.0f));
		}

		// Nothing queued between frames -- the scatter node should go inactive and leave the
		// table exactly as frame 1 left it.
		RunScatterFrame(device, registry, backend, pool, scatterNode, mgr);
		CHECK(mgr.GetLastFlushCount() == 0u);

		std::vector<ModelInstance> frame2 = ReadBackInstances(device, registry, pool);
		CHECK(std::memcmp(frame1.data(), frame2.data(), frame1.size() * sizeof(ModelInstance)) == 0);

		scatterNode.Destroy(device.GetDevice());
		device.GetDevice().destroyDescriptorPool(globalDescriptors.framePool);
		device.GetDevice().destroyDescriptorPool(globalDescriptors.globalPool);
		device.GetDevice().destroyDescriptorSetLayout(globalDescriptors.frameLayout);
		device.GetDevice().destroyDescriptorSetLayout(globalDescriptors.globalLayout);
		device.GetDevice().destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);

	ServiceLocator::SetInstance(nullptr);
}

// Regression test for a real startup crash: ModelInstanceWriteBuffer's sole producer
// (PredefinedBufferNode inside ModelInstanceManager) starts with dirty=true regardless of
// whether SetData has ever been called with real content -- so its very first Setup()/Execute()
// cycle runs unconditionally, before anything has ever spawned. Without a real desc set at
// construction time (see ModelInstanceManager's own comment), that first Execute() tries to
// write into a key Provision() never inserted (its desc was still default-empty), surfacing as
// BarrierTranslator's "resolved to neither a texture, a buffer, nor an acceleration structure."
// Every other test in this file spawns something before the first RegisterBufferNodes call, so
// none of them exercised this -- confirmed the hard way, on real hardware, not hypothetical.
TEST_CASE("ModelInstanceScatterNode survives a startup frame with nothing ever spawned") {
	ServiceLocator locator;
	ServiceLocator::SetInstance(&locator);

	ModelInstanceManager mgr;
	locator.Provide<ModelInstanceManager>(
		std::shared_ptr<ModelInstanceManager>(&mgr, [](ModelInstanceManager*) {})
	);

	testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		ServiceLocator::SetInstance(nullptr);
		return;
	}

	{
		vk::CommandPool pool = device.GetDevice().createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);

		graph::PhysicalResourceRegistry registry(device.GetDevice(), device.GetAllocator());
		graph::PhysicalExecutionBackend backend(registry);

		GlobalDescriptors globalDescriptors = CreateMinimalGlobalDescriptors(device.GetDevice());
		registry.SetGlobalDescriptorSet(
			graph::PhysicalResourceRegistry::BindlessBindings{
				.set = globalDescriptors.globalSet,
				.layout = globalDescriptors.globalLayout,
				.frameSet = globalDescriptors.frameSet,
				.frameSetLayout = globalDescriptors.frameLayout,
			}
		);

		render::PipelineLibrary pipelineLibrary(device.GetDevice(), nullptr);
		render::NodeServices     services{
			.device = device.GetDevice(),
			.pipelineLibrary = &pipelineLibrary,
			.physicalRegistry = &registry,
		};
		ModelInstanceScatterNode scatterNode;
		scatterNode.Init(services);

		// No Spawn()/SetClip() calls at all -- this is the real "nothing has ever spawned yet"
		// startup state the crash needs.
		RunScatterFrame(device, registry, backend, pool, scatterNode, mgr);
		RunScatterFrame(device, registry, backend, pool, scatterNode, mgr);

		scatterNode.Destroy(device.GetDevice());
		device.GetDevice().destroyDescriptorPool(globalDescriptors.framePool);
		device.GetDevice().destroyDescriptorPool(globalDescriptors.globalPool);
		device.GetDevice().destroyDescriptorSetLayout(globalDescriptors.frameLayout);
		device.GetDevice().destroyDescriptorSetLayout(globalDescriptors.globalLayout);
		device.GetDevice().destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);

	ServiceLocator::SetInstance(nullptr);
}
