#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <array>
#include <cstring>

#include "foliage/FoliageManager.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/ResourceCatalog.hpp"
#include "MinimalDevice.hpp"
#include "passes/FoliageNode.hpp"
#include "passes/ResourceKeys.hpp"
#include "ServiceLocator.hpp"

TEST_CASE("FoliageManager state and global properties") {
	brassica::FoliageManager mgr;
	mgr.Initialize();

	CHECK(mgr.IsEnabled() == true);

	mgr.SetEnabled(false);
	CHECK(mgr.IsEnabled() == false);

	brassica::GlobalGrassProperties props = mgr.GetGlobalProperties();
	props.densityMultiplier = 2.5f;
	props.lengthMultiplier = 1.2f;
	props.rockRatio = 0.12f;
	props.seaweedRatio = 0.18f;
	props.bushRatio = 0.25f;
	props.treeRatio = 0.35f;
	props.lodBaseRange = 500.0f;
	props.maxLODs = 10u;
	props.tilesPerRow = 24u;
	mgr.SetGlobalProperties(props);

	brassica::GlobalGrassProperties updatedProps = mgr.GetGlobalProperties();
	CHECK(updatedProps.densityMultiplier == doctest::Approx(2.5f));
	CHECK(updatedProps.lengthMultiplier == doctest::Approx(1.2f));
	CHECK(updatedProps.rockRatio == doctest::Approx(0.12f));
	CHECK(updatedProps.seaweedRatio == doctest::Approx(0.18f));
	CHECK(updatedProps.bushRatio == doctest::Approx(0.25f));
	CHECK(updatedProps.treeRatio == doctest::Approx(0.35f));
	CHECK(updatedProps.lodBaseRange == doctest::Approx(500.0f));
	CHECK(updatedProps.maxLODs == 10u);
	CHECK(updatedProps.tilesPerRow == 24u);

	// All 10 biomes are enabled by default -- a biome's weather classification can legitimately
	// drift into any of them over a long session (shaders/helpers/whittaker.glsl), and one left
	// disabled by default meant foliage could silently vanish wherever that happened.
	for (std::uint32_t i = 0; i < brassica::kFoliageBiomeCount; ++i) {
		brassica::GrassProperties biome = mgr.GetBiomeProperties(i);
		CHECK(biome.enabled == 1u);
	}
	brassica::GrassProperties biome1 = mgr.GetBiomeProperties(1);
	CHECK(biome1.height > 0.0f);
}

TEST_CASE("FoliageNode registration and push constants sync") {
	brassica::ServiceLocator locator;
	brassica::ServiceLocator::SetInstance(&locator);

	auto foliageMgr = std::make_shared<brassica::FoliageManager>();
	foliageMgr->Initialize();
	locator.Provide<brassica::IFoliageManager>(foliageMgr);

	brassica::GlobalGrassProperties props = foliageMgr->GetGlobalProperties();
	props.lodBaseRange = 800.0f;
	props.rockRatio = 0.3f;
	props.seaweedRatio = 0.2f;
	props.bushRatio = 0.4f;
	props.treeRatio = 0.5f;
	props.maxLODs = 12u;
	props.tilesPerRow = 16u;
	foliageMgr->SetGlobalProperties(props);

	brassica::FoliageNode node;
	brassica::render::NodeFrameParams frameParams{.time = 12.34f};
	node.SetFrameParams(frameParams);

	CHECK(node.push.windTime == doctest::Approx(12.34f));
	CHECK(node.push.lodBaseRange == doctest::Approx(800.0f));
	CHECK(node.push.rockRatio == doctest::Approx(0.3f));
	CHECK(node.push.seaweedRatio == doctest::Approx(0.2f));
	CHECK(node.push.bushRatio == doctest::Approx(0.4f));
	CHECK(node.push.treeRatio == doctest::Approx(0.5f));
	CHECK(node.push.gridParams.x == 12u);
	CHECK(node.push.gridParams.y == 16u);
	CHECK(node.push.gridParams.z == 12u * 16u * 16u);
	CHECK(node.push.hizIndex == 0u);

	brassica::ServiceLocator::SetInstance(nullptr);
}

// Regression test for a real startup crash: these five keys are catalog-converted (real Desc()
// methods, ResourceKeys.hpp) but FoliageBakeNode/FoliageNode's Init() never called
// catalog.Get<K>() to register them -- ResourceCatalog::DescFor() only answers for a key once
// something has touched it via Get<K>() (see ResourceCatalog.hpp's own comment), so without this,
// every bare Create<>() of these keys in Setup() carried an empty desc that
// PhysicalResourceRegistry::Provision() silently ignores (its "an empty desc is simply ignored"
// rule) -- the buffer never got inserted into the registry at all, surfacing downstream as
// BarrierTranslator's "resolved to neither a texture, a buffer, nor an acceleration structure" the
// first time anything tried to barrier it. No Vulkan device needed: ResourceCatalog itself is
// Vulkan-free, so this tests the exact mechanism directly rather than through Init() (which can't
// run headlessly, since it also compiles real shaders).
TEST_CASE("Foliage archetype/instance buffer keys are registered with the catalog") {
	using namespace brassica;
	graph::ResourceCatalog catalog;
	graph::FrameContext    ctx{.width = 1280, .height = 720};

	// Documents the failure mode this test guards against: an unregistered catalog key's Desc()
	// is simply never consulted, not an error on its own -- the bug was entirely in Provision()
	// silently treating the resulting empty desc as "nothing to provision."
	CHECK_FALSE(catalog.DescFor(graph::IdOf<FoliageArchetypeVertexBuffer>(), ctx).has_value());

	catalog.Get<FoliageArchetypeVertexBuffer>();
	catalog.Get<FoliageArchetypeIndexBuffer>();
	catalog.Get<FoliageArchetypeMetaBuffer>();
	catalog.Get<FoliageInstanceBuffer>();
	catalog.Get<FoliageIndirectArgsBuffer>();

	auto vertexDesc = catalog.DescFor(graph::IdOf<FoliageArchetypeVertexBuffer>(), ctx);
	REQUIRE(vertexDesc.has_value());
	CHECK(vertexDesc->byteSize == FoliageArchetypeVertexBufferDesc().byteSize);
	CHECK(vertexDesc->persistent);

	auto indexDesc = catalog.DescFor(graph::IdOf<FoliageArchetypeIndexBuffer>(), ctx);
	REQUIRE(indexDesc.has_value());
	CHECK(indexDesc->byteSize == FoliageArchetypeIndexBufferDesc().byteSize);
	CHECK(indexDesc->persistent);

	auto metaDesc = catalog.DescFor(graph::IdOf<FoliageArchetypeMetaBuffer>(), ctx);
	REQUIRE(metaDesc.has_value());
	CHECK(metaDesc->byteSize == FoliageArchetypeMetaBufferDesc().byteSize);
	CHECK(metaDesc->persistent);

	auto instanceDesc = catalog.DescFor(graph::IdOf<FoliageInstanceBuffer>(), ctx);
	REQUIRE(instanceDesc.has_value());
	CHECK(instanceDesc->byteSize > 0);
	CHECK(instanceDesc->persistent);

	auto indirectDesc = catalog.DescFor(graph::IdOf<FoliageIndirectArgsBuffer>(), ctx);
	REQUIRE(indirectDesc.has_value());
	CHECK(indirectDesc->byteSize > 0);
	CHECK(indirectDesc->persistent);
}

// Regression test for a real "nothing draws, no error anywhere" bug: FoliageIndirectArgsBuffer
// is a VkDrawMeshTasksIndirectCommandEXT{groupCountX,Y,Z}, and nothing in foliage.mesh (or
// anywhere else) ever writes groupCountY/Z -- only groupCountX, via atomicAdd. A Vulkan indirect
// dispatch's real total is groupCountX*Y*Z, so leaving Y/Z at 0 (exactly what the old plain
// ClearBufferToZero call left them at) means FoliageArchetypeRenderNode's indirect draw always
// dispatches zero workgroups, no matter how many instances got collected. This predates this
// session's own work -- it was masked for as long as foliage.mesh's since-removed
// emitTreeItem/emitBushItem calls drew trees/bushes live, independent of this buffer. Needs a
// real device: this exercises actual barrier/fillBuffer/updateBuffer command recording, not just
// graph-level scheduling (which was already correct -- see FoliageNode.hpp's own comment).
TEST_CASE("FoliageIndirectResetNode zeroes groupCountX but seeds groupCountY/Z to 1") {
	using namespace brassica;

	testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();
	std::array<std::uint32_t, 4> result{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()}
		);

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		graph::PhysicalExecutionBackend backend(registry);

		// graph::Graph::Register<T>() only default-constructs T -- it does not call lifecycle
		// Init(), which only EngineNodeRegistry's auto-registration path does for real. This
		// node's Init() is where its catalog registration lives (the fix this test exists to
		// verify), so it has to be called explicitly here, same as every other
		// MinimalDevice-based test that exercises a node needing real Init() side effects
		// (see test_particle_system.cpp's ParticleResetNode usage).
		render::NodeServices services{.physicalRegistry = &registry};
		FoliageIndirectResetNode resetNode;
		resetNode.Init(services);

		graph::Graph graph;
		graph.RegisterRef(resetNode);

		graph::FrameContext ctx{.width = 256, .height = 256};

		vk::CommandBuffer resetCmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
				.front();
		resetCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(resetCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));
		resetCmd.end();

		vk::SubmitInfo resetSubmit{};
		resetSubmit.setCommandBuffers(resetCmd);
		device.GetQueue().submit(resetSubmit);
		device.GetQueue().waitIdle();

		auto buf = registry.GetBuffer<FoliageIndirectArgsBuffer>();
		REQUIRE(buf != nullptr);

		VkBuffer           readbackBuffer = VK_NULL_HANDLE;
		VmaAllocation      readbackAllocation = nullptr;
		VmaAllocationInfo  readbackInfo{};
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = sizeof(result);
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
			sizeof(result),
		};
		vk::DependencyInfo readDep{};
		readDep.setBufferMemoryBarriers(toTransferRead);
		readCmd.pipelineBarrier2(readDep);
		readCmd.copyBuffer(buf->GetBuffer(), vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, sizeof(result)});
		readCmd.end();

		vk::SubmitInfo readSubmit{};
		readSubmit.setCommandBuffers(readCmd);
		device.GetQueue().submit(readSubmit);
		device.GetQueue().waitIdle();

		vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, sizeof(result));
		std::memcpy(result.data(), readbackInfo.pMappedData, sizeof(result));

		vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(result[0] == 0u); // groupCountX -- zeroed by the reset's main clear, correct
	CHECK(result[1] == 1u); // groupCountY -- the real bug: this used to stay 0
	CHECK(result[2] == 1u); // groupCountZ -- same
	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
