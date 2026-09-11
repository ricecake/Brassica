#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/Frame.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"
#include "graph/Validation.hpp"
#include "MinimalDevice.hpp"

using namespace brassica::graph;

namespace {

	struct TestColorTarget {};

	struct TestDepthTarget {};

	struct TestBufferTarget {};

	struct PassA {
		using Resources = Declares<Create<TestColorTarget>, Create<TestDepthTarget>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestColorTarget>(),
					.access = AccessKind::Write,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestDepthTarget>(),
					.access = AccessKind::Write,
					.desc = DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct PassB {
		using Resources = Declares<Read<TestColorTarget>, Create<TestBufferTarget>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Compute};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestColorTarget>(),
					.access = AccessKind::Read,
					.desc = ColorAttachmentDesc(1920, 1080, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBufferTarget>(),
					.access = AccessKind::Write,
					.desc = StorageBufferDesc(1024),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct TestTLAS {};

	// Mirrors what the migration's TerrainNode declares: a Graphics-domain node that refreshes
	// an already-built acceleration structure (the real build stays entirely out-of-band,
	// unchanged -- see the AccelerationStructure resource-kind plan). Setup() itself does
	// nothing here; the test registers the (fake) AS into the registry before Setup/Compile run,
	// exactly mirroring how Engine registers TerrainPass::GetTLAS() before constructing the
	// Frame each frame.
	struct PassTLASBuild {
		using Resources = Declares<Create<TestTLAS>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestTLAS>(),
					.access = AccessKind::Write,
					.desc = AccelerationStructureDesc()
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// Mirrors DeferredNode's ray-query read of the TLAS in shaders/deferred.frag.
	struct PassTLASRead {
		using Resources = Declares<Read<TestTLAS>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestTLAS>(),
					.access = AccessKind::Read,
					.desc = AccelerationStructureDesc()
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct TestTemporalMask {};

	using TemporalTestTemporal = TypeList<TestTemporalMask>;

	// Writes this frame's mask in place -- mirrors ShadingRateNode's Modify<ShadingRateMap>.
	struct TemporalWriter {
		using Resources = Declares<Modify<TestTemporalMask>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestTemporalMask>(),
					.access = AccessKind::ReadWrite,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// Reads *last* frame's mask -- mirrors TerrainNode's Read<History<ShadingRateMap>>. No node
	// here produces bare TestTemporalMask; PreviousFrame<TemporalTestTemporal> is what satisfies
	// this declaratively, and PhysicalRegistry::ProvisionTemporalPairs is what satisfies it
	// physically (see PhysicalExecutionBackend.hpp).
	struct TemporalReader {
		using Resources = Declares<Read<History<TestTemporalMask>>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<History<TestTemporalMask>>(),
					.access = AccessKind::Read,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct TestVersionedTarget {};

	// Modify<K, 1>'s single realization carries ReadWrite access on the *produced* version --
	// the same shape DeferredNode's real Modify<Swapchain> realization uses -- since consuming
	// version 0 and producing version 1 is one read-modify-write of one physical resource, not
	// two separate ones.
	struct FirstVersionWriter {
		using Resources = Declares<Modify<TestVersionedTarget, 1>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<VersionedKey<TestVersionedTarget, 1>>(),
					.access = AccessKind::ReadWrite,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct SecondVersionWriter {
		using Resources = Declares<Modify<TestVersionedTarget, 2>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<VersionedKey<TestVersionedTarget, 2>>(),
					.access = AccessKind::ReadWrite,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// -- bindless set 0 fixtures ------------------------------------------------------

	struct TestBindlessSampledA {};

	struct TestBindlessSampledB {};

	struct TestBindlessStorage {};

	struct TestBindlessArray {};

	struct TestBindlessResizable {};

	struct TestBindlessFresh {};

	struct SampledWriterA {
		using Resources = Declares<Create<TestBindlessSampledA>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBindlessSampledA>(),
					.access = AccessKind::Write,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct SampledWriterB {
		using Resources = Declares<Create<TestBindlessSampledB>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBindlessSampledB>(),
					.access = AccessKind::Write,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// eStorage | eSampled, like the real atmosphere LUTs (ComputeStorageImageDesc) -- written by
	// one node via imageStore, sampled by a different one via texture(). Needs a slot in *both*
	// bindless arrays at once (PhysicalTexture::GetSampledBindlessIndex's comment).
	struct StorageWriter {
		using Resources = Declares<Create<TestBindlessStorage>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Compute};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBindlessStorage>(),
					.access = AccessKind::Write,
					.desc = ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// layers > 1: routes to the sampled-2D-*array* binding, not the plain sampled-2D one --
	// mirrors the terrain clipmap's real shape (PhysicalResource.hpp's CreateView fix).
	struct ArrayWriter {
		using Resources = Declares<Create<TestBindlessArray>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe       r{.domain = ExecutionDomain::Graphics};
			ResourceDesc desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm);
			desc.layers = 4;
			r.realizations.push_back(
				ResourceRealization{.key = IdOf<TestBindlessArray>(), .access = AccessKind::Write, .desc = desc}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// Setup()'s desc tracks ctx.width/height directly, so re-running with a different FrameContext
	// extent is what exercises ProvisionTexture's desc-mismatch replace-and-retire path.
	struct ResizableWriter {
		using Resources = Declares<Modify<TestBindlessResizable>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBindlessResizable>(),
					.access = AccessKind::ReadWrite,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct FreshWriter {
		using Resources = Declares<Create<TestBindlessFresh>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBindlessFresh>(),
					.access = AccessKind::Write,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	// A real descriptor set/pool/layout for the 3 image bindings PhysicalResourceRegistry
	// actually writes to for textures (sampled-2D, sampled-2D-array, storage) -- deliberately not
	// the engine's real 6-binding layout (Engine::InitGlobalDescriptors, landing in a later
	// stage), and deliberately no acceleration-structure binding: VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
	// requires VK_KHR_acceleration_structure, which MinimalDevice deliberately doesn't enable
	// (see its header comment) -- the AS-persistence test below uses the full Engine instead,
	// exactly like this file's other AS-dependent case already does.
	struct BindlessTestSet {
		vk::DescriptorSetLayout                    layout{};
		vk::DescriptorPool                         pool{};
		PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	BindlessTestSet CreateBindlessTestSet(vk::Device device) {
		std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
		bindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(32)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		bindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		bindings[2]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eStorageImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 3> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;

		BindlessTestSet result;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 2> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 36},
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

		result.bindings.set = set;
		result.bindings.sampledImage2DBinding = 0;
		result.bindings.sampledImage2DArrayBinding = 1;
		result.bindings.storageImageBinding = 2;
		// accelerationStructureBinding stays 0/unused -- no AS binding in this layout, see above.
		return result;
	}

	void DestroyBindlessTestSet(vk::Device device, BindlessTestSet& s) {
		device.destroyDescriptorPool(s.pool);
		device.destroyDescriptorSetLayout(s.layout);
	}

	// Fixtures for the Subgraph-recursion test below: a two-node cluster (SubWriter -> SubReader,
	// a Graphics->Compute cross-domain edge entirely internal to the cluster) nested inside an
	// outer graph via Subgraph<SubClusterSpec>, plus an outer node reading what the cluster
	// produced -- the cross-boundary edge PhysicalExecutionBackend::RunSchedule's recursion must
	// get right for a real multipass post-processing cluster to work at all.
	struct TestSubColor {};

	struct TestSubBuffer {};

	struct TestOuterBuffer {};

	struct SubWriter {
		using Resources = Declares<Create<TestSubColor>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestSubColor>(),
					.access = AccessKind::Write,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	struct SubReader {
		using Resources = Declares<Read<TestSubColor>, Create<TestSubBuffer>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Compute};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestSubColor>(),
					.access = AccessKind::Read,
					.desc = ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestSubBuffer>(),
					.access = AccessKind::Write,
					.desc = StorageBufferDesc(1024),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

	using SubClusterSpec = FrameSpec<SubWriter, SubReader>;
	using SubCluster = Subgraph<SubClusterSpec>;

	static_assert(std::is_same_v<SubClusterSpec::Unsatisfied, TypeList<>>); // fully self-contained

	struct OuterConsumer {
		using Resources = Declares<Read<TestSubBuffer>, Create<TestOuterBuffer>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Compute};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestSubBuffer>(),
					.access = AccessKind::Read,
					.desc = StorageBufferDesc(1024),
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestOuterBuffer>(),
					.access = AccessKind::Write,
					.desc = StorageBufferDesc(64),
				}
			);
			return r;
		}

		void Execute(NodeContext&) {}
	};

} // namespace

// Real device/allocator via brassica::testing::MinimalDevice -- not the full production Engine.
// None of PassA/PassB/PassTLASBuild/PassTLASRead need mesh shaders, ray query, or real
// acceleration structures, so bootstrapping through Engine::InitVulkan would drag in
// requirements MoltenVK can't satisfy for no reason (see MinimalDevice.hpp's header comment).
// Skips gracefully (rather than failing) in an environment with no Vulkan physical device at
// all, while giving genuine coverage (real vmaCreateImage, real RAII teardown, real
// shared-ownership lifetime) wherever core Vulkan 1.3 is present -- MoltenVK included. Vulkan-free
// stub compilation was dropped for the physical layer specifically because it could never have
// caught the vkGetDeviceProcAddr(nullptr, ...) bug this layer once had -- every real code path
// behind such a stub is, by construction, unexercised.

TEST_CASE("PhysicalResourceRegistry provisions real GPU resources with correct RAII/sharing semantics") {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	{
		PhysicalResourceRegistry registry(device.GetDevice(), device.GetAllocator());

		Graph graph;
		graph.Register<PassA>();
		graph.Register<PassB>();

		FrameContext ctx{.width = 1920, .height = 1080};
		graph.Setup(ctx);
		REQUIRE(graph.Compile().has_value());

		registry.Provision(graph.GetSchedule(), graph.Recipes(), 0, true);

		auto colorTex = registry.GetTexture<TestColorTarget>();
		REQUIRE(colorTex != nullptr);
		CHECK(colorTex->GetDesc().width == 1920);
		CHECK(colorTex->GetDesc().height == 1080);
		CHECK(static_cast<bool>(colorTex->GetImage()));
		CHECK(static_cast<bool>(colorTex->GetView()));
		CHECK_FALSE(colorTex->IsImported());

		auto depthTex = registry.GetTexture<TestDepthTarget>();
		REQUIRE(depthTex != nullptr);
		CHECK(depthTex->GetDesc().formatCode == static_cast<std::uint32_t>(vk::Format::eD32Sfloat));

		auto buf = registry.GetBuffer<TestBufferTarget>();
		REQUIRE(buf != nullptr);
		CHECK(buf->GetDesc().byteSize == 1024);
		CHECK(static_cast<bool>(buf->GetBuffer()));

		// Shared ownership: a consumer holding its own reference keeps the resource alive past
		// Reset() dropping the registry's reference -- the whole point of switching to shared_ptr
		// handles instead of the old raw-pointer-into-the-map accessors.
		auto sharedColorTex = colorTex;
		registry.Reset();
		CHECK(registry.GetTexture<TestColorTarget>() == nullptr);
		CHECK(static_cast<bool>(sharedColorTex->GetImage()));
	}

	CHECK(device.GetValidationErrorCount() == 0);
}

TEST_CASE("PhysicalExecutionBackend runs the full physical pipeline against a real device") {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	{
		PhysicalResourceRegistry registry(device.GetDevice(), device.GetAllocator());
		PhysicalExecutionBackend backend(registry);

		Graph graph;
		graph.Register<PassA>();
		graph.Register<PassB>();

		FrameContext  ctx{.width = 1280, .height = 720};
		CommandBuffer cmd{}; // null vkCmd -- exercises provisioning + barrier/rendering no-op guards

		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));

		CHECK(registry.GetTexture<TestColorTarget>() != nullptr);
		CHECK(registry.GetTexture<TestDepthTarget>() != nullptr);
		CHECK(registry.GetBuffer<TestBufferTarget>() != nullptr);
	}
}

// The case above never records into a real vk::CommandBuffer -- a null vkCmd short-circuits
// every guard in BarrierTranslator::TranslateAndDispatch and DynamicRenderingWrapper::Begin, so
// none of the actual barrier/rendering code has ever executed on real hardware. This is the same
// class of blind spot the header comment above warns about for the old vkGetDeviceProcAddr bug:
// a stub that can never exercise the real path can never catch a defect in it either. This case
// closes that gap -- real command pool, real command buffer, real submission, and real
// validation-layer coverage of every barrier this layer synthesizes.
TEST_CASE(
	"PhysicalExecutionBackend records real barriers and rendering through an actual command buffer with no validation "
	"errors"
) {
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
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		PhysicalExecutionBackend backend(registry);

		Graph graph;
		graph.Register<PassA>();
		graph.Register<PassB>();

		FrameContext ctx{.width = 256, .height = 256};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));
		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// Subgraph is not in production use yet, but real multipass post-processing work is planned
// around it, so its execution path needs the same real-hardware coverage as everything else in
// this file rather than staying verified by structure/Dot tests alone (test_graph.cpp) -- those
// never actually run a Subgraph's inner nodes, so they could never have caught
// RunSchedule/Subgraph::Setup not existing at all before this test did.
//
// SubCluster (SubWriter -> SubReader, a Graphics->Compute edge entirely inside the cluster) is
// registered as one node in the outer graph; OuterConsumer reads TestSubBuffer, which only the
// cluster's own SubReader produces -- a real cross-boundary edge the outer Compile() must wire
// from the Subgraph node (by its declared Produces) to OuterConsumer, and RunSchedule's recursion
// must have the cluster's own barriers/dynamic-rendering actually run before that edge's Acquire
// makes sense. A real command buffer + validation layers is what actually proves the recursion
// emits a legal barrier sequence rather than the compile-only Vulkan-free tests' proof that the
// schedule merely *exists*.
TEST_CASE(
	"PhysicalExecutionBackend recurses into a Subgraph's inner graph with real barriers/dynamic-rendering, and an "
	"outer node can read what the subgraph produced"
) {
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
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		PhysicalExecutionBackend backend(registry);

		SubCluster cluster;
		cluster.InnerGraph().Register<SubWriter>();
		cluster.InnerGraph().Register<SubReader>();

		Graph graph;
		graph.Register<SubCluster>(std::move(cluster));
		graph.Register<OuterConsumer>();

		FrameContext ctx{.width = 256, .height = 256};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));
		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		vkDevice.destroyCommandPool(pool);

		// Provisioned into the same registry the outer graph used -- proof the recursion's own
		// Provision call, not just its barrier dispatch, actually ran for the cluster's nodes.
		CHECK(registry.GetTexture<TestSubColor>() != nullptr);
		CHECK(registry.GetBuffer<TestSubBuffer>() != nullptr);
		CHECK(registry.GetBuffer<TestOuterBuffer>() != nullptr);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// A fake, non-null handle is safe here specifically because nothing in this path ever
// dereferences it: Provision() skips AccelerationStructure-kind realizations outright (they're
// always pre-registered, see PhysicalRegistry.hpp), and per the Vulkan spec an acceleration
// structure is synchronized via a generic, handle-free vk::MemoryBarrier2 (BarrierTranslator.hpp's
// EmitAccelerationStructureBarrier) -- so the emitted barrier never names this handle either. Only
// the graph edge, the derived barrier, and the real command-buffer/validation-layer path are
// under test.
//
// This one case bootstraps through the full brassica::Engine rather than MinimalDevice, unlike
// the three above: it's not the fake handle that needs real AS support, it's the barrier itself
// -- VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR/VK_ACCESS_2_ACCELERATION_STRUCTURE_*
// are invalid on a device that hasn't enabled the accelerationStructure/rayQuery features,
// regardless of whether a real acceleration structure is ever touched. MinimalDevice
// deliberately doesn't enable those (see its header comment), so this genuinely needs Engine's
// full feature set -- which means it still skips on this Mac (MoltenVK doesn't implement
// VK_KHR_acceleration_structure/VK_KHR_ray_query either), same as before MinimalDevice existed.
TEST_CASE(
	"PhysicalExecutionBackend synthesizes a real AS-build -> ray-query-read barrier with no "
	"validation errors"
) {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device device = engine.GetDevice();

	vk::CommandPool pool = device.createCommandPool(
		vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, 0}
	);
	vk::CommandBuffer vkCmd =
		device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}).front();

	PhysicalResourceRegistry registry(device, engine.GetAllocator());
	PhysicalExecutionBackend backend(registry);

	auto fakeAS = vk::AccelerationStructureKHR{
		reinterpret_cast<VkAccelerationStructureKHR>(static_cast<std::uintptr_t>(1))
	};
	registry.RegisterImportedAccelerationStructure<TestTLAS>(fakeAS);

	Graph graph;
	graph.Register<PassTLASBuild>();
	graph.Register<PassTLASRead>();

	FrameContext ctx{.width = 256, .height = 256};

	vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
	CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));
	vkCmd.end();

	vk::Queue      queue = device.getQueue(0, 0);
	vk::SubmitInfo submitInfo{};
	submitInfo.setCommandBuffers(vkCmd);
	queue.submit(submitInfo);
	queue.waitIdle();

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	device.destroyCommandPool(pool);
	engine.Cleanup();
}

// History<K>'s temporal-pair wiring (PhysicalRegistry::ProvisionTemporalPairs,
// PhysicalExecutionBackend's ZeroInitializeUndefinedReads) is the regression this closes:
// before it, a History<K> realization had no producer PhysicalRegistry knew how to satisfy at
// all, and BarrierTranslator::DispatchAcquire threw outright (see its "resolved to neither a
// texture, a buffer, nor an acceleration structure" comment).
TEST_CASE(
	"History<K> resolves to a genuinely different, persistent image than K, swapped every "
	"frame, with no validation errors"
) {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	{
		// eResetCommandBuffer: unlike every other case in this file, runFrame below calls
		// vkCmd.begin() three times on the same buffer (once per simulated frame) -- an implicit
		// reset on begin() is only valid from a pool created with this flag.
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

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		PhysicalExecutionBackend backend(registry);

		auto runFrame = [&](std::uint64_t frameIndex) {
			Graph g;
			g.Register<PreviousFrame<TemporalTestTemporal>>();
			g.Register<TemporalWriter>();
			g.Register<TemporalReader>();

			FrameContext ctx{.width = 64, .height = 64, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();
		};

		runFrame(0);
		auto baseFrame0 = registry.GetTexture<TestTemporalMask>();
		auto historyFrame0 = registry.GetTexture<History<TestTemporalMask>>();
		REQUIRE(baseFrame0 != nullptr);
		REQUIRE(historyFrame0 != nullptr);
		CHECK(baseFrame0 != historyFrame0); // genuinely different images, not an alias of one
		CHECK_FALSE(baseFrame0->IsAliased());
		CHECK_FALSE(historyFrame0->IsAliased());

		runFrame(1);
		auto baseFrame1 = registry.GetTexture<TestTemporalMask>();
		auto historyFrame1 = registry.GetTexture<History<TestTemporalMask>>();
		// Swapped: frame 1's "base" slot is exactly frame 0's "history" slot, and vice versa.
		CHECK(baseFrame1 == historyFrame0);
		CHECK(historyFrame1 == baseFrame0);

		runFrame(2);
		auto baseFrame2 = registry.GetTexture<TestTemporalMask>();
		// Parity repeats with period 2 -- back to frame 0's assignment.
		CHECK(baseFrame2 == baseFrame0);

		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The regression this closes: before AccessOf and BarrierTranslator's coalesced map resolved by
// base id, a node consuming version N-1 and producing version N reported Read (not ReadWrite)
// for the edge into it, and two versions of one key could desynchronize into two separate
// EmitImageBarrier calls racing to mutate the same PhysicalTexture's tracked state.
TEST_CASE(
	"Two versions of one key resolve to one PhysicalTexture and coalesce into a single "
	"barrier"
) {
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
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		PhysicalExecutionBackend backend(registry);

		Graph graph;
		graph.Register<Import<TestVersionedTarget>>();
		graph.Register<FirstVersionWriter>();
		graph.Register<SecondVersionWriter>();

		FrameContext ctx{.width = 64, .height = 64};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		auto base = registry.GetTexture<TestVersionedTarget>();
		auto v1 = registry.GetTexture<VersionedKey<TestVersionedTarget, 1>>();
		auto v2 = registry.GetTexture<VersionedKey<TestVersionedTarget, 2>>();
		REQUIRE(base != nullptr);
		CHECK(v1 == base);
		CHECK(v2 == base);

		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The regression this closes: before PhysicalTexture owned its own bindless index (assigned once
// at creation, descriptor written once), the seed code's inert m_bindlessIndices map kept a
// single flat index per ResourceId, which can't express a texture needing a slot in *two* arrays
// at once (sampled and storage), and had no notion of index recycling at all.
TEST_CASE(
	"Bindless indices are assigned distinctly per resource, split correctly between the "
	"sampled and storage arrays, and stable across a desc-match hit"
) {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();
	{
		BindlessTestSet bindlessSet = CreateBindlessTestSet(vkDevice);

		// eResetCommandBuffer: runFrame below calls vkCmd.begin() twice on the same buffer.
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

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		PhysicalExecutionBackend backend(registry);

		auto runFrame = [&](std::uint64_t frameIndex) {
			Graph g;
			g.Register<SampledWriterA>();
			g.Register<SampledWriterB>();
			g.Register<StorageWriter>();
			g.Register<ArrayWriter>();

			FrameContext ctx{.width = 64, .height = 64, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();
		};

		runFrame(0);

		const std::uint32_t indexA = registry.GetBindlessIndex<TestBindlessSampledA>();
		const std::uint32_t indexB = registry.GetBindlessIndex<TestBindlessSampledB>();
		const std::uint32_t storageIndex = registry.GetStorageBindlessIndex<TestBindlessStorage>();
		const std::uint32_t storageSampledIndex = registry.GetBindlessIndex<TestBindlessStorage>();
		const std::uint32_t arrayIndex = registry.GetBindlessIndex<TestBindlessArray>();

		// 0 is the permanent fallback slot -- a real, distinct resource must never land there.
		CHECK(indexA != 0);
		CHECK(indexB != 0);
		CHECK(indexA != indexB);

		// eStorage | eSampled (ComputeStorageImageDesc's shape, mirroring the real atmosphere
		// LUTs) occupies a slot in *both* arrays -- neither is the fallback, and there's no
		// requirement that the two numbers differ (they're different namespaces).
		CHECK(storageIndex != 0);
		CHECK(storageSampledIndex != 0);

		CHECK(arrayIndex != 0);
		CHECK(arrayIndex != indexA);
		CHECK(arrayIndex != indexB);

		// Re-running with an identical desc must not reassign -- ProvisionTexture's desc-match
		// early return means this texture is untouched, so its index (and the descriptor already
		// written at it) must be exactly what it was.
		runFrame(1);
		CHECK(registry.GetBindlessIndex<TestBindlessSampledA>() == indexA);
		CHECK(registry.GetBindlessIndex<TestBindlessSampledB>() == indexB);
		CHECK(registry.GetStorageBindlessIndex<TestBindlessStorage>() == storageIndex);
		CHECK(registry.GetBindlessIndex<TestBindlessArray>() == arrayIndex);

		vkDevice.destroyCommandPool(pool);
		DestroyBindlessTestSet(vkDevice, bindlessSet);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The regression this closes: before the bindless index moved onto PhysicalTexture itself, there
// was no way to express "this key's index should follow whichever physical object it currently
// resolves to" -- ProvisionTemporalPairs' frame-to-frame swap of m_textures[base]/m_textures[historyId]
// would have needed its own bindless-specific bookkeeping. With the index on the object, the
// swap is the whole mechanism: no code anywhere has to know bindless indices exist.
TEST_CASE(
	"A History<K> pair's two bindless indices swap correctly with which key currently "
	"resolves to which physical texture"
) {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();
	{
		BindlessTestSet bindlessSet = CreateBindlessTestSet(vkDevice);

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

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		PhysicalExecutionBackend backend(registry);

		auto runFrame = [&](std::uint64_t frameIndex) {
			Graph g;
			g.Register<PreviousFrame<TemporalTestTemporal>>();
			g.Register<TemporalWriter>();
			g.Register<TemporalReader>();

			FrameContext ctx{.width = 64, .height = 64, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();
		};

		runFrame(0);
		const std::uint32_t base0 = registry.GetBindlessIndex<TestTemporalMask>();
		const std::uint32_t history0 = registry.GetBindlessIndex<History<TestTemporalMask>>();
		REQUIRE(base0 != 0);
		REQUIRE(history0 != 0);
		CHECK(base0 != history0);

		runFrame(1);
		// Swapped: frame 1's "base" index is exactly frame 0's "history" index, and vice versa --
		// each index is permanently owned by one of the pair's two PhysicalTexture objects, and
		// ProvisionTemporalPairs swapped which key currently resolves to which object.
		CHECK(registry.GetBindlessIndex<TestTemporalMask>() == history0);
		CHECK(registry.GetBindlessIndex<History<TestTemporalMask>>() == base0);

		runFrame(2);
		// Parity repeats with period 2 -- back to frame 0's assignment.
		CHECK(registry.GetBindlessIndex<TestTemporalMask>() == base0);
		CHECK(registry.GetBindlessIndex<History<TestTemporalMask>>() == history0);

		vkDevice.destroyCommandPool(pool);
		DestroyBindlessTestSet(vkDevice, bindlessSet);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The regression this closes: the seed code's m_nextBindlessIndex only ever incremented -- a
// long-running session that resizes its window repeatedly would eventually exhaust the array,
// even though every earlier size's texture is long gone.
TEST_CASE(
	"A replaced texture's bindless index is retired and only becomes reusable after the "
	"retirement delay"
) {
	brassica::testing::MinimalDevice device;

	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();
	{
		BindlessTestSet bindlessSet = CreateBindlessTestSet(vkDevice);

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

		PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		PhysicalExecutionBackend backend(registry);

		auto runFrame = [&](std::uint32_t extent, std::uint64_t frameIndex, bool includeFresh) {
			Graph g;
			g.Register<ResizableWriter>();
			if (includeFresh) {
				g.Register<FreshWriter>();
			}

			FrameContext ctx{.width = extent, .height = extent, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			device.GetQueue().submit(submitInfo);
			device.GetQueue().waitIdle();
		};

		runFrame(64, 0, false);
		const std::uint32_t smallIndex = registry.GetBindlessIndex<TestBindlessResizable>();
		REQUIRE(smallIndex != 0);

		// Desc mismatch (64x64 -> 128x128): the old texture is replaced, retiring smallIndex --
		// not reusable yet (retire delay is 2 frames from here).
		runFrame(128, 1, false);
		const std::uint32_t bigIndex = registry.GetBindlessIndex<TestBindlessResizable>();
		CHECK(bigIndex != smallIndex);

		runFrame(128, 2, false); // desc matches now -- no change, and one frame closer to eligible
		CHECK(registry.GetBindlessIndex<TestBindlessResizable>() == bigIndex);

		// Frame 3 is when smallIndex (retired at frame 1) becomes eligible -- and a fresh
		// resource provisioned in the very same Provision() call picks it straight up.
		runFrame(128, 3, true);
		CHECK(registry.GetBindlessIndex<TestBindlessResizable>() == bigIndex);
		CHECK(registry.GetBindlessIndex<TestBindlessFresh>() == smallIndex);

		vkDevice.destroyCommandPool(pool);
		DestroyBindlessTestSet(vkDevice, bindlessSet);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}

// The regression this closes: RegisterImportedAccelerationStructure always replaces the
// PhysicalAccelerationStructure object outright (a rebuilt TLAS has no desc-match early return
// the way a texture does), so an index assigned once must be carried forward across that
// replacement by the registry itself, not reset to a fresh one every rebuild.
// VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR requires VK_KHR_acceleration_structure, which
// MinimalDevice deliberately doesn't enable (this file's other AS-dependent case has the same
// requirement, for the same reason) -- so this one needs the full Engine, not MinimalDevice.
TEST_CASE(
	"An acceleration structure keeps the same bindless index across a handle rebuild, and "
	"the descriptor is only rewritten when the handle actually changes"
) {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device device = engine.GetDevice();

	{
		// Just the one binding this test actually touches -- Engine::InitGlobalDescriptors's real
		// 6-binding layout lands in a later stage.
		vk::DescriptorSetLayoutBinding asBinding{};
		asBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorBindingFlags                    asBindingFlags = vk::DescriptorBindingFlagBits::ePartiallyBound;
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(asBindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(asBinding);
		layoutInfo.pNext = &bindingFlagsInfo;
		vk::DescriptorSetLayout layout = device.createDescriptorSetLayout(layoutInfo);

		vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eAccelerationStructureKHR, 4};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setMaxSets(1);
		vk::DescriptorPool pool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(pool);
		allocInfo.setSetLayouts(layout);
		vk::DescriptorSet set = device.allocateDescriptorSets(allocInfo).front();

		PhysicalResourceRegistry::BindlessBindings bindings{};
		bindings.set = set;
		bindings.accelerationStructureBinding = 0;

		PhysicalResourceRegistry registry(device, engine.GetAllocator());
		registry.SetGlobalDescriptorSet(bindings);

		// Fake, non-null handles -- safe here for the same reason the AS-barrier test elsewhere
		// in this file uses one: nothing in this path ever dereferences the handle, it's only
		// stored and named in a descriptor write.
		auto handleA = vk::AccelerationStructureKHR{
			reinterpret_cast<VkAccelerationStructureKHR>(static_cast<std::uintptr_t>(1))
		};
		auto handleB = vk::AccelerationStructureKHR{
			reinterpret_cast<VkAccelerationStructureKHR>(static_cast<std::uintptr_t>(2))
		};

		registry.RegisterImportedAccelerationStructure<TestTLAS>(handleA);
		const std::uint32_t index = registry.GetAccelerationStructureBindlessIndex<TestTLAS>();
		REQUIRE(index != 0);

		registry.RegisterImportedAccelerationStructure<TestTLAS>(handleB);
		CHECK(registry.GetAccelerationStructureBindlessIndex<TestTLAS>() == index);
		CHECK(registry.GetAccelerationStructure<TestTLAS>()->Get() == handleB);

		// Same handle again -- must not error or duplicate anything.
		registry.RegisterImportedAccelerationStructure<TestTLAS>(handleB);
		CHECK(registry.GetAccelerationStructureBindlessIndex<TestTLAS>() == index);

		device.destroyDescriptorPool(pool);
		device.destroyDescriptorSetLayout(layout);
	}

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	engine.Cleanup();
}
