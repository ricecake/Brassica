#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceKey.hpp"
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

		void Execute(CommandBuffer&) {}
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

		void Execute(CommandBuffer&) {}
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
				ResourceRealization{.key = IdOf<TestTLAS>(), .access = AccessKind::Write, .desc = AccelerationStructureDesc()}
			);
			return r;
		}

		void Execute(CommandBuffer&) {}
	};

	// Mirrors DeferredNode's ray-query read of the TLAS in shaders/deferred.frag.
	struct PassTLASRead {
		using Resources = Declares<Read<TestTLAS>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{.key = IdOf<TestTLAS>(), .access = AccessKind::Read, .desc = AccelerationStructureDesc()}
			);
			return r;
		}

		void Execute(CommandBuffer&) {}
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

		registry.Provision(graph.GetSchedule(), graph.Recipes(), true);

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
		vk::CommandBuffer vkCmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
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

	auto fakeAS =
		vk::AccelerationStructureKHR{reinterpret_cast<VkAccelerationStructureKHR>(static_cast<std::uintptr_t>(1))};
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
