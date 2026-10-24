// S1 vertical slice (design plan silly-pondering-tome.md, §Stages): proves the record-time,
// state-driven cross-queue hazard mechanism (QueueHazardTracker.hpp/MultiQueueExecutionBackend.hpp)
// is correct before anything is wired into the real Engine. ForceDomain placement throughout: a
// node's own Recipe::domain picks its lane directly (Graphics/Compute/Transfer -> that lane),
// no HEFT involved -- that's a separate, later layer (S3).
//
// Every logical lane aliases this Mac's one real VkQueue (MinimalDevice has exactly one queue
// family of queueCount 1). That's deliberate, not a limitation worked around: a wait semaphore
// referencing a value already submitted earlier on the same real queue is trivially satisfiable
// (the same queue processes submissions in the order they were submitted), so the full segment/
// wait machinery -- cutting, timeline chaining, barrier sanitization -- runs for real here. What
// this Mac *cannot* prove is actual hardware overlap; that needs the other machine's separate
// queue families. See TotalWaitsInserted()/TotalJoinWaitsInserted() for how these tests confirm
// the mechanism actually fired, rather than merely "nothing went wrong" -- on one real queue, a
// missing wait often wouldn't corrupt data anyway (submission order alone would save it), so the
// validation-error count (syncval, which reasons about declared barriers/semaphores independent
// of what the one real queue happens to serialize) is the primary signal here, not data
// correctness -- data checks are a secondary sanity check that will matter more on real hardware.
//
// Every test resource below is deliberately given a usage mask whose derived barrier scope
// (ResourceState.hpp's DeriveBufferState/DeriveImageState) doesn't vary with ExecutionDomain in
// the one way that matters for a shader-free node body -- see TransferOnlyBufferDesc and
// SampledColorImageDesc's own comments. That isolates the multi-queue *sync* mechanism this file
// tests from the barrier-derivation table's domain-specific stage/access rules, which
// test_resource_state.cpp already covers exhaustively and Vulkan-free.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "doctest/doctest.h"

#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/MultiQueueExecutionBackend.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceKey.hpp"
#include "MinimalDevice.hpp"

using namespace brassica::graph;

namespace {

	// Transfer-only usage: DeriveBufferState's Transfer-domain branch derives eAllTransfer/
	// eTransferWrite|Read for this; every *other* domain, having no recognized usage bit to
	// match (no eStorageBuffer/eUniformBuffer/eIndirectBuffer), falls into the "a usage this
	// table doesn't model yet" defensive fallback -- eAllCommands/eTransferWrite|Read (write) or
	// eAllCommands/eMemoryRead (read). All three are broad enough to cover a plain
	// fillBuffer/copyBuffer regardless of which domain the touching node declares.
	inline ResourceDesc TransferOnlyBufferDesc(std::uint64_t byteSize) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Buffer,
			.usageMask = static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc
			),
			.byteSize = byteSize,
		};
	}

	// A small color-attachment-and-sampled image: a Graphics-domain write genuinely lands in
	// eColorAttachmentOptimal via DynamicRenderingWrapper's real (if empty) render pass -- same
	// trick test_physical_backend.cpp's PassA uses -- and a Read from *any* domain wants
	// eShaderReadOnlyOptimal (DeriveImageState's Sampled branch, which is domain-independent for
	// layout; only the stage varies by domain). That gives a real, physically-forced layout
	// change on a plain Read with no shader needed, which is exactly what case (b) below tests.
	inline ResourceDesc SampledColorImageDesc(std::uint32_t width, std::uint32_t height) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR8G8B8A8Unorm),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
			),
		};
	}

	vk::CommandBuffer VkCmdOf(NodeContext& ctx) { return vk::CommandBuffer(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd)); }

	// Reads `count` uint32s back from `src` via a fresh host-visible staging buffer -- the same
	// copyBuffer/invalidate/read-the-mapped-pointer shape test_physical_backend.cpp's own
	// readback helpers use. Submitted and waited on independently, after the caller's own
	// device.waitIdle(), so it never needs to know about the recorder's lanes/timelines.
	std::vector<std::uint32_t> ReadbackU32(
		brassica::testing::MinimalDevice& device,
		vk::CommandPool                   pool,
		vk::Buffer                        src,
		std::size_t                       count
	) {
		// This copy is a bare, unsynchronized submission with no semaphore wait on any of the
		// recorder's lanes -- waitIdle first so it never races whatever those lanes last
		// submitted, same as every pre-existing readback helper in test_physical_backend.cpp.
		device.GetDevice().waitIdle();

		const vk::DeviceSize byteSize = count * sizeof(std::uint32_t);

		VkBufferCreateInfo readbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		readbackInfo.size = byteSize;
		readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          readbackBuffer{};
		VmaAllocation     readbackAllocation{};
		VmaAllocationInfo readbackAllocInfo{};
		REQUIRE(
			vmaCreateBuffer(
				device.GetAllocator(),
				&readbackInfo,
				&allocCreateInfo,
				&readbackBuffer,
				&readbackAllocation,
				&readbackAllocInfo
			) == VK_SUCCESS
		);

		vk::CommandBuffer vkCmd =
			device.GetDevice()
				.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})[0];
		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		vkCmd.copyBuffer(src, vk::Buffer(readbackBuffer), vk::BufferCopy{0, 0, byteSize});
		vkCmd.end();

		vk::CommandBufferSubmitInfo cmdInfo{};
		cmdInfo.setCommandBuffer(vkCmd);
		vk::SubmitInfo2 submit{};
		submit.setCommandBufferInfos(cmdInfo);
		device.GetQueue().submit2(submit, nullptr);
		device.GetDevice().waitIdle();

		vmaInvalidateAllocation(device.GetAllocator(), readbackAllocation, 0, byteSize);
		std::vector<std::uint32_t> out(count);
		std::memcpy(out.data(), readbackAllocInfo.pMappedData, byteSize);

		vmaDestroyBuffer(device.GetAllocator(), readbackBuffer, readbackAllocation);
		return out;
	}

	// Common per-TEST_CASE rig: a syncval-enabled MinimalDevice, a registry, a QueueRecorder with
	// all three lanes aliasing the device's one real queue (see this file's header comment), and
	// a scratch command pool for readbacks.
	struct Rig {
		brassica::testing::MinimalDevice device{/*enableSyncValidation=*/true};
		PhysicalResourceRegistry         registry{device.GetDevice(), device.GetAllocator()};
		QueueRecorder                    recorder;
		vk::CommandPool                  pool{};

		Rig() {
			if (!device.IsValid()) {
				return;
			}
			std::array<vk::Queue, kNumGpuQueues>     queues{device.GetQueue(), device.GetQueue(), device.GetQueue()};
			std::array<std::uint32_t, kNumGpuQueues> families{
				device.GetQueueFamily(),
				device.GetQueueFamily(),
				device.GetQueueFamily()
			};
			recorder.Init(device.GetDevice(), device.GetPhysicalDevice(), queues, families);
			pool = device.GetDevice().createCommandPool(
				vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, device.GetQueueFamily()}
			);
		}

		~Rig() {
			if (!device.IsValid()) {
				return;
			}
			device.GetDevice().waitIdle();
			recorder.Destroy();
			device.GetDevice().destroyCommandPool(pool);
		}
	};

	// ================================================================================
	// Case (a): G write -> C write ("read-modify-write") -> G read.
	// Two cross-queue waits expected: C-waits-on-G before its write, G-waits-on-C before its
	// read. Final data must be C's pattern, not G's stale one.
	// ================================================================================
	struct CaseABuf {};

	struct CaseA_Writer {
		PhysicalResourceRegistry* registry = nullptr;
		ExecutionDomain           domain = ExecutionDomain::Graphics;
		std::uint32_t             pattern = 0;
		bool                      modify = false; // Create on the first touch, Modify thereafter

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(domain);
			if (modify) {
				r.Modify<CaseABuf>().Desc(TransferOnlyBufferDesc(256));
			} else {
				r.Create<CaseABuf>().Desc(TransferOnlyBufferDesc(256));
			}
		}

		void Execute(NodeContext& ctx) {
			auto buf = registry->GetBuffer<CaseABuf>();
			REQUIRE(buf != nullptr);
			VkCmdOf(ctx).fillBuffer(buf->GetBuffer(), 0, VK_WHOLE_SIZE, pattern);
		}
	};

	struct CaseA_Reader {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<CaseABuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext&) {} // the barrier is what this case tests, not the read itself
	};

	TEST_CASE("Multi-queue: G write, C write, G read -- two cross-queue waits, correct final data") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseA_Writer>(
			CaseA_Writer{.registry = &rig.registry, .domain = ExecutionDomain::Graphics, .pattern = 0x11111111u}
		);
		graph.Register<CaseA_Writer>(
			CaseA_Writer{
				.registry = &rig.registry,
				.domain = ExecutionDomain::Compute,
				.pattern = 0x22222222u,
				.modify = true,
			}
		);
		graph.Register<CaseA_Reader>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		CHECK(rig.recorder.TotalWaitsInserted() == 2);

		auto buf = rig.registry.GetBuffer<CaseABuf>();
		REQUIRE(buf != nullptr);
		auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
		for (std::uint32_t word : data) {
			CHECK(word == 0x22222222u);
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// Case (b): a plain cross-queue Read that changes layout still forces a wait, even though
	// nothing about it is a "write."
	// ================================================================================
	struct CaseBImg {};

	struct CaseB_Writer {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<CaseBImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {} // DynamicRenderingWrapper's real (empty) render pass does the work
	};

	struct CaseB_Reader {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<CaseBImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	TEST_CASE("Multi-queue: a cross-queue read that changes layout still forces a wait") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseB_Writer>();
		graph.Register<CaseB_Reader>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		CHECK(rig.recorder.TotalWaitsInserted() == 1);
		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// Case (c): an inactive middle writer. CaseC_InactiveGraphicsModifier never runs (Active
	// false), so Graph::Compile contracts it out entirely -- no edge connects it to anything.
	// Only the physical-state hazard table (not a compiler edge) is what makes the real G
	// reader wait on the real C creator. One cross-queue wait expected.
	// ================================================================================
	struct CaseCBuf {};

	struct CaseC_ComputeCreator {
		PhysicalResourceRegistry* registry = nullptr;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Create<CaseCBuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext& ctx) {
			auto buf = registry->GetBuffer<CaseCBuf>();
			REQUIRE(buf != nullptr);
			VkCmdOf(ctx).fillBuffer(buf->GetBuffer(), 0, VK_WHOLE_SIZE, 0x33333333u);
		}
	};

	// Never scheduled (Active(false) -> RelationalCompiler.hpp contracts it out of the DAG
	// entirely) -- exists purely to prove the hazard table isn't confused by a node that
	// *declares* a Modify but never actually runs or touches anything physical.
	struct CaseC_InactiveGraphicsModifier {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<CaseCBuf>().Desc(TransferOnlyBufferDesc(256));
			r.Active(false);
		}

		void Execute(NodeContext&) { FAIL("inactive node must never execute"); }
	};

	struct CaseC_GraphicsReader {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<CaseCBuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext&) {}
	};

	TEST_CASE("Multi-queue: an inactive middle writer doesn't confuse the hazard table") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseC_ComputeCreator>(CaseC_ComputeCreator{.registry = &rig.registry});
		graph.Register<CaseC_InactiveGraphicsModifier>();
		graph.Register<CaseC_GraphicsReader>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		CHECK(rig.recorder.TotalWaitsInserted() == 1);

		auto buf = rig.registry.GetBuffer<CaseCBuf>();
		REQUIRE(buf != nullptr);
		auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
		for (std::uint32_t word : data) {
			CHECK(word == 0x33333333u);
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// Case (d): zero-init clear, read on a different queue. No active producer exists for
	// CaseDBuf (CaseD_InactiveCreator never runs -- it exists only so ValidateRuntime sees a
	// producer declared for the key at all), so Graph::ZeroInitReads flags the Compute reader's
	// access and QueueRecorder::Execute clears it on the Graphics lane before anything else
	// runs. One cross-queue wait expected (C waiting on G's clear); final data must be all zero.
	// ================================================================================
	struct CaseDBuf {};

	struct CaseD_InactiveCreator {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<CaseDBuf>().Desc(TransferOnlyBufferDesc(256));
			r.Active(false);
		}

		void Execute(NodeContext&) { FAIL("inactive node must never execute"); }
	};

	struct CaseD_ComputeReader {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<CaseDBuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext&) {}
	};

	TEST_CASE("Multi-queue: a zero-init clear on G is correctly waited on by a C read") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseD_InactiveCreator>();
		graph.Register<CaseD_ComputeReader>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		CHECK(rig.recorder.TotalWaitsInserted() == 1);

		auto buf = rig.registry.GetBuffer<CaseDBuf>();
		REQUIRE(buf != nullptr);
		auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
		for (std::uint32_t word : data) {
			CHECK(word == 0u);
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// Case (e): frame-boundary join. C writes frame N; G writes frame N+1. The *same* node is
	// reused both frames (one real Create<CaseEBuf> target the whole time, matching
	// test_physical_backend.cpp's own InactiveZeroInitProducer/ActiveZeroInitReader two-frame
	// precedent for "Provision reuses the existing physical resource across frames when the
	// desc matches") -- only its declared domain and fill pattern change between frames, via
	// pointers to test-local variables the test mutates between Execute() calls.
	// ================================================================================
	struct CaseEBuf {};

	struct CaseE_Writer {
		PhysicalResourceRegistry* registry = nullptr;
		const ExecutionDomain*    domain = nullptr;
		const std::uint32_t*      pattern = nullptr;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(*domain);
			r.Create<CaseEBuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext& ctx) {
			auto buf = registry->GetBuffer<CaseEBuf>();
			REQUIRE(buf != nullptr);
			VkCmdOf(ctx).fillBuffer(buf->GetBuffer(), 0, VK_WHOLE_SIZE, *pattern);
		}
	};

	TEST_CASE("Multi-queue: frame-boundary join -- C writes frame N, G writes frame N+1") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		ExecutionDomain domain = ExecutionDomain::Compute;
		std::uint32_t   pattern = 0x44444444u;

		Graph graph;
		graph.Register<CaseE_Writer>(CaseE_Writer{.registry = &rig.registry, .domain = &domain, .pattern = &pattern});

		NodeContext nodeCtx{};
		nodeCtx.resources = &rig.registry;

		// Frame N: Compute writes. Nothing to join against yet -- neither other lane has run.
		FrameContext ctxN{.width = 4, .height = 4, .frameIndex = 0};
		rig.recorder.BeginFrame(ctxN.frameIndex);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctxN, rig.registry, nodeCtx));
		CHECK(rig.recorder.TotalJoinWaitsInserted() == 0);

		// Frame N+1: Graphics writes. Its first segment this frame must join against Compute's
		// final value from frame N.
		domain = ExecutionDomain::Graphics;
		pattern = 0x55555555u;
		FrameContext ctxN1{.width = 4, .height = 4, .frameIndex = 1};
		rig.recorder.BeginFrame(ctxN1.frameIndex);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctxN1, rig.registry, nodeCtx));
		CHECK(rig.recorder.TotalJoinWaitsInserted() == 1);

		auto buf = rig.registry.GetBuffer<CaseEBuf>();
		REQUIRE(buf != nullptr);
		auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
		for (std::uint32_t word : data) {
			CHECK(word == 0x55555555u);
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// S4b: QueueLane's FRAME_OVERLAP-bounded command-pool reuse (added when persistent,
	// multi-frame operation became real -- the S1-era design signaled a segment's timeline
	// value directly from an ever-growing segment counter, which worked for a 1-2 frame test
	// but would have grown command buffers unboundedly, and reset nothing frame to frame, in a
	// real long-running engine). Alternates which queue writes every single frame, so every
	// frame after the first is a cross-queue join (TotalJoinWaitsInserted keeps climbing) --
	// this is what actually exercises BeginFrame's wait-for-this-slot's-own-reuse path
	// repeatedly, past FRAME_OVERLAP, not just the two frames every other test here covers.
	// ================================================================================
	TEST_CASE("Multi-queue: FRAME_OVERLAP-bounded command buffer reuse survives many frames") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		ExecutionDomain domain = ExecutionDomain::Compute;
		std::uint32_t   pattern = 0;

		Graph graph;
		graph.Register<CaseE_Writer>(CaseE_Writer{.registry = &rig.registry, .domain = &domain, .pattern = &pattern});

		NodeContext nodeCtx{};
		nodeCtx.resources = &rig.registry;

		constexpr std::uint64_t kFrameCount = 11;
		for (std::uint64_t frameIndex = 0; frameIndex < kFrameCount; ++frameIndex) {
			domain = (frameIndex % 2 == 0) ? ExecutionDomain::Compute : ExecutionDomain::Graphics;
			pattern = 0x10000000u + static_cast<std::uint32_t>(frameIndex);

			FrameContext ctx{.width = 4, .height = 4, .frameIndex = frameIndex};
			rig.recorder.BeginFrame(frameIndex);
			CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

			auto buf = rig.registry.GetBuffer<CaseEBuf>();
			REQUIRE(buf != nullptr);
			auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
			for (std::uint32_t word : data) {
				CHECK(word == pattern);
			}
		}

		// A lower bound, not an exact count: Execute() also unconditionally joins Graphics
		// every frame for zero-init clears, regardless of which queue this frame's writer
		// lands on, so a frame where the writer is Compute actually joins twice (once for
		// Graphics' own unconditional check, once for Compute's) -- every one of the 10
		// non-first frames here contributes at least one, which is the property this test
		// actually cares about (joins keep firing correctly well past FRAME_OVERLAP, not a
		// precise count of an incidental second code path).
		CHECK(rig.recorder.TotalJoinWaitsInserted() >= kFrameCount - 1);
		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	TEST_CASE("Multi-queue: real GPU timestamps actually feed a measured node weight") {
		// Design plan's S5: QueueRecorder's per-batch vkCmdWriteTimestamp2 + readback + EMA
		// (MultiQueueExecutionBackend.hpp). Runs enough frames to guarantee at least one full
		// FRAME_OVERLAP readback cycle, then checks the *presence* of a measured entry for
		// "CaseE_Writer" -- not a specific value, since real GPU timing is hardware-dependent
		// and nondeterministic frame to frame, but presence alone proves the whole pipeline
		// (query pool creation, reset, write, confirmed-complete readback, EMA update, all keyed
		// correctly by name) actually ran end to end on real Vulkan, not just compiled.
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		ExecutionDomain domain = ExecutionDomain::Graphics;
		std::uint32_t   pattern = 0;

		Graph graph;
		graph.Register<CaseE_Writer>(CaseE_Writer{.registry = &rig.registry, .domain = &domain, .pattern = &pattern});

		NodeContext nodeCtx{};
		nodeCtx.resources = &rig.registry;

		constexpr std::uint64_t kFrameCount = brassica::FRAME_OVERLAP + 2;
		for (std::uint64_t frameIndex = 0; frameIndex < kFrameCount; ++frameIndex) {
			pattern = 0x30000000u + static_cast<std::uint32_t>(frameIndex);
			FrameContext ctx{.width = 4, .height = 4, .frameIndex = frameIndex};
			rig.recorder.BeginFrame(frameIndex);
			CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);

		// TypeName<T>() (ResourceKey.hpp) returns a fully-qualified name whose exact
		// qualification (anonymous-namespace wrapping, etc.) is compiler-specific -- search for
		// the substring rather than assuming an exact "CaseE_Writer" key.
		const auto& weights = rig.recorder.Weights().node_execution_weights;
		auto        it = std::find_if(weights.begin(), weights.end(), [](const auto& entry) {
			return entry.first.find("CaseE_Writer") != std::string::npos;
		});
		REQUIRE(it != weights.end());
		CHECK(it->second > 0);
	}

	// ================================================================================
	// Case (f): T copies an image G samples. G writes, then G itself reads (samples) it --
	// same queue, ordinary barrier, no wait. Then T reads it too: same layout G's read already
	// left it in (eShaderReadOnlyOptimal, domain-independent -- see SampledColorImageDesc's own
	// comment), so this is a pure cross-queue read-after-read (RAR) relative to G's *read* --
	// not a hazard -- but T still must wait past G's earlier *write*, which no compiler edge
	// connects it to (G's read and T's read are siblings with no edge between them at all; see
	// Graph.hpp's "no RAR edges" rule). One cross-queue wait expected, and *zero* additional
	// pipeline barriers for T's own read -- the semaphore wait alone covers it, since nothing
	// about T's access needs a layout change or a narrower access scope.
	// ================================================================================
	struct CaseFImg {};

	struct CaseF_Writer {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<CaseFImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	struct CaseF_GraphicsSampler {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<CaseFImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	struct CaseF_TransferCopier {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Transfer);
			r.Read<CaseFImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	TEST_CASE("Multi-queue: T's read only waits past G's write, not G's own prior read (RAR)") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseF_Writer>();
		graph.Register<CaseF_GraphicsSampler>();
		graph.Register<CaseF_TransferCopier>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		CHECK(rig.recorder.TotalWaitsInserted() == 1);
		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// Distinct types, not three instances of CaseA_Writer: CompileRelational's last-writer
	// derivation orders Create<K>() before a later Modify<K>() of the same key via their
	// implicit RAW relationship (the Modify reads what Create wrote), but two *further*
	// Modify<K>()s of the same key, with nothing else telling them apart, have no such relation
	// at all -- confirmed the hard way, this test's first version used CaseA_Writer three times
	// and hit CompileRelational's ambiguous-write-order check.
	struct CaseG_MiddleWriter {
		PhysicalResourceRegistry* registry = nullptr;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Modify<CaseABuf>().Desc(TransferOnlyBufferDesc(256));
		}

		void Execute(NodeContext& ctx) {
			auto buf = registry->GetBuffer<CaseABuf>();
			REQUIRE(buf != nullptr);
			VkCmdOf(ctx).fillBuffer(buf->GetBuffer(), 0, VK_WHOLE_SIZE, 0x22222222u);
		}
	};

	struct CaseG_FinalWriter {
		PhysicalResourceRegistry* registry = nullptr;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<CaseABuf>().Desc(TransferOnlyBufferDesc(256));
			// Needed regardless of queue placement: two ReadWrite (Modify) accesses to the same
			// key, with nothing else distinguishing them, are an ambiguous write order on their
			// own (unlike a plain Write producer, a Modify's "which prior state am I reading"
			// has more than one valid reading without this) -- CompileRelational's own
			// ambiguity check caught it when this test's first version omitted it.
			r.After<CaseG_MiddleWriter>();
		}

		void Execute(NodeContext& ctx) {
			auto buf = registry->GetBuffer<CaseABuf>();
			REQUIRE(buf != nullptr);
			VkCmdOf(ctx).fillBuffer(buf->GetBuffer(), 0, VK_WHOLE_SIZE, 0x33333333u);
		}
	};

	// ================================================================================
	// Case (g), S4: Heft placement through the *real* compiler + backend, not just
	// Scheduler.hpp's own unit tests (test_scheduler.cpp) or ForceDomain (every case above).
	// A Compute-domain node sandwiched between two Graphics writes of the same key -- under
	// default weights, hopping to Compute costs more than staying on Graphics (same arithmetic
	// test_scheduler.cpp's "stays on Graphics" case already proved), so all three should land
	// on the same real queue with zero cross-queue waits. The point here is proving that a
	// node whose *declared* domain (Compute) differs from its *placed* queue (Graphics) still
	// gets correct barriers end to end -- declared domain must keep driving
	// DeriveBufferState's stage/access (a real compute dispatch, if this node did one, is a
	// property of the operation, not of which queue recorded it) while placed queue drives the
	// hazard table, with neither one accidentally substituted for the other (exactly the S4 bug
	// this test's first run would have caught, had it existed before that fix).
	// ================================================================================
	TEST_CASE("Multi-queue: Heft places a sandwiched Compute node onto Graphics, zero cross-queue waits") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseA_Writer>(
			CaseA_Writer{.registry = &rig.registry, .domain = ExecutionDomain::Graphics, .pattern = 0x11111111u}
		);
		graph.Register<CaseG_MiddleWriter>(CaseG_MiddleWriter{.registry = &rig.registry});
		graph.Register<CaseG_FinalWriter>(CaseG_FinalWriter{.registry = &rig.registry});

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx, QueuePlacementMode::Heft));

		CHECK(rig.recorder.TotalWaitsInserted() == 0);

		auto buf = rig.registry.GetBuffer<CaseABuf>();
		REQUIRE(buf != nullptr);
		auto data = ReadbackU32(rig.device, rig.pool, buf->GetBuffer(), 64);
		for (std::uint32_t word : data) {
			CHECK(word == 0x33333333u);
		}

		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

	// ================================================================================
	// Case (h), S5: producer-side pre-transition. G writes (leaving the image in
	// eColorAttachmentOptimal), then *two* cross-domain readers -- Compute and Transfer -- each
	// need a real layout change to eShaderReadOnlyOptimal. Without the optimization, each
	// reader's own acquire-phase barrier would perform that transition independently, and the
	// hazard table treats a layout-changing access as write-shaped -- so whichever reader
	// records second would ALSO need to wait on the first reader's queue, not just the
	// producer's (the exact "S1(b)-style serializing wait" the design plan's S5 exists to
	// remove). With it, BarrierTranslator::DispatchRelease performs the one transition on G's
	// own release, so both readers see an already-correct layout (no layout change at all) and
	// each only waits on the producer -- never on each other.
	// ================================================================================
	struct CaseHImg {};

	struct CaseH_Writer {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<CaseHImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	struct CaseH_ComputeReader {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<CaseHImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	struct CaseH_TransferReader {
		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			r.Domain(ExecutionDomain::Transfer);
			r.Read<CaseHImg>().Desc(SampledColorImageDesc(ctx.width, ctx.height));
		}

		void Execute(NodeContext&) {}
	};

	TEST_CASE("Multi-queue: producer-side pre-transition means two cross-queue readers never wait on each other") {
		Rig rig;
		if (!rig.device.IsValid()) {
			MESSAGE("Vulkan physical device not available in this environment; skipping.");
			return;
		}

		Graph graph;
		graph.Register<CaseH_Writer>();
		graph.Register<CaseH_ComputeReader>();
		graph.Register<CaseH_TransferReader>();

		FrameContext ctx{.width = 4, .height = 4};
		NodeContext  nodeCtx{};
		nodeCtx.resources = &rig.registry;

		rig.recorder.BeginFrame(0);
		CHECK_NOTHROW(rig.recorder.Execute(graph, ctx, rig.registry, nodeCtx));

		// Exactly one wait per reader against the producer (Compute-on-Graphics,
		// Transfer-on-Graphics) -- never a third wait between the two readers themselves, which
		// is precisely what the pre-transition exists to avoid.
		CHECK(rig.recorder.TotalWaitsInserted() == 2);
		CHECK(rig.device.GetValidationErrorCount() == 0);
	}

} // namespace
