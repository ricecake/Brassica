#pragma once
#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "EngineConstants.hpp"
#include "graph/BarrierTranslator.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/QueueHazardTracker.hpp"
#include "graph/VulkanSeam.hpp"

// S1 vertical slice (design plan's §Stages): real multi-queue recording, correctness-first.
// Deliberately bypasses Graph::Compile()'s placement entirely -- a node's own Recipe::domain
// *is* its queue (ForceDomain, in the plan's terms: Compute -> the compute lane, Transfer -> the
// transfer lane, Graphics -> the graphics lane, Host -> no GPU lane at all). HEFT placement and
// ordering (S3) are a layer above this one; this file only has to prove the sync layer
// underneath them is correct, independent of how a node ends up assigned to a queue.
namespace brassica::graph {

	// True iff this access needs to wait past *every* other queue's last touch of the resource,
	// not just its last write -- a write always does (WAR/WAW); so does a read that changes a
	// texture's layout, since the transition itself is a write-shaped access that must not race
	// a concurrent reader. A plain same-layout read only needs to wait past writes (RAR is not a
	// hazard). Buffers/accelerationstructures have no layout, so a Read on either is never a
	// "layout change" -- IsWrite(mb.access) alone decides it for them.
	inline bool IsWriteOrLayoutChangeAccess(
		PhysicalResourceRegistry& registry,
		ResourceId                 resource,
		AccessKind                  access,
		ExecutionDomain             domain
	) {
		if (access == AccessKind::Write || access == AccessKind::ReadWrite) {
			return true;
		}
		auto tex = registry.GetTexture(resource);
		if (!tex) {
			return false; // buffer or AS Read -- no layout to change
		}
		const ResourceState derived = DeriveImageState(
			access,
			domain,
			tex->GetDesc().usageMask ? vk::ImageUsageFlags(tex->GetDesc().usageMask) : vk::ImageUsageFlags{},
			static_cast<vk::Format>(tex->GetDesc().formatCode)
		);
		return tex->GetCurrentLayout() != derived.layout;
	}

	// Owns one timeline semaphore and FRAME_OVERLAP command pools (one per frame-in-flight slot,
	// each with its own growable command-buffer list) for a single logical GPU queue. Every
	// logical queue's real vk::Queue may be the *same* VkQueue (today's only Mac, which has one
	// queue family of queueCount 1) -- that's fine: a wait semaphore referencing a value that
	// was already submitted earlier on the same real queue is trivially satisfiable, never a
	// deadlock, because the same queue processes submissions in the order they were submitted
	// regardless of logical-lane bookkeeping. See QueueRecorder's own comment for why segments
	// are always closed before a later segment's wait can reference them, which is what makes
	// this safe.
	//
	// Two separate counters, deliberately not one: the timeline value CloseAndSubmit signals
	// must be globally monotonic across this queue's *entire* lifetime (a timeline semaphore can
	// never signal a value it has already passed), but QueueHazardTracker's segment index is
	// frame-local by design (it resets every BeginFrame, same as its per-resource state) --
	// conflating them (an earlier revision of this class used segment-index+1 as the signal
	// value directly) works only as long as nothing ever resets, which is exactly what FRAME_OVERLAP-
	// bounded command buffer reuse requires.
	class QueueLane {
	public:
		QueueLane() = default;
		~QueueLane() { Destroy(); }
		QueueLane(const QueueLane&) = delete;
		QueueLane& operator=(const QueueLane&) = delete;

		// physicalDevice: only needed to query timestamp support (design plan's S5) -- the plan's
		// own "T may lack it" warning is real, so this checks queueFamily's actual
		// timestampValidBits rather than assuming every family supports it. A family that
		// doesn't gets m_timestampsSupported=false and no query pool at all: every
		// BeginBatchTimestamp call becomes a no-op (nullopt) rather than recording a query no
		// result from it could ever be trusted.
		void Init(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Queue queue, std::uint32_t queueFamily) {
			m_device = device;
			m_queue = queue;

			for (auto& pool : m_pools) {
				vk::CommandPoolCreateInfo poolInfo{};
				poolInfo.setQueueFamilyIndex(queueFamily);
				poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
				pool = device.createCommandPool(poolInfo);
			}

			vk::SemaphoreTypeCreateInfo typeInfo{};
			typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
			typeInfo.setInitialValue(0);
			vk::SemaphoreCreateInfo semInfo{};
			semInfo.setPNext(&typeInfo);
			m_timeline = device.createSemaphore(semInfo);

			auto familyProps = physicalDevice.getQueueFamilyProperties();
			m_timestampsSupported = queueFamily < familyProps.size() && familyProps[queueFamily].timestampValidBits > 0;
			if (m_timestampsSupported) {
				m_timestampPeriodNs = physicalDevice.getProperties().limits.timestampPeriod;
				vk::QueryPoolCreateInfo queryInfo{};
				queryInfo.setQueryType(vk::QueryType::eTimestamp);
				queryInfo.setQueryCount(static_cast<std::uint32_t>(FRAME_OVERLAP * kMaxBatchesPerFrame * 2));
				m_queryPool = device.createQueryPool(queryInfo);
			}
		}

		void Destroy() {
			if (m_queryPool) {
				m_device.destroyQueryPool(m_queryPool);
				m_queryPool = nullptr;
			}
			if (m_timeline) {
				m_device.destroySemaphore(m_timeline);
				m_timeline = nullptr;
			}
			for (auto& pool : m_pools) {
				if (pool) {
					m_device.destroyCommandPool(pool);
					pool = nullptr;
				}
			}
			for (auto& buffers : m_buffers) {
				buffers.clear();
			}
		}

		[[nodiscard]] vk::Semaphore Timeline() const { return m_timeline; }

		[[nodiscard]] std::uint64_t LastSubmittedValue() const { return m_lastSubmittedValue; }

		[[nodiscard]] bool HasOpenSegment() const { return m_openBufferIndex.has_value(); }

		[[nodiscard]] std::uint32_t OpenSegmentIndex() const { return m_openFrameLocalIndex; }

		// The frame-local index of the segment CloseAndSubmit most recently closed (whether in
		// *this* call or an earlier one this frame) -- what CutAndWait/EnsureJoined need to pass
		// to QueueHazardTracker::MarkCaughtUp when a wait targets a segment someone else already
		// closed, since that's frame-local bookkeeping CloseAndSubmit's own (globally monotonic)
		// return value can no longer double as.
		[[nodiscard]] std::uint32_t LastClosedFrameLocalIndex() const { return m_lastClosedFrameLocalIndex; }

		// Rotates to this frame's slot (frameIndex % FRAME_OVERLAP) and, once the queue has run
		// at least FRAME_OVERLAP frames, waits for that slot's own last use (FRAME_OVERLAP
		// frames ago) to finish on the GPU before anything in it is reset/reused -- the same
		// CPU-ahead-of-GPU discipline Engine::DrawFrame's own frameTimelineSemaphore wait already
		// applies to its one command buffer per slot, just now per queue instead of per frame.
		// Resets the frame-local segment counter QueueHazardTracker reads (it resets its own
		// state the same way, same frame boundary) -- never the timeline value itself, which
		// stays globally monotonic regardless of frame boundaries.
		void BeginFrame(std::uint64_t frameIndex) {
			m_activeSlot = static_cast<std::size_t>(frameIndex % FRAME_OVERLAP);
			if (frameIndex >= FRAME_OVERLAP && m_slotFinalValue[m_activeSlot] > 0) {
				vk::SemaphoreWaitInfo waitInfo{};
				waitInfo.setSemaphores(m_timeline);
				waitInfo.setValues(m_slotFinalValue[m_activeSlot]);
				(void)m_device.waitSemaphores(waitInfo, UINT64_MAX);
			}
			m_nextBufferIndexInSlot = 0;
			m_nextFrameLocalIndex = 0;
			m_openBufferIndex.reset();
			m_pendingWaits.clear();
			m_queryPoolResetThisFrame = false;
		}

		[[nodiscard]] std::size_t ActiveSlot() const { return m_activeSlot; }

		[[nodiscard]] bool SupportsTimestamps() const { return m_timestampsSupported; }

		// Writes a start timestamp for one batch (design plan's S5: "per-batch
		// vkCmdWriteTimestamp2") into this frame's active slot's query range, resetting that
		// range first if this is the first batch recorded into it this frame -- a query must be
		// reset before every write, including its very first one after pool creation. Returns
		// the batch index to pass to EndBatchTimestamp, or nullopt if timestamps aren't
		// supported on this lane's queue family or this frame's fixed batch budget
		// (kMaxBatchesPerFrame) is exhausted -- either way the caller just skips timestamping
		// that batch; this is diagnostic/measurement infrastructure, never something a missing
		// capability or an unusually batch-heavy frame should make fatal.
		std::optional<std::uint32_t> BeginBatchTimestamp(vk::CommandBuffer cmd) {
			if (!m_timestampsSupported || m_batchCountInSlot[m_activeSlot] >= kMaxBatchesPerFrame) {
				return std::nullopt;
			}
			if (!m_queryPoolResetThisFrame) {
				const std::uint32_t slotBase = static_cast<std::uint32_t>(m_activeSlot) * kMaxBatchesPerFrame * 2;
				cmd.resetQueryPool(m_queryPool, slotBase, kMaxBatchesPerFrame * 2);
				m_queryPoolResetThisFrame = true;
			}
			const std::uint32_t batchIndex = m_batchCountInSlot[m_activeSlot]++;
			const std::uint32_t queryIndex =
				static_cast<std::uint32_t>(m_activeSlot) * kMaxBatchesPerFrame * 2 + batchIndex * 2;
			cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queryPool, queryIndex);
			return batchIndex;
		}

		void EndBatchTimestamp(vk::CommandBuffer cmd, std::uint32_t batchIndex) {
			const std::uint32_t queryIndex =
				static_cast<std::uint32_t>(m_activeSlot) * kMaxBatchesPerFrame * 2 + batchIndex * 2 + 1;
			cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queryPool, queryIndex);
		}

		// Called once per frame, right after BeginFrame has rotated into this slot (and, once
		// the queue has run at least FRAME_OVERLAP frames, waited for that slot's own prior use
		// to finish on the GPU) -- which is exactly the guarantee that makes it safe to read this
		// slot's query results now: whatever batches were recorded into it last time are
		// confirmed complete. Returns (batchIndex, durationMicroseconds) pairs for however many
		// batches were actually timestamped that time, and resets this slot's own batch count to
		// 0 so BeginBatchTimestamp starts fresh for the frame about to record -- empty if
		// timestamps aren't supported, or this is one of the first FRAME_OVERLAP frames (nothing
		// in this slot yet to read back).
		std::vector<std::pair<std::uint32_t, double>> CollectBatchDurationsMicroseconds(std::uint64_t frameIndex) {
			std::vector<std::pair<std::uint32_t, double>> results;
			const std::uint32_t                            oldCount = m_batchCountInSlot[m_activeSlot];
			m_batchCountInSlot[m_activeSlot] = 0;
			if (!m_timestampsSupported || frameIndex < FRAME_OVERLAP || oldCount == 0) {
				return results;
			}

			const std::uint32_t      base = static_cast<std::uint32_t>(m_activeSlot) * kMaxBatchesPerFrame * 2;
			std::vector<std::uint64_t> raw(static_cast<std::size_t>(oldCount) * 2);
			const vk::Result getResult = m_device.getQueryPoolResults(
				m_queryPool,
				base,
				oldCount * 2,
				raw.size() * sizeof(std::uint64_t),
				raw.data(),
				sizeof(std::uint64_t),
				vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait
			);
			if (getResult != vk::Result::eSuccess) {
				return results; // defensive -- a readback failure just means no measurement this round
			}

			results.reserve(oldCount);
			for (std::uint32_t i = 0; i < oldCount; ++i) {
				const std::uint64_t ticks = raw[static_cast<std::size_t>(i) * 2 + 1] - raw[static_cast<std::size_t>(i) * 2];
				const double micros = (static_cast<double>(ticks) * static_cast<double>(m_timestampPeriodNs)) / 1000.0;
				results.emplace_back(i, micros);
			}
			return results;
		}

		// Opens a fresh segment if none is currently open (begins recording); returns the open
		// segment's command buffer either way, so this is always safe to call before recording
		// anything regardless of whether a segment happens to be open already.
		vk::CommandBuffer OpenIfNeeded() {
			auto& slotBuffers = m_buffers[m_activeSlot];
			if (m_openBufferIndex.has_value()) {
				return slotBuffers[*m_openBufferIndex];
			}
			const std::uint32_t index = m_nextBufferIndexInSlot++;
			if (index >= slotBuffers.size()) {
				vk::CommandBufferAllocateInfo allocInfo{};
				allocInfo.setCommandPool(m_pools[m_activeSlot]);
				allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
				allocInfo.setCommandBufferCount(1);
				slotBuffers.push_back(m_device.allocateCommandBuffers(allocInfo).front());
			}
			vk::CommandBuffer cmd = slotBuffers[index];
			cmd.reset();
			cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			m_openBufferIndex = index;
			m_openFrameLocalIndex = m_nextFrameLocalIndex++;
			m_pendingWaits.clear();
			return cmd;
		}

		// Only valid immediately after OpenIfNeeded opened a *fresh* segment (nothing recorded
		// into it yet) -- a wait only blocks the start of a submission, so adding one after
		// commands already exist in this segment would silently fail to cover them.
		void AddWait(vk::Semaphore semaphore, std::uint64_t value, vk::PipelineStageFlags2 stage) {
			vk::SemaphoreSubmitInfo info{};
			info.setSemaphore(semaphore);
			info.setValue(value);
			info.setStageMask(stage);
			m_pendingWaits.push_back(info);
		}

		// Ends and submits the open segment, if any -- no-op (returns nullopt) otherwise.
		// Returns the frame-local index of the segment just closed, same value
		// LastClosedFrameLocalIndex() reads back afterward.
		//
		// extraSignal: an additional (typically binary, e.g. Engine's swapchain present-wait)
		// semaphore to signal in the same submit2 alongside this lane's own timeline value --
		// used only for the Graphics lane's last segment of a frame that touches the swapchain
		// (design plan's S4 Submission section: "Render-finished is signalled on the last G
		// segment"). nullopt for every other call, including every call this file's own test
		// suite makes, which keeps the single-signal submit identical to before this param
		// existed.
		std::optional<std::uint32_t> CloseAndSubmit(
			std::optional<vk::Semaphore> extraSignal = std::nullopt,
			vk::PipelineStageFlags2      extraSignalStage = vk::PipelineStageFlagBits2::eAllCommands
		) {
			if (!m_openBufferIndex.has_value()) {
				return std::nullopt;
			}
			const std::uint32_t frameLocalIndex = m_openFrameLocalIndex;
			vk::CommandBuffer   cmd = m_buffers[m_activeSlot][*m_openBufferIndex];
			cmd.end();

			vk::CommandBufferSubmitInfo cmdInfo{};
			cmdInfo.setCommandBuffer(cmd);

			const std::uint64_t signalValue = ++m_nextTimelineValue; // globally monotonic, never reset
			vk::SemaphoreSubmitInfo signalInfo{};
			signalInfo.setSemaphore(m_timeline);
			signalInfo.setValue(signalValue);
			signalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

			std::vector<vk::SemaphoreSubmitInfo> signalInfos{signalInfo};
			if (extraSignal.has_value()) {
				vk::SemaphoreSubmitInfo extra{};
				extra.setSemaphore(*extraSignal);
				extra.setStageMask(extraSignalStage);
				signalInfos.push_back(extra);
			}

			vk::SubmitInfo2 submit{};
			submit.setCommandBufferInfos(cmdInfo);
			submit.setWaitSemaphoreInfos(m_pendingWaits);
			submit.setSignalSemaphoreInfos(signalInfos);
			m_queue.submit2(submit, nullptr);

			m_lastSubmittedValue = signalValue;
			m_slotFinalValue[m_activeSlot] = signalValue; // this slot's own BeginFrame reuse wait, FRAME_OVERLAP frames from now
			m_lastClosedFrameLocalIndex = frameLocalIndex;
			m_openBufferIndex.reset();
			m_pendingWaits.clear();
			return frameLocalIndex;
		}

		// Ends (but never submits) the open segment, if any -- the exception-path counterpart to
		// CloseAndSubmit, design plan's S4 catch path: "never submit partially recorded buffers."
		// Consumes no timeline value (m_nextTimelineValue is untouched), so this segment's
		// would-be signal never happens and nothing downstream can be waiting on it (nothing can
		// wait on a value that was never promised). The command buffer itself is left in the
		// ended state; the next time this slot index is reused, OpenIfNeeded's own cmd.reset()
		// handles an abandoned buffer identically to a submitted one.
		void Abandon() {
			if (!m_openBufferIndex.has_value()) {
				return;
			}
			m_buffers[m_activeSlot][*m_openBufferIndex].end();
			m_openBufferIndex.reset();
			m_pendingWaits.clear();
		}

	private:
		vk::Device     m_device{};
		vk::Queue      m_queue{};
		vk::Semaphore  m_timeline{};
		std::uint64_t  m_nextTimelineValue = 0;
		std::uint64_t  m_lastSubmittedValue = 0;

		std::array<vk::CommandPool, FRAME_OVERLAP>                 m_pools{};
		std::array<std::vector<vk::CommandBuffer>, FRAME_OVERLAP>  m_buffers{};
		std::array<std::uint64_t, FRAME_OVERLAP>                   m_slotFinalValue{};
		std::size_t                                                m_activeSlot = 0;
		std::uint32_t                                              m_nextBufferIndexInSlot = 0;

		std::optional<std::uint32_t>         m_openBufferIndex;
		std::uint32_t                        m_openFrameLocalIndex = 0;
		std::uint32_t                        m_nextFrameLocalIndex = 0;
		std::uint32_t                        m_lastClosedFrameLocalIndex = 0;
		std::vector<vk::SemaphoreSubmitInfo> m_pendingWaits;

		// Design plan's S5 timestamp capture. kMaxBatchesPerFrame is a fixed budget (query pools
		// can't be resized after creation) -- generous for any realistic per-frame batch count on
		// one queue; BeginBatchTimestamp just stops timestamping (returns nullopt) past it rather
		// than throwing.
		static constexpr std::uint32_t    kMaxBatchesPerFrame = 32;
		vk::QueryPool                     m_queryPool{};
		bool                              m_timestampsSupported = false;
		float                             m_timestampPeriodNs = 1.0f;
		std::array<std::uint32_t, FRAME_OVERLAP> m_batchCountInSlot{};
		bool                              m_queryPoolResetThisFrame = false;
	};

	// External (non-graph) synchronization Execute() threads onto the Graphics lane for a frame
	// that also has to satisfy a swapchain's own semaphore contract -- design plan's S4
	// Submission section: "The acquire wait goes on the first G segment that touches Swapchain
	// ... Render-finished is signalled on the last G segment, which also holds Engine's present
	// barrier." Every field defaults to "nothing external to do," so a caller that never
	// constructs one of these (every existing test) gets byte-identical behavior to before this
	// struct existed.
	struct ExternalGraphicsSync {
		std::optional<vk::Semaphore> waitSemaphore;
		vk::PipelineStageFlags2      waitStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
		std::optional<vk::Semaphore> signalSemaphore;
		vk::PipelineStageFlags2      signalStage = vk::PipelineStageFlagBits2::eAllCommands;

		// Called with the Graphics lane's final command buffer for the frame, after every graph
		// node has recorded into it but before it closes -- Engine's hook for appending the
		// swapchain present-layout-transition barrier, which is manual, non-graph recording the
		// same way it is under PhysicalExecutionBackend's single-queue path.
		std::function<void(vk::CommandBuffer)> beforeClose;
	};

	// Drives three QueueLanes plus a QueueHazardTracker through a Graph's existing Schedule
	// (unmodified ASAP leveling from Graph::Compile() -- see this file's header comment) and
	// emits real, synchronized multi-queue command streams. One instance covers a device's whole
	// lifetime; call BeginFrame() once per frame before RunSchedule().
	class QueueRecorder {
	public:
		void Init(
			vk::Device                                device,
			vk::PhysicalDevice                         physicalDevice,
			const std::array<vk::Queue, kNumGpuQueues>& queues,
			const std::array<std::uint32_t, kNumGpuQueues>& families
		) {
			for (std::size_t i = 0; i < kNumGpuQueues; ++i) {
				m_lanes[i].Init(device, physicalDevice, queues[i], families[i]);
			}
		}

		void Destroy() {
			for (auto& lane : m_lanes) {
				lane.Destroy();
			}
		}

		[[nodiscard]] QueueLane& Lane(ExecutionDomain domain) { return m_lanes[GpuQueueIndex(domain)]; }

		// frameIndex must be the same value the caller is about to pass as FrameContext::frameIndex
		// -- each lane's own BeginFrame uses it to rotate FRAME_OVERLAP-bounded command pools and
		// wait, if needed, for this slot's own last use to finish on the GPU. That same wait is
		// what makes it safe to read back whatever that slot's query pool holds right here
		// (design plan's S5): the batches that wrote those timestamps are now confirmed complete.
		void BeginFrame(std::uint64_t frameIndex) {
			m_hazards.BeginFrame();
			m_joinedThisFrame.fill(false);
			for (std::size_t i = 0; i < kNumGpuQueues; ++i) {
				m_lanes[i].BeginFrame(frameIndex);
				auto  durations = m_lanes[i].CollectBatchDurationsMicroseconds(frameIndex);
				auto& namesForSlot = m_batchNames[i][m_lanes[i].ActiveSlot()];
				for (const auto& [batchIndex, micros] : durations) {
					if (batchIndex >= namesForSlot.size() || namesForSlot[batchIndex].empty()) {
						continue;
					}
					// Split one batch's measured duration evenly across its member nodes --
					// several nodes sharing a batch have no relative order/barrier between them
					// (BuildBatches' own grouping rule), so there's no finer-grained real
					// measurement available than the batch's own total.
					const double perNode =
						micros / static_cast<double>(namesForSlot[batchIndex].size());
					for (const auto& name : namesForSlot[batchIndex]) {
						m_weights.UpdateNodeWeight(name, std::max(1, static_cast<int>(perNode)));
					}
				}
				namesForSlot.clear();
			}
		}

		[[nodiscard]] const DynamicWeights& Weights() const { return m_weights; }

		[[nodiscard]] PlacementHysteresis& Hysteresis() { return m_hysteresis; }

		// Test observability only: how many times CutAndWait has actually cut a segment and
		// inserted a real cross-queue wait, cumulative across the recorder's whole lifetime. Lets
		// a test distinguish "stayed correct because a wait genuinely fired" from "stayed correct
		// because nothing in the test needed one" -- zero validation errors alone can't tell
		// those apart.
		[[nodiscard]] std::uint32_t TotalWaitsInserted() const { return m_totalWaitsInserted; }

		// Test observability only, same reasoning as TotalWaitsInserted -- cumulative count of
		// real frame-boundary-join waits EnsureJoined has actually added (as opposed to finding
		// nothing to join and skipping).
		[[nodiscard]] std::uint32_t TotalJoinWaitsInserted() const { return m_totalJoinWaitsInserted; }

		// Executes one PhysicalExecutionBackend-equivalent frame: Setup/Compile/Provision are
		// identical to the single-queue path (PhysicalExecutionBackend::Execute) -- only
		// RunSchedule's recording differs. aliasing is unconditionally off: alias lifetimes are
		// stage-based (PhysicalRegistry.hpp's ImageAliasPool), and lanes now overlap in GPU time,
		// so the two are incompatible (design plan's constraints). mode defaults to ForceDomain,
		// not GraphicsOnly -- this class exists specifically to run more than one queue; a
		// caller that wants GraphicsOnly behavior should use PhysicalExecutionBackend instead,
		// which never pays this class's segment/hazard-tracking overhead at all.
		void Execute(
			Graph&                       graph,
			const FrameContext&          ctx,
			PhysicalResourceRegistry&    registry,
			NodeContext&                 nodeCtx,
			QueuePlacementMode           mode = QueuePlacementMode::ForceDomain,
			const ExternalGraphicsSync&  externalSync = {}
		) {
			graph.Setup(ctx);
			if (auto compileRes = graph.Compile(mode, m_weights, &m_hysteresis); !compileRes) {
				throw std::runtime_error("Graph compilation failed: " + compileRes.error().message);
			}
			registry.Provision(graph.GetSchedule(), graph.Recipes(), ctx.frameIndex, /*enableAliasing=*/false, graph.Nodes(), ctx);

			// Zero-init clears always land on the graphics lane (design plan: "Recorded as
			// graphics writes in a virtual batch at the start of G's first segment") -- open it
			// explicitly before anything else touches anything, so a clear's barrier chains off
			// a real segment and the hazard table sees it as a genuine Graphics-lane write. The
			// external wait (if any) goes right here too -- nothing has been recorded into this
			// segment yet, which is the one and only window AddWait accepts (its own comment).
			vk::CommandBuffer graphicsCmd = Lane(ExecutionDomain::Graphics).OpenIfNeeded();
			if (externalSync.waitSemaphore.has_value()) {
				Lane(ExecutionDomain::Graphics).AddWait(*externalSync.waitSemaphore, 0, externalSync.waitStage);
			}
			EnsureJoined(ExecutionDomain::Graphics);
			ZeroInitializeUndefinedReads(graphicsCmd, registry, graph);
			ZeroInitializeInactiveReads(graphicsCmd, registry, graph);
			for (ResourceId key : ClearedKeysThisFrame(graph, registry)) {
				m_hazards.RecordTouch(
					registry.ResolveId(key),
					ExecutionDomain::Graphics,
					/*isWriteOrLayoutChange=*/true,
					Lane(ExecutionDomain::Graphics).OpenSegmentIndex()
				);
			}

			RunSchedule(graph, nodeCtx, registry);

			// The Graphics lane's beforeClose hook (Engine's present-layout-transition barrier)
			// runs on whatever its current segment is -- opening a fresh one if RunSchedule
			// happened not to leave one open (no Graphics-placed work this frame is legal, if
			// unusual) is correct either way: a segment with nothing but this one barrier in it
			// is still a valid, if minimal, submission.
			if (externalSync.beforeClose) {
				externalSync.beforeClose(Lane(ExecutionDomain::Graphics).OpenIfNeeded());
			}

			for (std::size_t i = 0; i < kNumGpuQueues; ++i) {
				if (i == GpuQueueIndex(ExecutionDomain::Graphics)) {
					m_lanes[i].CloseAndSubmit(externalSync.signalSemaphore, externalSync.signalStage);
				} else {
					m_lanes[i].CloseAndSubmit();
				}
			}
		}

		// Exception path, mirroring PhysicalExecutionBackend::Execute's single-queue catch block
		// (Engine.cpp's DrawFrame): abandons whatever any lane had partially recorded this frame
		// (design plan's S4 catch path, "never submit partially recorded buffers"), then does one
		// empty Graphics submission purely to consume the external wait semaphore (so it isn't
		// left signaled for the next acquire that reuses it) and advance the Graphics lane's own
		// timeline by exactly one value, matching "signals G's next value." Never signals
		// signalSemaphore and never touches Compute/Transfer again this frame -- this frame is
		// being dropped, not shown, same as the single-queue path's own recovery submit.
		void AbandonFrame(std::optional<vk::Semaphore> waitSemaphore, vk::PipelineStageFlags2 waitStage) {
			for (auto& lane : m_lanes) {
				lane.Abandon();
			}
			QueueLane& g = Lane(ExecutionDomain::Graphics);
			g.OpenIfNeeded();
			if (waitSemaphore.has_value()) {
				g.AddWait(*waitSemaphore, 0, waitStage);
			}
			g.CloseAndSubmit();
		}

	private:
		// ClearImageToZero/ClearBufferToZero don't report what they cleared, and re-deriving
		// exactly which keys they touched (History<K> first-use plus §E's zero-init list) would
		// duplicate ZeroInitializeUndefinedReads/ZeroInitializeInactiveReads's own walks --
		// walking the same two lists here instead, read-only, keeps that logic in one place.
		static std::vector<ResourceId> ClearedKeysThisFrame(Graph& graph, PhysicalResourceRegistry& registry) {
			std::vector<ResourceId> keys;
			for (const auto& recipe : graph.Recipes()) {
				if (!recipe.isActive) {
					continue;
				}
				for (const auto& r : recipe.realizations) {
					if (r.access != AccessKind::Read || !r.key->isHistory) {
						continue;
					}
					if (auto tex = registry.GetTexture(r.key); tex && tex->HasDefinedContents()) {
						keys.push_back(r.key);
					}
				}
			}
			for (const auto& zeroInit : graph.ZeroInitReads()) {
				keys.push_back(zeroInit.second);
			}
			return keys;
		}

		// Opens `queue`'s very first segment of the frame (if it doesn't have one yet) and makes
		// it wait on every other queue's last submitted value from the *previous* frame -- the
		// frame-boundary join (design plan's §Barrier inventory). A queue that never ran last
		// frame has LastSubmittedValue() == 0, which is already trivially satisfied (a fresh
		// timeline semaphore starts at 0), so skipping it isn't just an optimization: waiting on
		// value 0 explicitly is also legal but pointless. Idempotent per frame via
		// m_joinedThisFrame -- safe to call before every queue's first touch.
		//
		// Deliberately does not call m_hazards.MarkCaughtUp: that index would be the *other*
		// queue's frame-local segment number from last frame's numbering, meaningless against
		// m_hazards' own this-frame-only state (reset by BeginFrame, same as every lane's own
		// frame-local counter) -- comparing stale numbers from a different frame's numbering
		// space could even wrongly suppress a real wait this frame if the old value happened to
		// be the larger one. The real GPU-level wait AddWait inserts below is what actually
		// carries the cross-frame guarantee; the hazard table's own bookkeeping doesn't need
		// (and must not try) to reason about segment numbers from outside this frame at all.
		void EnsureJoined(ExecutionDomain queue) {
			const std::size_t q = GpuQueueIndex(queue);
			if (m_joinedThisFrame[q]) {
				return;
			}
			m_joinedThisFrame[q] = true;

			bool opened = false;
			for (std::size_t p = 0; p < kNumGpuQueues; ++p) {
				if (p == q) {
					continue;
				}
				QueueLane& other = m_lanes[p];
				if (other.LastSubmittedValue() == 0) {
					continue;
				}
				if (!opened) {
					Lane(queue).OpenIfNeeded();
					opened = true;
				}
				Lane(queue).AddWait(other.Timeline(), other.LastSubmittedValue(), vk::PipelineStageFlagBits2::eAllCommands);
				++m_totalJoinWaitsInserted;
			}
		}

		// Closes `other`'s open segment (if any -- it may already be closed, in which case
		// whatever it last submitted is exactly what gets waited on) and adds a wait for it to
		// `queue`'s segment, cutting `queue`'s own open segment first if it already has commands
		// recorded in it (a wait can only cover a segment's own start, never retroactively).
		void CutAndWait(ExecutionDomain queue, ExecutionDomain other) {
			QueueLane& otherLane = m_lanes[GpuQueueIndex(other)];
			// A no-op if `other` has nothing open: CutAndWait only ever runs after
			// QueuesToWaitOn named `other`, which only happens once `other` has a recorded
			// touch -- so LastSubmittedValue() below is always > 0 by the time we read it,
			// whether this call closes it just now or it was already closed earlier.
			// LastClosedFrameLocalIndex() is read *after* CloseAndSubmit(), not from its return
			// value, specifically so the no-op case (already closed earlier this frame) still
			// gets the right frame-local index for MarkCaughtUp below.
			otherLane.CloseAndSubmit();
			const std::uint32_t otherFrameLocalIndex = otherLane.LastClosedFrameLocalIndex();

			QueueLane& queueLane = m_lanes[GpuQueueIndex(queue)];
			queueLane.CloseAndSubmit(); // cut: whatever queue already had queued, submit it now
			queueLane.OpenIfNeeded();   // fresh segment, nothing recorded into it yet
			queueLane.AddWait(otherLane.Timeline(), otherLane.LastSubmittedValue(), vk::PipelineStageFlagBits2::eAllCommands);
			m_hazards.MarkCaughtUp(queue, other, otherFrameLocalIndex);
			++m_totalWaitsInserted;
		}

		// stage.queue is already a single value per stage -- BuildBatches (Scheduler.hpp) only
		// ever groups nodes that landed on the same placed queue, so this no longer needs to
		// split a stage's nodes by domain the way an earlier revision of this function did (back
		// when a node's own Recipe::domain *was* its queue, with no placement step in between).
		void RunSchedule(Graph& graph, NodeContext& nodeCtx, PhysicalResourceRegistry& registry) {
			const Schedule& schedule = graph.GetSchedule();
			const auto      recipes = graph.Recipes();

			for (const auto& stage : schedule.stages) {
				if (stage.nodes.empty()) {
					continue;
				}
				if (stage.queue == ExecutionDomain::Host) {
					// No GPU segment, no barriers -- BuildBatches gives every Host node its own
					// singleton stage, so this loop only ever runs once per Host stage.
					for (std::size_t nodeIndex : stage.nodes) {
						nodeCtx.cmd = CommandBuffer{};
						nodeCtx.insideRendering = false;
						nodeCtx.domain = ExecutionDomain::Host;
						nodeCtx.placedQueue = ExecutionDomain::Host;
						nodeCtx.declared = recipes[nodeIndex].realizations;
						graph.ExecuteNode(nodeIndex, nodeCtx);
					}
					continue;
				}

				EnsureJoined(stage.queue);
				RunQueueBatch(stage, graph, nodeCtx, registry);
			}
		}

		// One stage's worth of work, already confined to a single queue (stage.queue):
		// resolve its acquire batch, hazard-check every resource in it (cutting/waiting as
		// needed), emit the sanitized barriers, record the nodes, then release this stage's
		// postBarriers. Mirrors PhysicalExecutionBackend::RunSchedule's single-queue walk almost
		// exactly -- the only real difference is the hazard-check pass between Resolve and Emit.
		void RunQueueBatch(const ScheduleStage& stage, Graph& graph, NodeContext& nodeCtx, PhysicalResourceRegistry& registry) {
			const auto            recipes = graph.Recipes();
			const ExecutionDomain queue = stage.queue;

			BarrierBatch acquireBatch;
			for (const auto& mb : stage.preBarriers.Items()) {
				acquireBatch.Add(mb);
			}
			for (std::size_t nodeIndex : stage.nodes) {
				for (const auto& r : recipes[nodeIndex].realizations) {
					acquireBatch.Add(
						MemoryBarrier{
							.resource = r.key,
							.access = r.access,
							.srcDomain = recipes[nodeIndex].domain,
							.dstDomain = recipes[nodeIndex].domain,
						}
					);
				}
			}

			// Hazard check: every resource this queue is about to touch, against every other
			// queue's last relevant touch of it. IsWriteOrLayoutChangeAccess needs mb.dstDomain
			// -- the *declared* domain, what DeriveImageState actually derives a layout from (a
			// property of the operation a node's Execute() performs, not of which queue
			// recorded it) -- but the hazard table itself is keyed on `queue`, the *placed*
			// queue: a Compute-domain node HEFT placed onto Graphics still dispatches a real
			// compute shader (declared domain unaffected), but it's Graphics' own timeline a
			// cross-queue wait needs to reason about. Conflating the two here was a real S1 bug
			// invisible under ForceDomain, where they always happened to be the same value.
			for (const auto& mb : acquireBatch.Items()) {
				const ResourceId resolved = registry.ResolveId(mb.resource);
				const bool       writeish = IsWriteOrLayoutChangeAccess(registry, mb.resource, mb.access, mb.dstDomain);
				for (ExecutionDomain other : m_hazards.QueuesToWaitOn(resolved, queue, writeish)) {
					CutAndWait(queue, other);
				}
			}

			vk::CommandBuffer cmd = Lane(queue).OpenIfNeeded();
			BarrierTranslator::TranslateAndDispatch(cmd, registry, acquireBatch, BarrierPhase::Acquire, queue);

			const std::uint32_t segmentIndex = Lane(queue).OpenSegmentIndex();
			for (const auto& mb : acquireBatch.Items()) {
				const ResourceId resolved = registry.ResolveId(mb.resource);
				const bool       writeish = IsWriteOrLayoutChangeAccess(registry, mb.resource, mb.access, mb.dstDomain);
				m_hazards.RecordTouch(resolved, queue, writeish, segmentIndex);
			}

			// Design plan's S5: brackets just this batch's node-execution work (not the acquire
			// barriers just emitted above, nor the release barriers below) so the measured
			// duration reflects real GPU work, not synchronization overhead. Records which node
			// names are in this batch alongside it -- BeginFrame reads both back together once
			// this slot's prior use is confirmed complete, feeding the EMA (design plan's S5,
			// "readback ... then an EMA keyed by node name").
			const std::optional<std::uint32_t> timestampBatchIndex = Lane(queue).BeginBatchTimestamp(cmd);
			if (timestampBatchIndex.has_value()) {
				auto& namesForSlot = m_batchNames[GpuQueueIndex(queue)][Lane(queue).ActiveSlot()];
				if (*timestampBatchIndex >= namesForSlot.size()) {
					namesForSlot.resize(*timestampBatchIndex + 1);
				}
				auto& namesThisBatch = namesForSlot[*timestampBatchIndex];
				const auto nodes = graph.Nodes();
				for (std::size_t nodeIndex : stage.nodes) {
					namesThisBatch.emplace_back(nodes[nodeIndex].Descriptor().name);
				}
			}

			for (std::size_t nodeIndex : stage.nodes) {
				const auto& recipe = recipes[nodeIndex];
				const bool  activeRendering = DynamicRenderingWrapper::Begin(cmd, registry, recipe);
				nodeCtx.cmd = CommandBuffer{static_cast<void*>(static_cast<VkCommandBuffer>(cmd))};
				nodeCtx.insideRendering = activeRendering;
				nodeCtx.domain = recipe.domain;
				nodeCtx.placedQueue = queue; // stage.queue -- may differ from recipe.domain under Heft
				nodeCtx.declared = recipe.realizations;
				graph.ExecuteNode(nodeIndex, nodeCtx);
				if (activeRendering) {
					DynamicRenderingWrapper::End(cmd);
				}
				nodeCtx.insideRendering = false;

				for (const auto& r : recipe.realizations) {
					if (r.access != AccessKind::Write && r.access != AccessKind::ReadWrite) {
						continue;
					}
					if (auto tex = registry.GetTexture(r.key)) {
						tex->SetHasDefinedContents(true);
					} else if (auto buf = registry.GetBuffer(r.key)) {
						buf->SetHasDefinedContents(true);
					}
				}
			}

			if (timestampBatchIndex.has_value()) {
				Lane(queue).EndBatchTimestamp(cmd, *timestampBatchIndex);
			}

			BarrierTranslator::TranslateAndDispatch(cmd, registry, stage.postBarriers, BarrierPhase::Release);
		}

		std::array<QueueLane, kNumGpuQueues>                                              m_lanes;
		QueueHazardTracker                                                                m_hazards;
		std::array<bool, kNumGpuQueues>                                                   m_joinedThisFrame{};
		std::uint32_t                                                                     m_totalWaitsInserted = 0;
		std::uint32_t                                                                     m_totalJoinWaitsInserted = 0;

		// Design plan's S5 timestamp/EMA feedback loop. m_batchNames[lane][slot][batchIndex] =
		// node names recorded into that (queue, frame-slot, batch) triple -- cleared and
		// refilled each time BeginFrame rotates back into that slot, right after reading back
		// (and attributing to m_weights) whatever was recorded into it last time.
		std::array<std::array<std::vector<std::vector<std::string>>, FRAME_OVERLAP>, kNumGpuQueues> m_batchNames;
		DynamicWeights      m_weights;
		PlacementHysteresis m_hysteresis;
	};

} // namespace brassica::graph
