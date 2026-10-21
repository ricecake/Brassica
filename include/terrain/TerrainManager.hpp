#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "passes/ResourceKeys.hpp"
#include "terrain/ITerrainManager.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	// One planned (or in-flight, or just-completed) readback request: the region to copy, plus
	// the world-space context needed to interpret it once it lands. Filled in by PlanReadback,
	// consumed by TerrainReadbackNode's Execute (graph/TerrainReadbackNode.hpp), then copied
	// into the cached* fields below once PollReadbackData confirms the GPU work completed --
	// same split today's code already had (pending* vs cached*), just named as one struct
	// instead of two sets of loose members.
	struct TerrainReadbackRegion {
		uint32_t     arrayLayer = 0;
		uint32_t     mipLevel = 0;
		vk::Offset2D offset{};
		vk::Extent2D extent{};
		glm::vec2    originWorld{0.0f};
		float        texelSize{0.5f};
		glm::vec2    camPos{0.0f};
	};

	// CPU-side terrain config/state (numLODs/baseTexelSize, Regenerate()) plus the GPU->CPU
	// readback system backing GetCachedGroundHeight() -- the two things left once the GPU image/
	// view/sampler ownership, the CPU noise-based paging/generation machinery, and the noise-based
	// CPU height sampler (SampleTerrain -- it queried an unrelated noise function, not what's
	// actually rendered) were all deleted. The terrain arrays are now normal pass-owned graph
	// resources, Created<> by TerrainGenNode; see terrain/TerrainClipmap.hpp's git history for
	// what used to live here. Named for what it actually does now, not what it used to be.
	//
	// The readback itself is a real graph node now (TerrainReadbackNode, passes/
	// TerrainReadbackNode.hpp) -- this class no longer owns a transfer queue, command pool, or
	// semaphore of its own. It decides *whether* a new readback is wanted and *what* region
	// (PlanReadback, moved here from Engine::UpdateCamera), the node records the actual copy
	// through the graph's own barrier-synthesized command buffer, and this class picks the
	// result back up by frame retirement (PollReadbackData, keyed off Engine's own
	// frameTimelineSemaphore -- no dedicated semaphore needed, since the copy now completes
	// exactly when the rest of its frame does). That's what closes the design plan's bug 6: the
	// old TriggerImageRegionReadbackAsync transitioned a graph image on its own transfer queue
	// with no semaphore and no tracked-state update at all -- a real race on any device with a
	// separate transfer family, now gone because there is no side-channel submit left.
	class TerrainManager: public ITerrainManager {
	public:
		TerrainManager() = default;
		~TerrainManager() override = default;

		void Initialize() override { m_initialized = true; }

		void Shutdown() override {
			Cleanup();
			m_initialized = false;
		}

		State GetState() const override { return State{numLODs, baseTexelSize}; }

		void SetState(const State& state) override {
			numLODs = state.numLODs;
			baseTexelSize = state.baseTexelSize;
		}

		// concurrentFamilies: forwarded straight to the readback staging buffer's VkBufferCreateInfo
		// (TerrainManager.cpp) -- this buffer is out-of-registry (TerrainReadbackStaging is only
		// ever RegisterImportedBuffer'd, never provisioned), so it doesn't pick up
		// PhysicalResourceRegistry's own CONCURRENT handling automatically. See the design plan's
		// S4 "Queue-family ownership": a Transfer-domain TerrainReadbackNode writes it, the CPU
		// reads it, and neither this engine has a real ownership-transfer barrier to give it.
		void Init(
			vk::Device                     device,
			VmaAllocator                   allocator,
			uint32_t                       numLODs = constants::Class::Terrain::DefaultMaxLODs,
			float                          baseTexelSize = constants::Class::Terrain::BaseTexelSize,
			std::span<const std::uint32_t> concurrentFamilies = {}
		);
		void Cleanup();

		void Regenerate() override { m_forceRegenerate = true; }

		bool ShouldForceRegeneration() { return m_forceRegenerate; }

		void ResetForceRegeneration() { m_forceRegenerate = false; }

		uint32_t GetNumLODs() const { return numLODs; }

		float GetBaseTexelSize() const { return baseTexelSize; }

		[[nodiscard]] vk::Buffer GetReadbackStagingBuffer() const { return readbackStagingBuffer; }

		static constexpr vk::DeviceSize kReadbackStagingBytes = 1024 * 1024;

		// Decides whether a new ground-height-and-normal readback should be requested this
		// frame (same cooldown this used to apply inline in Engine::UpdateCamera: skip while
		// one is still in flight, and skip if the camera hasn't moved far enough since the last
		// one landed) and, if so, computes the region to copy -- a 128x128 window at LOD 0 mip
		// 0 (64m x 64m footprint) centered on camPos2D, clamped to the clipmap's bounds. Reads
		// the real per-texel height+normal map (TerrainClipmapTexture), not
		// TerrainMinMaxTexture's hierarchical occlusion bound -- it's written directly by
		// TerrainGenNode with no downstream pass to wait on, and carries the surface normal
		// (GetInterpolatedGroundNormal) for physics besides.
		void PlanReadback(glm::vec2 camPos2D);

		[[nodiscard]] bool HasPendingReadback() const { return hasPendingReadback; }

		[[nodiscard]] const TerrainReadbackRegion& GetPendingReadback() const { return pendingRegion; }

		// Called by TerrainReadbackNode::Execute once it has actually recorded the copy, with
		// NodeContext::placedQueue -- which real GPU queue's timeline will eventually carry this
		// copy's completion. Under PhysicalExecutionBackend that's always Graphics; under
		// MultiQueueExecutionBackend it's whichever queue ForceDomain/Heft actually placed this
		// Transfer-domain node on, which the two can disagree about. The real (semaphore, value)
		// pair isn't knowable yet at record time (QueueLane's timeline value is only assigned
		// when a segment actually closes, which can happen more than once per frame) -- see
		// CaptureReadbackCompletion, which Engine calls once it is, right after this frame's
		// backend Execute() call returns.
		void MarkReadbackSubmitted(graph::ExecutionDomain placedQueue) {
			pendingReadbackQueue = placedQueue;
			awaitingCompletionCapture = true;
			hasPendingReadback = false;
		}

		[[nodiscard]] bool HasPendingCompletionCapture() const { return awaitingCompletionCapture; }

		[[nodiscard]] graph::ExecutionDomain PendingReadbackQueue() const { return pendingReadbackQueue; }

		// Called by Engine.cpp immediately after this frame's backend Execute() returns, only
		// when HasPendingCompletionCapture() -- by then the real completion (semaphore, value)
		// pair for whichever queue actually ran the readback this frame is known: Engine's own
		// frameTimelineSemaphore + frameIndex+1 under GraphicsOnly (TerrainReadbackNode always
		// lands on Graphics there, which signals that value exactly once per frame), or
		// PendingReadbackQueue()'s own QueueLane::Timeline() + LastSubmittedValue() under
		// ForceDomain/Heft.
		void CaptureReadbackCompletion(vk::Semaphore completionSemaphore, uint64_t completionValue) {
			if (!awaitingCompletionCapture) {
				return;
			}
			awaitingCompletionCapture = false;
			pendingCompletionSemaphore = completionSemaphore;
			pendingCompletionValue = completionValue;
			readbackInFlight = true;
		}

		void PollReadbackData(vk::Device device);

		// Evaluates 2D bilinear height interpolation across the cached terrain height readback grid.
		float GetInterpolatedGroundHeight(glm::vec2 worldXZ, float fallback) const;
		float GetInterpolatedGroundHeight(float fallback) const;

		float GetCachedGroundHeight(float fallback) const;
		float GetCachedGroundHeight(glm::vec2 worldXZ, float fallback) const;

		// Same bilinear grid as GetInterpolatedGroundHeight, reading the readback's normal.xyz
		// channels (TerrainClipmapTexture's vec4(h, normal.x, normal.y, normal.z)) instead of
		// height -- re-normalized after interpolation since blending four unit normals doesn't
		// generally stay unit length. For physics (slope-dependent sliding, collision response).
		glm::vec3 GetInterpolatedGroundNormal(glm::vec2 worldXZ, glm::vec3 fallback) const;
		glm::vec3 GetInterpolatedGroundNormal(glm::vec3 fallback) const;

		bool IsReadbackInFlight() const { return readbackInFlight; }

		bool HasReadbackData() const { return hasReadbackData; }

		glm::vec2 GetLastReadbackCamPos() const { return cachedReadbackCamPos; }

	private:
		uint32_t numLODs{constants::Class::Terrain::DefaultMaxLODs};
		float    baseTexelSize{0.5f};
		bool     m_forceRegenerate{true};

		vk::Device   device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};

		vk::Buffer    readbackStagingBuffer{nullptr};
		VmaAllocation readbackStagingAllocation{VK_NULL_HANDLE};
		void*         readbackStagingMapped{nullptr};

		bool                  hasPendingReadback{false};
		TerrainReadbackRegion pendingRegion{};

		graph::ExecutionDomain pendingReadbackQueue{graph::ExecutionDomain::Graphics};
		bool                   awaitingCompletionCapture{false};
		vk::Semaphore          pendingCompletionSemaphore{nullptr};
		uint64_t               pendingCompletionValue{0};
		bool                   readbackInFlight{false};

		std::vector<glm::vec4> cachedReadbackData;
		uint32_t               cachedReadbackWidth{0};
		uint32_t               cachedReadbackHeight{0};
		glm::vec2              cachedReadbackOriginWorld{0.0f};
		float                  cachedReadbackTexelSize{0.5f};
		glm::vec2              cachedReadbackCamPos{0.0f};
		bool                   hasReadbackData{false};
	};

	// TerrainClipmapDesc/TerrainMinMaxDesc/TerrainBiomeDesc, and the type-owned Desc() methods on
	// TerrainClipmapTexture/TerrainMinMaxTexture/TerrainBiomeTexture that call them, moved to
	// passes/ResourceKeys.hpp -- the key structs live there, and a type's Desc() must be defined
	// alongside the type itself, not in a separate header reached only by some of its callers.

} // namespace brassica
