#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "passes/model/ModelTypes.hpp"

namespace brassica {

	struct DecorChunkKey {
		int32_t x{0};
		int32_t z{0};
		bool    operator==(const DecorChunkKey& other) const { return x == other.x && z == other.z; }
	};

} // namespace brassica

namespace std {
	template <>
	struct hash<brassica::DecorChunkKey> {
		std::size_t operator()(const brassica::DecorChunkKey& k) const noexcept {
			return (static_cast<std::size_t>(static_cast<std::uint32_t>(k.x)) << 32) ^ static_cast<std::uint32_t>(k.z);
		}
	};
} // namespace std

namespace brassica {

	// One chunk's worth of decor generation work -- the CPU hands this to a GPU gen node, which
	// dispatches one workgroup per entry to actually place that chunk's decor.
	struct DecorGenerateEntry {
		glm::vec3     worldOffset;   // this tile's min XZ corner (Y unused -- terrain height is sampled in the gen shader)
		std::uint32_t blockBaseSlot; // kDecorSlotBase + blockIndex * kDecorInstancesPerChunk
		std::uint32_t cellSeed;      // deterministic from the tile's own grid coordinate alone
	};

	// CPU-side chunk -> block allocation bookkeeping for Stage 5.2's persistent, disocclusion-
	// driven decor population -- adapted from external/boidish/src/decor_manager.cpp's
	// DecorManager::_UpdateAllocation (a tried, proven-at-scale pattern), not a 1:1 port:
	// Brassica has no terrain-chunk-list API to query (the terrain clipmap is pure GPU-texture
	// resampling against camera position, no CPU-side tile bookkeeping), so this generates its
	// own square candidate grid directly from camera position + a tile size, rather than
	// querying a render manager. Distance-only priority for now, not boidish's facing/turn-
	// prediction scoring -- a reasonable first cut; revisit if churn near the view frustum's
	// edges looks bad in practice.
	class DecorChunkAllocator {
	public:
		// Recomputes which chunks should be resident given the camera's current position,
		// evicts chunks that have been out of range for longer than the grace period, allocates
		// blocks for newly-needed chunks (capped per frame), and returns the chunks that need
		// (re)generation this frame. outFreedBlockBases receives the base slot of every block
		// freed this call, so the caller can clear those slots' alive bits.
		std::vector<DecorGenerateEntry> UpdateAllocation(
			glm::vec3 cameraPosition, float tileSize, std::vector<std::uint32_t>& outFreedBlockBases
		);

		[[nodiscard]] std::uint32_t ActiveChunkCount() const { return static_cast<std::uint32_t>(m_activeChunks.size()); }

	private:
		struct ChunkAllocation {
			std::uint32_t blockIndex{0};
			std::uint32_t lastSeenFrame{0};
		};

		std::unordered_map<DecorChunkKey, ChunkAllocation> m_activeChunks;
		std::vector<std::uint32_t>                         m_freeBlocks;
		std::uint32_t                                      m_frameCounter{0};
		bool                                                m_freeBlocksInitialized{false};

		// Keeps a chunk's block alive for this many frames after it drops out of the candidate
		// set, so camera jitter near a tile boundary doesn't thrash allocation every frame.
		static constexpr std::uint32_t kGracePeriodFrames = 2;
		// Caps new-chunk generation per frame so a camera teleport/fast flythrough (many tiles
		// becoming candidates at once) doesn't spike a single frame.
		static constexpr std::uint32_t kMaxNewChunksPerFrame = 8;
		// A 9x9 grid (radius 4) -- matches kDecorMaxActiveChunks' own sizing comment
		// (ModelTypes.hpp).
		static constexpr int kCandidateRadiusTiles = 4;
	};

} // namespace brassica
