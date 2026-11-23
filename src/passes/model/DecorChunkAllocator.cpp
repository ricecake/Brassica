#include "passes/model/DecorChunkAllocator.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	namespace {

		// Same hash shape as FoliageNode's own cellSeed derivation (shaders/foliage.mesh:
		// cellSeed = hashUint(cellCoord.x*1973u ^ cellCoord.y*9277u ^ lodLevel*101u)), evaluated
		// on the CPU for a whole tile instead of a sub-tile cell -- a real, deterministic
		// per-tile seed, not an arbitrary one.
		std::uint32_t HashTileCoord(std::int32_t x, std::int32_t z) {
			std::uint32_t ux = static_cast<std::uint32_t>(x);
			std::uint32_t uz = static_cast<std::uint32_t>(z);
			std::uint32_t h = (ux * 1973u) ^ (uz * 9277u);
			h ^= h >> 16;
			h *= 0x7feb352du;
			h ^= h >> 15;
			h *= 0x846ca68bu;
			h ^= h >> 16;
			return h;
		}

	} // namespace

	std::vector<DecorGenerateEntry> DecorChunkAllocator::UpdateAllocation(
		glm::vec3 cameraPosition, float tileSize, std::vector<std::uint32_t>& outFreedBlockBases
	) {
		++m_frameCounter;
		outFreedBlockBases.clear();

		if (!m_freeBlocksInitialized) {
			m_freeBlocks.reserve(kDecorMaxActiveChunks);
			for (std::uint32_t b = kDecorMaxActiveChunks; b-- > 0;) {
				m_freeBlocks.push_back(b); // reverse order: block 0 pops first
			}
			m_freeBlocksInitialized = true;
		}

		auto centerX = static_cast<std::int32_t>(std::floor(cameraPosition.x / tileSize));
		auto centerZ = static_cast<std::int32_t>(std::floor(cameraPosition.z / tileSize));

		struct Candidate {
			DecorChunkKey key;
			float         priority; // lower = generate sooner
		};
		std::vector<Candidate> candidates;
		candidates.reserve(static_cast<std::size_t>(2 * kCandidateRadiusTiles + 1) * (2 * kCandidateRadiusTiles + 1));
		for (int dz = -kCandidateRadiusTiles; dz <= kCandidateRadiusTiles; ++dz) {
			for (int dx = -kCandidateRadiusTiles; dx <= kCandidateRadiusTiles; ++dx) {
				DecorChunkKey key{centerX + dx, centerZ + dz};
				glm::vec2     tileCenter((key.x + 0.5f) * tileSize, (key.z + 0.5f) * tileSize);
				float         dist = glm::length(tileCenter - glm::vec2(cameraPosition.x, cameraPosition.z));
				candidates.push_back(Candidate{key, dist});
			}
		}

		// Eviction: any active chunk no longer in the candidate set, absent for longer than the
		// grace period, frees its block.
		for (auto it = m_activeChunks.begin(); it != m_activeChunks.end();) {
			bool present = std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& c) { return c.key == it->first; });
			if (present) {
				it->second.lastSeenFrame = m_frameCounter;
				++it;
			} else if (m_frameCounter - it->second.lastSeenFrame > kGracePeriodFrames) {
				std::uint32_t blockBase = kDecorSlotBase + it->second.blockIndex * kDecorInstancesPerChunk;
				outFreedBlockBases.push_back(blockBase);
				m_freeBlocks.push_back(it->second.blockIndex);
				it = m_activeChunks.erase(it);
			} else {
				++it;
			}
		}

		std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.priority < b.priority; });

		std::vector<DecorGenerateEntry> toGenerate;
		std::uint32_t                   newChunks = 0;
		for (const Candidate& cand : candidates) {
			if (m_activeChunks.find(cand.key) != m_activeChunks.end()) {
				// Already resident -- decor chunks are static once placed (no per-frame
				// dirtying the way terrain deformation would need), nothing to regenerate.
				continue;
			}
			if (newChunks >= kMaxNewChunksPerFrame || m_freeBlocks.empty()) {
				continue; // deferred to a later frame
			}
			std::uint32_t blockIndex = m_freeBlocks.back();
			m_freeBlocks.pop_back();
			m_activeChunks[cand.key] = ChunkAllocation{blockIndex, m_frameCounter};

			DecorGenerateEntry entry;
			entry.worldOffset = glm::vec3(cand.key.x * tileSize, 0.0f, cand.key.z * tileSize);
			entry.blockBaseSlot = kDecorSlotBase + blockIndex * kDecorInstancesPerChunk;
			entry.cellSeed = HashTileCoord(cand.key.x, cand.key.z);
			toGenerate.push_back(entry);
			++newChunks;
		}

		return toGenerate;
	}

} // namespace brassica
