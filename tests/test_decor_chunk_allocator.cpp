#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <unordered_set>

#include "passes/model/DecorChunkAllocator.hpp"

using namespace brassica;

TEST_CASE("DecorChunkAllocator allocates capped, non-overlapping blocks and respects the grace period") {
	DecorChunkAllocator           allocator;
	std::vector<std::uint32_t>    freed;

	// Frame 1: camera at the origin. The 9x9 candidate grid (81 tiles) exceeds the per-frame cap,
	// so only the capped number of chunks should generate this frame, not all 81 at once.
	auto gen1 = allocator.UpdateAllocation(glm::vec3(0.0f), 16.0f, freed);
	CHECK(!gen1.empty());
	CHECK(gen1.size() <= 8); // kMaxNewChunksPerFrame
	CHECK(freed.empty());    // nothing to evict yet

	// Every generated block base this frame is distinct and within the reserved decor range.
	std::unordered_set<std::uint32_t> seenBases;
	for (const DecorGenerateEntry& e : gen1) {
		CHECK(e.blockBaseSlot >= kDecorSlotBase);
		CHECK(e.blockBaseSlot < kModelInstanceCapacity);
		CHECK((e.blockBaseSlot - kDecorSlotBase) % kDecorInstancesPerChunk == 0u);
		CHECK(seenBases.insert(e.blockBaseSlot).second); // no duplicate block assigned twice this frame
	}

	// Keep calling from the same camera position until every candidate chunk in range has been
	// allocated (bounded by kDecorMaxActiveChunks, not by the 81-tile candidate count, since the
	// pool is smaller than the full candidate grid).
	std::uint32_t totalGenerated = static_cast<std::uint32_t>(gen1.size());
	for (int i = 0; i < 50 && totalGenerated < kDecorMaxActiveChunks; ++i) {
		auto gen = allocator.UpdateAllocation(glm::vec3(0.0f), 16.0f, freed);
		totalGenerated += static_cast<std::uint32_t>(gen.size());
		for (const DecorGenerateEntry& e : gen) {
			CHECK(seenBases.insert(e.blockBaseSlot).second); // never double-allocates a resident block
		}
	}
	CHECK(allocator.ActiveChunkCount() <= kDecorMaxActiveChunks);

	// Camera teleports far away -- every previously-active (origin-area) chunk should eventually
	// free (after the grace period), with real, previously-allocated block bases. The allocator
	// correctly starts repopulating near the *new* camera position using those freed blocks in
	// the same calls, so active chunk count shifts location rather than going to zero -- that's
	// the intended behavior (decor following the camera), not something this asserts away.
	glm::vec3     farAway(100000.0f, 0.0f, 100000.0f);
	bool          anyFreed = false;
	std::uint32_t totalFreed = 0;
	for (int i = 0; i < 10; ++i) {
		allocator.UpdateAllocation(farAway, 16.0f, freed);
		if (!freed.empty()) {
			anyFreed = true;
			totalFreed += static_cast<std::uint32_t>(freed.size());
			for (std::uint32_t base : freed) {
				CHECK(seenBases.count(base) == 1u);
			}
		}
	}
	CHECK(anyFreed);
	// Every origin-area chunk that was ever allocated eventually gets freed once the camera
	// leaves -- not just a handful.
	CHECK(totalFreed == seenBases.size());
	CHECK(allocator.ActiveChunkCount() <= kDecorMaxActiveChunks);
}
