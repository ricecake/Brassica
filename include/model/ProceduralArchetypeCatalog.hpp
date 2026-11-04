#pragma once

#include <cstdint>

namespace brassica {

	class ModelRegistry;

	// First index of each contiguous model-index run RegisterProceduralArchetypes() registers --
	// Stage 5.2's decor generation shader (decor_gen_chunk.comp) needs these to pick a real
	// modelIndex per placed instance (treeModelBase + variant, etc.), since ModelRegistry hands
	// out indices in registration order and nothing else in the engine knows this catalog's
	// layout otherwise.
	struct ProceduralArchetypeIndices {
		std::uint32_t treeBase{0}; // 3 consecutive tree variants
		std::uint32_t bushBase{0}; // 2 consecutive bush variants
		std::uint32_t rockBase{0}; // 3 consecutive rock variants
	};

	// Stage 5.1 of the GPU-driven model-instance plan: the real fixed archetype catalog
	// (confirmed with sgreenhusted: a small bounded set, not per-instance/per-seed infinite
	// variation), replacing the old GPU-baked FoliageBakeNode/FoliageArchetypeRenderNode tree/
	// bush path and rocks' live-every-frame emitRockItem path. Bushes reuse
	// BuildProceduralBranchingTree with different parameters (shorter, more branches per node,
	// wider pitch spread) rather than a separate generator -- the only real difference between
	// "tree" and "bush" here is parametrization, not mechanism. Registers every variant via the
	// unchanged ModelRegistry::RegisterAsset(), same call site Engine.cpp already uses for the
	// cow/procedural-tree/Manifold-test-tree.
	ProceduralArchetypeIndices RegisterProceduralArchetypes(ModelRegistry& registry);

} // namespace brassica
