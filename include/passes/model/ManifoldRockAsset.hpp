#pragma once

#include <cstdint>

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// Stage 5.1 of the GPU-driven model-instance plan: rocks never had an archetype at all
	// before (the old foliage pipeline generates them live, every frame, inline -- see
	// shaders/foliage.mesh's emitRockItem). Unlike trees/bushes, rocks don't need wind, so
	// there's no skeleton here at all -- jointCount stays 0, matching the jointless-model
	// convention ModelAsset.hpp/GltfModel::BakeClips already document (a single identity "rest"
	// palette entry, every vertex's JOINTS_0/WEIGHTS_0 left zero-filled so model_render.mesh's
	// existing weightSum<=0 fallback renders raw positions with no skinning).
	struct RockGenParams {
		std::uint32_t seed{0};
		float         baseRadius{1.0f};
		int           clusterCount{2};     // overlapping spheres unioned together, for an irregular silhouette
		float         noiseAmplitude{0.3f}; // fraction of baseRadius
		float         noiseFrequency{1.5f};
		int           sphereSegments{10}; // kept low -- a faceted look suits a rock better than a smooth sphere
		std::uint32_t materialIndex{0}; // caller's concern, same contract as BuildMeshlets' own
	};

	ModelAsset BuildProceduralRock(const RockGenParams& params);

} // namespace brassica
