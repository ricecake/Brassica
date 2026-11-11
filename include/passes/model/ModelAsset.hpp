#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "passes/model/GltfTypes.hpp"
#include "passes/model/ModelTypes.hpp"

namespace brassica {

	// Producer-agnostic model data (Stage 2): the one shape every producer -- GltfModelManager
	// (parses a real glTF/GLB file) or a hand-built procedural asset -- hands to
	// ModelRegistry::RegisterAsset(). Every field is model-local (0-based): meshlet
	// vertexOffset/triangleOffset index into this asset's own meshletVertices/
	// meshletTrianglesPacked, and clipTable's paletteBase values index into this asset's own
	// palette, exactly as GltfModel::BakeClips already produces them. ModelRegistry concatenates
	// these into shared buffers without rewriting any of it -- see ModelDescEntry's own comment
	// for why rebasing happens at the shader's lookup point instead.
	//
	// meshlet.materialIndex is the one field NOT model-local: it must already be a real,
	// resolved MaterialBuffer index (via MaterialManager::RegisterMaterial +
	// GltfModel::RemapMaterialIndices or equivalent) by the time the producer hands this over --
	// materials are a single shared catalog, not something ModelRegistry rebases.
	struct ModelAsset {
		std::vector<GltfVertex>    vertices;
		std::vector<std::uint32_t> meshletVertices;
		std::vector<std::uint32_t> meshletTrianglesPacked;
		std::vector<GltfMeshlet>   meshlets;

		// palette[0 .. jointCount) is the rest pose; each clipTable entry's paletteBase points
		// past that, same layout GltfModel::BakeClips already produces. A jointCount == 0 asset
		// (no skin) should still supply a single identity-matrix "rest" entry here, matching
		// BakeClips's own jointless-model behavior.
		std::vector<glm::mat4>     palette;
		std::vector<ModelClipEntry> clipTable;
		std::uint32_t               jointCount{0};

		// Model-local (pre instance-transform), the union of every rest-pose skinned vertex --
		// used for the rest-pose cull bounds (ModelInstance::clipIndex == kModelClipRestPose has
		// no clip-table entry of its own to read bounds from).
		glm::vec3 restBoundsCenter{0.0f};
		float     restBoundsRadius{0.0f};

		// Stage 4: joints in topological order (parentIndex < this joint's own index), for
		// ModelLiveEvalNode's live wind/seed-bend evaluation. Empty for a producer that has no
		// skeleton to offer (e.g. GltfModelManager's cow doesn't build one) -- ModelRegistry
		// gives such a model ModelDescEntry::skeletonBase == kModelNoSkeletonSentinel rather than
		// a meaningless zero-length real range, so a live-capable instance of a non-skeletal
		// model safely falls back to baked/rest instead of reading another model's joints.
		std::vector<ModelSkeletonJoint> skeleton;
	};

} // namespace brassica
