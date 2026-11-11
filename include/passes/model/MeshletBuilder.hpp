#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "passes/model/GltfTypes.hpp"

namespace brassica {

	// Shared meshopt_buildMeshlets/meshopt_optimizeMeshlet/meshopt_computeMeshletBounds sequence,
	// extracted from GltfModel.cpp's own primitive-loading code (Stage 5.0 of the GPU-driven
	// model-instance plan) so a non-glTF producer (a Manifold-built procedural mesh) can build
	// real meshlets the same way instead of reimplementing this. Appends into the model-wide
	// outVertices/outMeshletVertices/triangleBytes/outMeshlets arrays, rebasing every index/
	// offset so they address those shared arrays rather than primVertices/primIndices' own
	// local-to-itself ones -- same contract GltfModel's own call site already relies on.
	// materialIndex is the caller's concern (glTF derives it from the primitive; a procedural
	// generator just picks one) since Manifold-built meshes have no fastgltf::Primitive to read
	// it from.
	void BuildMeshlets(
		std::span<const GltfVertex>    primVertices,
		std::span<const std::uint32_t> primIndices,
		std::uint32_t                  materialIndex,
		std::vector<GltfVertex>&       outVertices,
		std::vector<std::uint32_t>&    outMeshletVertices,
		std::vector<std::uint8_t>&     triangleBytes,
		std::vector<GltfMeshlet>&      outMeshlets
	);

} // namespace brassica
