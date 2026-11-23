#include "passes/model/MeshletBuilder.hpp"

#include "meshoptimizer.h"

namespace brassica {

	namespace {
		constexpr std::size_t kMaxMeshletVerts = 128;
		constexpr std::size_t kMaxMeshletTris = 128;
	} // namespace

	void BuildMeshlets(
		std::span<const GltfVertex>    primVertices,
		std::span<const std::uint32_t> primIndices,
		std::uint32_t                  materialIndex,
		std::vector<GltfVertex>&       outVertices,
		std::vector<std::uint32_t>&    outMeshletVertices,
		std::vector<std::uint8_t>&     triangleBytes,
		std::vector<GltfMeshlet>&      outMeshlets
	) {
		if (primVertices.empty() || primIndices.empty()) {
			return;
		}

		std::size_t                  maxMeshlets = meshopt_buildMeshletsBound(primIndices.size(), kMaxMeshletVerts, kMaxMeshletTris);
		std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
		std::vector<unsigned int>    rawMeshletVertices(maxMeshlets * kMaxMeshletVerts);
		std::vector<unsigned char>   rawMeshletTriangles(maxMeshlets * kMaxMeshletTris * 3);

		std::size_t meshletCount = meshopt_buildMeshlets(
			rawMeshlets.data(),
			rawMeshletVertices.data(),
			rawMeshletTriangles.data(),
			primIndices.data(),
			primIndices.size(),
			&primVertices[0].position.x,
			primVertices.size(),
			sizeof(GltfVertex),
			kMaxMeshletVerts,
			kMaxMeshletTris,
			0.0f
		);
		rawMeshlets.resize(meshletCount);
		if (meshletCount > 0) {
			const meshopt_Meshlet& last = rawMeshlets.back();
			rawMeshletVertices.resize(last.vertex_offset + last.vertex_count);
			rawMeshletTriangles.resize(last.triangle_offset + last.triangle_count * 3);
		} else {
			rawMeshletVertices.clear();
			rawMeshletTriangles.clear();
		}

		// Recommended by meshoptimizer's own meshopt_buildMeshlets doc comment: reorders each
		// meshlet's own vertex/triangle ranges in place for better locality, same arrays/offsets,
		// no rebasing implications for the code below.
		for (const meshopt_Meshlet& m : rawMeshlets) {
			meshopt_optimizeMeshlet(&rawMeshletVertices[m.vertex_offset], &rawMeshletTriangles[m.triangle_offset], m.triangle_count, m.vertex_count);
		}

		const auto vertexBase = static_cast<std::uint32_t>(outVertices.size());
		const auto meshletVertexBase = static_cast<std::uint32_t>(outMeshletVertices.size());
		const auto triangleByteBase = static_cast<std::uint32_t>(triangleBytes.size());

		outVertices.insert(outVertices.end(), primVertices.begin(), primVertices.end());

		outMeshletVertices.reserve(outMeshletVertices.size() + rawMeshletVertices.size());
		for (unsigned int localVertexIndex : rawMeshletVertices) {
			outMeshletVertices.push_back(vertexBase + localVertexIndex);
		}

		triangleBytes.insert(triangleBytes.end(), rawMeshletTriangles.begin(), rawMeshletTriangles.end());

		outMeshlets.reserve(outMeshlets.size() + rawMeshlets.size());
		for (const meshopt_Meshlet& m : rawMeshlets) {
			meshopt_Bounds bounds = meshopt_computeMeshletBounds(
				&rawMeshletVertices[m.vertex_offset],
				&rawMeshletTriangles[m.triangle_offset],
				m.triangle_count,
				&primVertices[0].position.x,
				primVertices.size(),
				sizeof(GltfVertex)
			);

			GltfMeshlet gpuMeshlet{};
			gpuMeshlet.vertexOffset = meshletVertexBase + m.vertex_offset;
			// meshopt's own triangle_offset addresses its flat uint8 micro-index buffer directly
			// (a byte/corner offset); GltfMeshlet stores triangleOffset in *triangle* units
			// instead (see shaders/gltf_mesh.glsl's GltfMeshletTriangleRef comment: byteIndex =
			// meshlet.triangleOffset*3 + ...), so divide by 3 here. Always exact: meshopt lays
			// out each meshlet's byte range as count*3, a whole multiple of 3.
			gpuMeshlet.triangleOffset = (triangleByteBase + m.triangle_offset) / 3u;
			gpuMeshlet.vertexCount = m.vertex_count;
			gpuMeshlet.triangleCount = m.triangle_count;
			gpuMeshlet.boundsCenter = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
			gpuMeshlet.boundsRadius = bounds.radius;
			gpuMeshlet.coneAxis = glm::vec3(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]);
			gpuMeshlet.coneCutoff = bounds.cone_cutoff;
			gpuMeshlet.materialIndex = materialIndex;
			outMeshlets.push_back(gpuMeshlet);
		}
	}

} // namespace brassica
