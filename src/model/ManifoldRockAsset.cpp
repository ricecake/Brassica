#include "model/ManifoldRockAsset.hpp"

#include <random>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/noise.hpp>
#include <manifold/manifold.h>
#include <spdlog/spdlog.h>

#include "model/MeshletBuilder.hpp"

namespace brassica {

	namespace {

		manifold::vec3 ToManifold(glm::vec3 v) { return manifold::vec3(v.x, v.y, v.z); }
		glm::vec3      FromManifold(manifold::vec3 v) {
			 return glm::vec3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
		}

	} // namespace

	ModelAsset BuildProceduralRock(const RockGenParams& params) {
		ModelAsset asset;
		asset.jointCount = 0;

		std::mt19937                          rng(params.seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		int clusterCount = std::max(1, params.clusterCount);

		std::vector<manifold::Manifold> spheres;
		spheres.reserve(clusterCount);
		for (int i = 0; i < clusterCount; ++i) {
			// First sphere centered at the origin (so a 1-cluster rock stays centered); extra
			// cluster spheres get a small random offset/radius to break up the perfect-sphere
			// silhouette before the noise displacement pass below adds the finer roughness.
			float     radius = params.baseRadius * (0.6f + 0.4f * unit(rng));
			glm::vec3 offset = (i == 0) ? glm::vec3(0.0f)
										 : glm::vec3(unit(rng) - 0.5f, unit(rng) - 0.5f, unit(rng) - 0.5f) * params.baseRadius * 0.5f;
			manifold::Manifold sphere = manifold::Manifold::Sphere(static_cast<double>(radius), params.sphereSegments);
			sphere = sphere.Translate(ToManifold(offset));
			spheres.push_back(sphere);
		}

		manifold::Manifold combined =
			(spheres.size() == 1) ? spheres[0] : manifold::Manifold::BatchBoolean(spheres, manifold::OpType::Add);
		if (combined.Status() != manifold::Manifold::Error::NoError) {
			spdlog::critical("BuildProceduralRock: boolean union failed with status {}", static_cast<int>(combined.Status()));
			return asset;
		}

		manifold::MeshGL mesh = combined.GetMeshGL();

		// Per-vertex Perlin displacement along each vertex's own direction from the origin --
		// cheap, smooth (unlike a per-vertex hash jitter, which would be discontinuous between
		// adjacent vertices), and the seed offset keeps different rocks from sharing the exact
		// same noise field.
		glm::vec3              seedOffset(static_cast<float>(params.seed % 1000u), static_cast<float>((params.seed / 1000u) % 1000u), 0.0f);
		std::vector<glm::vec3> positions(mesh.NumVert());
		for (std::uint32_t v = 0; v < mesh.NumVert(); ++v) {
			std::size_t base = static_cast<std::size_t>(v) * mesh.numProp;
			glm::vec3   pos(mesh.vertProperties[base + 0], mesh.vertProperties[base + 1], mesh.vertProperties[base + 2]);
			float       n = glm::perlin(pos * params.noiseFrequency + seedOffset);
			positions[v] = pos * (1.0f + params.noiseAmplitude * n);
		}

		std::vector<std::uint32_t> primIndices(mesh.triVerts.begin(), mesh.triVerts.end());

		// Manual face-normal averaging, not a Manifold round-trip -- rocks don't need the
		// skinned-mesh pipeline's joint attribution, and recomputing normals this way avoids
		// rebuilding a second Manifold just to call CalculateNormals again after displacement.
		std::vector<glm::vec3> normals(positions.size(), glm::vec3(0.0f));
		for (std::size_t t = 0; t + 2 < primIndices.size(); t += 3) {
			std::uint32_t i0 = primIndices[t], i1 = primIndices[t + 1], i2 = primIndices[t + 2];
			glm::vec3     faceNormal = glm::cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
			float         len = glm::length(faceNormal);
			if (len > 1e-8f) {
				faceNormal /= len;
			}
			normals[i0] += faceNormal;
			normals[i1] += faceNormal;
			normals[i2] += faceNormal;
		}

		std::vector<GltfVertex> primVertices(positions.size());
		for (std::size_t v = 0; v < positions.size(); ++v) {
			primVertices[v].position = positions[v];
			float len = glm::length(normals[v]);
			primVertices[v].normal = (len > 1e-6f) ? (normals[v] / len) : glm::vec3(0.0f, 1.0f, 0.0f);
			// joints/weights stay zero-filled (GltfVertex's own in-class defaults) -- no
			// skeleton, model_render.mesh's weightSum<=0 fallback uses raw position directly.
		}

		std::vector<std::uint8_t> triangleBytes;
		BuildMeshlets(primVertices, primIndices, 0u, asset.vertices, asset.meshletVertices, triangleBytes, asset.meshlets);

		std::size_t wordCount = (triangleBytes.size() + 3) / 4;
		asset.meshletTrianglesPacked.assign(wordCount, 0u);
		for (std::size_t i = 0; i < triangleBytes.size(); ++i) {
			asset.meshletTrianglesPacked[i >> 2] |= static_cast<std::uint32_t>(triangleBytes[i]) << ((i & 3u) * 8u);
		}

		// Jointless-model convention (see ModelAsset.hpp's own comment): a single identity rest
		// entry, no real skeleton.
		asset.palette.assign(1, glm::mat4(1.0f));

		glm::vec3 restMin(1e30f), restMax(-1e30f);
		for (const glm::vec3& pos : positions) {
			restMin = glm::min(restMin, pos);
			restMax = glm::max(restMax, pos);
		}
		asset.restBoundsCenter = (restMin + restMax) * 0.5f;
		asset.restBoundsRadius = glm::length(restMax - asset.restBoundsCenter);

		return asset;
	}

} // namespace brassica
