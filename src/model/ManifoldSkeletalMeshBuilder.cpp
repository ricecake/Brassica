#include "model/ManifoldSkeletalMeshBuilder.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <manifold/manifold.h>
#include <spdlog/spdlog.h>

#include "model/MeshletBuilder.hpp"

namespace brassica {

	namespace {

		manifold::vec3 ToManifold(glm::vec3 v) { return manifold::vec3(v.x, v.y, v.z); }

		// Shared by both SegmentTransform (geometry placement) and the skeleton's own rest
		// rotation below -- maps local axis `from` to world direction `to`.
		glm::quat RotationBetween(glm::vec3 from, glm::vec3 to) {
			glm::vec3 dir = glm::normalize(to);
			glm::vec3 rotAxis = glm::cross(from, dir);
			float     axisLen = glm::length(rotAxis);
			if (axisLen < 1e-6f) {
				return (glm::dot(from, dir) > 0.0f) ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
													 : glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			}
			rotAxis /= axisLen;
			float angle = std::acos(glm::clamp(glm::dot(from, dir), -1.0f, 1.0f));
			return glm::angleAxis(angle, rotAxis);
		}

		// Maps local +Z (Manifold::Cylinder's own growth axis -- base at z=0, growing to +Z,
		// confirmed against external/manifold/test/manifold_test.cpp's own
		// CylinderZeroRadiusLow test) to worldDir, translated to worldStart. A single
		// straight-segment version of manifoldTestStart.cpp's parallel-transport frame -- each
		// joint here is one rigid tapered segment, not a curved multi-point sweep.
		manifold::mat3x4 SegmentTransform(glm::vec3 worldStart, glm::vec3 worldDir) {
			glm::quat rot = RotationBetween(glm::vec3(0.0f, 0.0f, 1.0f), worldDir);
			glm::mat3 r = glm::mat3_cast(rot);
			return manifold::mat3x4(ToManifold(r[0]), ToManifold(r[1]), ToManifold(r[2]), ToManifold(worldStart));
		}

	} // namespace

	ModelAsset BuildSkinnedManifoldAsset(
		std::span<const ManifoldJointSpec> joints, int circularSegments, std::uint32_t materialIndex
	) {
		ModelAsset asset;
		auto       jointCount = static_cast<std::uint32_t>(joints.size());
		asset.jointCount = jointCount;
		if (jointCount == 0) {
			return asset;
		}

		std::vector<manifold::Manifold> segments;
		segments.reserve(jointCount);
		for (const ManifoldJointSpec& joint : joints) {
			double              height = static_cast<double>(glm::length(joint.end - joint.start));
			manifold::Manifold segment = manifold::Manifold::Cylinder(
				height, static_cast<double>(joint.radiusLow), static_cast<double>(joint.radiusHigh), circularSegments, false
			);
			segment = segment.Transform(SegmentTransform(joint.start, joint.end - joint.start));
			segment = segment.AsOriginal();
			segments.push_back(segment);
		}

		manifold::Manifold combined = manifold::Manifold::BatchBoolean(segments, manifold::OpType::Add);
		if (combined.Status() != manifold::Manifold::Error::NoError) {
			spdlog::critical(
				"BuildSkinnedManifoldAsset: boolean union failed with status {}", static_cast<int>(combined.Status())
			);
			return asset;
		}

		// Mesh-relation tracking, not interpolated properties -- exact, verified against
		// external/manifold/test/manifold_test.cpp's own Decompose test, which unions several
		// AsOriginal()-tagged shapes and recovers each one's triangles via runOriginalID
		// afterward.
		std::unordered_map<std::uint32_t, std::uint32_t> idToJoint;
		for (std::uint32_t j = 0; j < jointCount; ++j) {
			idToJoint[static_cast<std::uint32_t>(segments[j].OriginalID())] = j;
		}

		// CalculateNormals(0)/no-arg GetMeshGL() -- not CalculateNormals(3)/GetMeshGL(3) --
		// verified empirically against the real vendored library, not assumed: normalIdx is an
		// offset *past* the implicit 3 position channels, not an absolute property index. The
		// library's own doc comment also flags the explicit-idx form as legacy, to be removed in
		// a future release -- this is the recommended form, not just the one that happens to work
		// (a real hardware-tested "pitch black" bug before this fix).
		combined = combined.CalculateNormals(0);
		manifold::MeshGL mesh = combined.GetMeshGL();

		std::vector<std::uint32_t> vertexJoint(mesh.NumVert(), 0u);
		for (std::size_t run = 0; run + 1 < mesh.runIndex.size(); ++run) {
			auto          it = idToJoint.find(mesh.runOriginalID[run]);
			std::uint32_t jointIdx = (it != idToJoint.end()) ? it->second : 0u;
			for (std::uint32_t t = mesh.runIndex[run]; t < mesh.runIndex[run + 1]; ++t) {
				// Shared seam vertices between two runs resolve by last-write-wins -- an
				// acceptable simplification matching ProceduralTreeAsset's own single-rigid-
				// weight convention (no cross-joint blending at segment boundaries there either).
				vertexJoint[mesh.triVerts[t]] = jointIdx;
			}
		}

		std::vector<GltfVertex> primVertices(mesh.NumVert());
		for (std::uint32_t v = 0; v < mesh.NumVert(); ++v) {
			std::size_t base = static_cast<std::size_t>(v) * mesh.numProp;
			primVertices[v].position = glm::vec3(mesh.vertProperties[base + 0], mesh.vertProperties[base + 1], mesh.vertProperties[base + 2]);
			primVertices[v].normal = glm::vec3(mesh.vertProperties[base + 3], mesh.vertProperties[base + 4], mesh.vertProperties[base + 5]);
			primVertices[v].joints = glm::uvec4(vertexJoint[v], 0u, 0u, 0u);
			primVertices[v].weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
		}
		std::vector<std::uint32_t> primIndices(mesh.triVerts.begin(), mesh.triVerts.end());

		std::vector<std::uint8_t> triangleBytes;
		BuildMeshlets(primVertices, primIndices, materialIndex, asset.vertices, asset.meshletVertices, triangleBytes, asset.meshlets);

		std::size_t wordCount = (triangleBytes.size() + 3) / 4;
		asset.meshletTrianglesPacked.assign(wordCount, 0u);
		for (std::size_t i = 0; i < triangleBytes.size(); ++i) {
			asset.meshletTrianglesPacked[i >> 2] |= static_cast<std::uint32_t>(triangleBytes[i]) << ((i & 3u) * 8u);
		}

		// Skeleton: same TRS-chain composition as ProceduralTreeAsset::BuildProceduralTreeAsset --
		// the branch's own authored direction is baked into restRotation (T*R*S per joint), not
		// restTranslation alone, since a zero-rotation child wouldn't otherwise follow it.
		asset.skeleton.resize(jointCount);
		std::vector<glm::mat4> restGlobals(jointCount);
		for (std::uint32_t j = 0; j < jointCount; ++j) {
			const ManifoldJointSpec& joint = joints[j];
			glm::vec3                parentOrigin =
				(joint.parentIndex == kModelNoSkeletonSentinel) ? glm::vec3(0.0f) : joints[joint.parentIndex].start;
			glm::vec3 localTranslation = joint.start - parentOrigin;
			glm::quat localRotation = RotationBetween(glm::vec3(0.0f, 1.0f, 0.0f), joint.end - joint.start);

			glm::mat4 parentGlobal = (joint.parentIndex == kModelNoSkeletonSentinel) ? glm::mat4(1.0f) : restGlobals[joint.parentIndex];
			glm::mat4 local = glm::translate(glm::mat4(1.0f), localTranslation) * glm::mat4_cast(localRotation);
			restGlobals[j] = parentGlobal * local;

			ModelSkeletonJoint& sj = asset.skeleton[j];
			sj.inverseBind = glm::inverse(restGlobals[j]);
			sj.restTranslation = localTranslation;
			sj.restScale = 1.0f;
			sj.restRotation = glm::vec4(localRotation.x, localRotation.y, localRotation.z, localRotation.w);
			sj.restUpAxis = glm::vec3(0.0f, 1.0f, 0.0f); // matches ProceduralTreeAsset's own convention
			sj.rigidity = joint.rigidity;
			sj.parentIndex = joint.parentIndex;
		}

		// Rest pose = skinMatrices(restGlobals), same layout GltfModel::BakeClips/
		// ProceduralTreeAsset already produce (palette[0 .. jointCount) is the rest pose).
		asset.palette.resize(jointCount);
		for (std::uint32_t j = 0; j < jointCount; ++j) {
			asset.palette[j] = restGlobals[j] * asset.skeleton[j].inverseBind;
		}
		// No clips -- live wind eval (Stage 4) is the animation source for these, not baked
		// clip playback.

		glm::vec3 restMin(1e30f), restMax(-1e30f);
		for (const GltfVertex& v : asset.vertices) {
			glm::mat4 skin = asset.palette[v.joints.x];
			glm::vec3 pos = glm::vec3(skin * glm::vec4(v.position, 1.0f));
			restMin = glm::min(restMin, pos);
			restMax = glm::max(restMax, pos);
		}
		asset.restBoundsCenter = (restMin + restMax) * 0.5f;
		asset.restBoundsRadius = glm::length(restMax - asset.restBoundsCenter);

		return asset;
	}

} // namespace brassica
