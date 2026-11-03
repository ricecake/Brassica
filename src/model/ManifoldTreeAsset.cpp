#include "model/ManifoldTreeAsset.hpp"

#include <array>
#include <cmath>
#include <unordered_map>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <manifold/manifold.h>
#include <spdlog/spdlog.h>

#include "model/MeshletBuilder.hpp"

namespace brassica {

	namespace {

		manifold::vec3 ToManifold(glm::vec3 v) { return manifold::vec3(v.x, v.y, v.z); }

		glm::vec3 FromManifold(manifold::vec3 v) {
			return glm::vec3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
		}

		// Maps local +Z (Manifold::Cylinder's own growth axis -- base at z=0, growing to
		// +Z, confirmed against external/manifold/test/manifold_test.cpp's own
		// CylinderZeroRadiusLow test) to worldDir, translated to worldStart. A single
		// straight-segment version of manifoldTestStart.cpp's parallel-transport frame --
		// Stage 5.0's "one simple shape" doesn't need a curved multi-point sweep.
		manifold::mat3x4 SegmentTransform(glm::vec3 worldStart, glm::vec3 worldDir) {
			glm::vec3 dir = glm::normalize(worldDir);
			glm::vec3 zAxis(0.0f, 0.0f, 1.0f);
			glm::vec3 rotAxis = glm::cross(zAxis, dir);
			float     axisLen = glm::length(rotAxis);

			glm::quat rot;
			if (axisLen < 1e-6f) {
				rot = (glm::dot(zAxis, dir) > 0.0f) ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
													 : glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			} else {
				rotAxis /= axisLen;
				float angle = std::acos(glm::clamp(glm::dot(zAxis, dir), -1.0f, 1.0f));
				rot = glm::angleAxis(angle, rotAxis);
			}

			glm::mat3 r = glm::mat3_cast(rot);
			return manifold::mat3x4(ToManifold(r[0]), ToManifold(r[1]), ToManifold(r[2]), ToManifold(worldStart));
		}

		// Rest-space (not local-to-parent) joint endpoints, same "absolute rest-space
		// authoring" lesson as ProceduralTreeAsset's own Stage 2 fix -- inverse-bind
		// skinning needs vertices positioned where they already look right at rest, since
		// skin = global*inverseBind reduces to identity there.
		struct TestJoint {
			glm::vec3     start;
			glm::vec3     end;
			float         radiusLow;
			float         radiusHigh;
			std::uint32_t parentIndex;
			float         rigidity;
		};

	} // namespace

	ModelAsset BuildManifoldTestTree() {
		ModelAsset asset;

		constexpr std::uint32_t kJointCount = 2;
		constexpr int           kCircularSegments = 12;

		// Joint 0: trunk, straight up from the origin. Joint 1: a branch angled off
		// partway up the trunk. Trunk stays rigid; the branch is the one Stage 4's
		// ModelLiveEvalNode should visibly bend under wind.
		const std::array<TestJoint, kJointCount> joints{
			TestJoint{glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 2.0f, 0.0f), 0.25f, 0.18f, kModelNoSkeletonSentinel, 0.9f},
			TestJoint{glm::vec3(0.0f, 1.3f, 0.0f), glm::vec3(0.9f, 2.3f, 0.0f), 0.14f, 0.05f, 0u, 0.15f},
		};

		std::vector<manifold::Manifold> segments;
		segments.reserve(kJointCount);
		for (const TestJoint& joint : joints) {
			double           height = static_cast<double>(glm::length(joint.end - joint.start));
			manifold::Manifold segment = manifold::Manifold::Cylinder(
				height, static_cast<double>(joint.radiusLow), static_cast<double>(joint.radiusHigh), kCircularSegments, false
			);
			segment = segment.Transform(SegmentTransform(joint.start, joint.end - joint.start));
			segment = segment.AsOriginal();
			segments.push_back(segment);
		}

		manifold::Manifold combined = manifold::Manifold::BatchBoolean(segments, manifold::OpType::Add);
		if (combined.Status() != manifold::Manifold::Error::NoError) {
			spdlog::critical(
				"BuildManifoldTestTree: boolean union failed with status {}", static_cast<int>(combined.Status())
			);
			return asset;
		}

		// Mesh-relation tracking, not interpolated properties (see this file's header
		// comment) -- exact, verified against external/manifold/test/manifold_test.cpp's
		// own Decompose test, which unions several AsOriginal()-tagged shapes and recovers
		// each one's triangles via runOriginalID afterward.
		std::unordered_map<std::uint32_t, std::uint32_t> idToJoint;
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			idToJoint[static_cast<std::uint32_t>(segments[j].OriginalID())] = j;
		}

		// CalculateNormals(0)/no-arg GetMeshGL() -- not CalculateNormals(3)/GetMeshGL(3) --
		// verified empirically against the real vendored library, not assumed: normalIdx is an
		// offset *past* the implicit 3 position channels, not an absolute property index, so
		// passing 3 to both put real normals at channels 6-8 while leaving 3-5 as unused zero
		// padding (read as a degenerate all-zero normal -- the real cause of a hardware-tested
		// "pitch black" tree, found via sgreenhusted's real-hardware test). The library's own
		// doc comment also flags the explicit-idx form as legacy, to be removed in a future
		// release -- this is the recommended form, not just the one that happens to work.
		combined = combined.CalculateNormals(0);
		manifold::MeshGL mesh = combined.GetMeshGL();

		std::vector<std::uint32_t> vertexJoint(mesh.NumVert(), 0u);
		for (std::size_t run = 0; run + 1 < mesh.runIndex.size(); ++run) {
			auto          it = idToJoint.find(mesh.runOriginalID[run]);
			std::uint32_t jointIdx = (it != idToJoint.end()) ? it->second : 0u;
			for (std::uint32_t t = mesh.runIndex[run]; t < mesh.runIndex[run + 1]; ++t) {
				// Shared seam vertices between two runs resolve by last-write-wins -- an
				// acceptable simplification matching ProceduralTreeAsset's own single-
				// rigid-weight convention (no cross-joint blending at segment boundaries
				// there either).
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
		BuildMeshlets(primVertices, primIndices, 0u, asset.vertices, asset.meshletVertices, triangleBytes, asset.meshlets);

		std::size_t wordCount = (triangleBytes.size() + 3) / 4;
		asset.meshletTrianglesPacked.assign(wordCount, 0u);
		for (std::size_t i = 0; i < triangleBytes.size(); ++i) {
			asset.meshletTrianglesPacked[i >> 2] |= static_cast<std::uint32_t>(triangleBytes[i]) << ((i & 3u) * 8u);
		}

		// Skeleton: same TRS-chain composition as ProceduralTreeAsset::BuildProceduralTreeAsset,
		// just with the branch's rest rotation expressing its own angled direction (identity
		// rotation alone wouldn't put joint1's local +Y where the branch geometry actually is --
		// GltfModel::EvaluatePose/ProceduralTreeAsset's convention is T*R*S per joint, so the
		// branch's authored direction is baked into restRotation here instead of restTranslation
		// alone pointing somewhere a zero-rotation child wouldn't follow).
		asset.jointCount = kJointCount;
		asset.skeleton.resize(kJointCount);
		std::vector<glm::mat4> restGlobals(kJointCount);
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			const TestJoint& joint = joints[j];
			glm::vec3        parentOrigin = (joint.parentIndex == kModelNoSkeletonSentinel) ? glm::vec3(0.0f) : joints[joint.parentIndex].start;
			glm::vec3        localTranslation = joint.start - parentOrigin;

			glm::vec3 dir = glm::normalize(joint.end - joint.start);
			glm::vec3 zAxis(0.0f, 1.0f, 0.0f); // rest up-axis convention matches ProceduralTreeAsset's own (+Y)
			glm::quat localRotation;
			glm::vec3 rotAxis = glm::cross(zAxis, dir);
			float     axisLen = glm::length(rotAxis);
			if (axisLen < 1e-6f) {
				localRotation = (glm::dot(zAxis, dir) > 0.0f) ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
															   : glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			} else {
				rotAxis /= axisLen;
				float angle = std::acos(glm::clamp(glm::dot(zAxis, dir), -1.0f, 1.0f));
				localRotation = glm::angleAxis(angle, rotAxis);
			}

			glm::mat4 parentGlobal = (joint.parentIndex == kModelNoSkeletonSentinel) ? glm::mat4(1.0f) : restGlobals[joint.parentIndex];
			glm::mat4 local = glm::translate(glm::mat4(1.0f), localTranslation) * glm::mat4_cast(localRotation);
			restGlobals[j] = parentGlobal * local;

			ModelSkeletonJoint& sj = asset.skeleton[j];
			sj.inverseBind = glm::inverse(restGlobals[j]);
			sj.restTranslation = localTranslation;
			sj.restScale = 1.0f;
			sj.restRotation = glm::vec4(localRotation.x, localRotation.y, localRotation.z, localRotation.w);
			sj.restUpAxis = glm::vec3(0.0f, 1.0f, 0.0f);
			sj.rigidity = joint.rigidity;
			sj.parentIndex = joint.parentIndex;
		}

		// Rest pose = skinMatrices(restGlobals), same layout GltfModel::BakeClips/
		// ProceduralTreeAsset already produce (palette[0 .. jointCount) is the rest pose).
		asset.palette.resize(kJointCount);
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			asset.palette[j] = restGlobals[j] * asset.skeleton[j].inverseBind;
		}
		// No clips -- this prototype's whole point is live wind eval (Stage 4), not baked
		// clip playback; instances stay at kModelClipRestPose with the live-capable flag set.

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
