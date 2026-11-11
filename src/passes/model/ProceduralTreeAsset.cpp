#include "passes/model/ProceduralTreeAsset.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace brassica {

	namespace {

		constexpr float kPi = 3.14159265358979323846f;
		constexpr int   kSides = 8;

		// One rigidly-skinned (single joint, weight 1) cylinder segment. Vertex positions are
		// authored in ABSOLUTE rest-space (offset by the joint's own rest-world origin), not in
		// the joint's own local frame: standard inverse-bind skinning (globalPose * invBind *
		// localPos) reduces to the identity at exactly the rest pose, so a vertex's "local"
		// position IS its rest-space world position -- a per-joint-local y=[0,height] authoring
		// would collapse every segment onto the same local range instead of stacking them (caught
		// via tests/test_model_registry.cpp's bounds check reporting an implausibly small
		// restBoundsRadius). Sides + two capped rings; CCW winding viewed from outside each face,
		// matching this engine's eCounterClockwise front-face convention
		// (src/render/PipelineLibrary.cpp).
		void AppendCylinderSegment(
			float                       height,
			float                       radius,
			std::uint32_t               jointIndex,
			glm::vec3                   restWorldOrigin,
			std::vector<GltfVertex>&    vertices,
			std::vector<std::uint32_t>& trianglesLocal
		) {
			std::uint32_t base = static_cast<std::uint32_t>(vertices.size());

			auto push = [&](glm::vec3 pos, glm::vec3 normal, glm::vec2 uv) {
				GltfVertex v{};
				v.position = pos + restWorldOrigin;
				v.normal = normal;
				v.uv = uv;
				v.joints = glm::uvec4(jointIndex, 0u, 0u, 0u);
				v.weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
				vertices.push_back(v);
			};

			std::uint32_t bottomRing = base;
			for (int i = 0; i < kSides; ++i) {
				float     angle = (2.0f * kPi * static_cast<float>(i)) / kSides;
				glm::vec3 dir(std::cos(angle), 0.0f, std::sin(angle));
				push(dir * radius, dir, glm::vec2(static_cast<float>(i) / kSides, 0.0f));
			}
			std::uint32_t topRing = static_cast<std::uint32_t>(vertices.size());
			for (int i = 0; i < kSides; ++i) {
				float     angle = (2.0f * kPi * static_cast<float>(i)) / kSides;
				glm::vec3 dir(std::cos(angle), 0.0f, std::sin(angle));
				push(dir * radius + glm::vec3(0.0f, height, 0.0f), dir, glm::vec2(static_cast<float>(i) / kSides, 1.0f));
			}
			for (int i = 0; i < kSides; ++i) {
				int           ni = (i + 1) % kSides;
				std::uint32_t b0 = bottomRing + i, b1 = bottomRing + ni;
				std::uint32_t t0 = topRing + i, t1 = topRing + ni;
				trianglesLocal.insert(trianglesLocal.end(), {b0, t0, t1});
				trianglesLocal.insert(trianglesLocal.end(), {b0, t1, b1});
			}

			std::uint32_t bottomCenter = static_cast<std::uint32_t>(vertices.size());
			push(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.5f));
			std::uint32_t bottomCap = static_cast<std::uint32_t>(vertices.size());
			for (int i = 0; i < kSides; ++i) {
				float     angle = (2.0f * kPi * static_cast<float>(i)) / kSides;
				glm::vec3 dir(std::cos(angle), 0.0f, std::sin(angle));
				push(dir * radius, glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.5f) + 0.5f * glm::vec2(dir.x, dir.z));
			}
			for (int i = 0; i < kSides; ++i) {
				int ni = (i + 1) % kSides;
				trianglesLocal.insert(trianglesLocal.end(), {bottomCenter, bottomCap + ni, bottomCap + i});
			}

			std::uint32_t topCenter = static_cast<std::uint32_t>(vertices.size());
			push(glm::vec3(0.0f, height, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.5f));
			std::uint32_t topCap = static_cast<std::uint32_t>(vertices.size());
			for (int i = 0; i < kSides; ++i) {
				float     angle = (2.0f * kPi * static_cast<float>(i)) / kSides;
				glm::vec3 dir(std::cos(angle), 0.0f, std::sin(angle));
				push(
					dir * radius + glm::vec3(0.0f, height, 0.0f),
					glm::vec3(0.0f, 1.0f, 0.0f),
					glm::vec2(0.5f) + 0.5f * glm::vec2(dir.x, dir.z)
				);
			}
			for (int i = 0; i < kSides; ++i) {
				int ni = (i + 1) % kSides;
				trianglesLocal.insert(trianglesLocal.end(), {topCenter, topCap + i, topCap + ni});
			}
		}

		// One cylinder segment's worth of vertices/triangles, packed as this asset's sole
		// meshlet -- small enough (34 verts / 32 tris per segment) to never need meshoptimizer's
		// real clustering, unlike GltfModel's arbitrary-size path.
		void AppendSegmentMeshlet(
			float                        height,
			float                        radius,
			std::uint32_t                jointIndex,
			glm::vec3                    restWorldOrigin,
			std::vector<GltfVertex>&     vertices,
			std::vector<std::uint32_t>&  meshletVertices,
			std::vector<std::uint8_t>&   triangleBytes,
			std::vector<GltfMeshlet>&    meshlets
		) {
			std::vector<GltfVertex>    segVertices;
			std::vector<std::uint32_t> segTrianglesLocal;
			AppendCylinderSegment(height, radius, jointIndex, restWorldOrigin, segVertices, segTrianglesLocal);

			GltfMeshlet meshlet{};
			meshlet.vertexOffset = static_cast<std::uint32_t>(meshletVertices.size());
			meshlet.triangleOffset = static_cast<std::uint32_t>(triangleBytes.size() / 3);
			meshlet.vertexCount = static_cast<std::uint32_t>(segVertices.size());
			meshlet.triangleCount = static_cast<std::uint32_t>(segTrianglesLocal.size() / 3);
			meshlet.materialIndex = 0; // the shared MaterialManager's always-present default material
			glm::vec3 lo(1e30f), hi(-1e30f);
			for (const GltfVertex& v : segVertices) {
				lo = glm::min(lo, v.position);
				hi = glm::max(hi, v.position);
			}
			meshlet.boundsCenter = (lo + hi) * 0.5f;
			meshlet.boundsRadius = glm::length(hi - meshlet.boundsCenter);
			meshlet.coneAxis = glm::vec3(0.0f, 1.0f, 0.0f);
			meshlet.coneCutoff = 1.0f; // no real cone culling for this test asset -- never rejects

			std::uint32_t vertexBase = static_cast<std::uint32_t>(vertices.size());
			vertices.insert(vertices.end(), segVertices.begin(), segVertices.end());
			for (std::uint32_t i = 0; i < segVertices.size(); ++i) {
				meshletVertices.push_back(vertexBase + i); // gathered 1:1, same as GltfModel's own per-meshlet gather
			}
			for (std::uint32_t localIndex : segTrianglesLocal) {
				triangleBytes.push_back(static_cast<std::uint8_t>(localIndex));
			}

			meshlets.push_back(meshlet);
		}

	} // namespace

	ModelAsset BuildProceduralTreeAsset() {
		ModelAsset asset;

		constexpr std::uint32_t kJointCount = 3;
		constexpr float         kSegmentHeights[kJointCount] = {1.5f, 1.0f, 0.8f};
		constexpr float         kSegmentRadii[kJointCount] = {0.3f, 0.2f, 0.12f};

		asset.jointCount = kJointCount;

		// Rest translations: joint j sits at the TOP of segment j-1 (joint0 is the root, at the
		// world/model origin). TRS composition matches GltfModel::EvaluatePose exactly
		// (T * R * S per joint, global = parentGlobal * local) so this hierarchy behaves exactly
		// like a real glTF skin's would.
		glm::vec3 restTranslation[kJointCount] = {
			glm::vec3(0.0f),
			glm::vec3(0.0f, kSegmentHeights[0], 0.0f),
			glm::vec3(0.0f, kSegmentHeights[1], 0.0f),
		};

		auto computeGlobals = [&](const glm::quat* localRotations, std::vector<glm::mat4>& outGlobals) {
			outGlobals.resize(kJointCount);
			glm::mat4 parent(1.0f);
			for (std::uint32_t j = 0; j < kJointCount; ++j) {
				glm::mat4 local = glm::translate(glm::mat4(1.0f), restTranslation[j]) * glm::mat4_cast(localRotations[j]);
				outGlobals[j] = parent * local;
				parent = outGlobals[j];
			}
		};

		glm::quat restRotations[kJointCount] = {glm::quat(1, 0, 0, 0), glm::quat(1, 0, 0, 0), glm::quat(1, 0, 0, 0)};
		std::vector<glm::mat4> restGlobals;
		computeGlobals(restRotations, restGlobals);
		std::vector<glm::mat4> inverseBind(kJointCount);
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			inverseBind[j] = glm::inverse(restGlobals[j]);
		}

		// Stage 4: skeleton for ModelLiveEvalNode's live wind/seed-bend. restUpAxis is +Y for
		// every joint -- each cylinder segment's own local frame authors "up" as +Y (see
		// AppendCylinderSegment), and bending happens around an axis derived from that, same
		// convention the existing "Sway" clip's own rotation axis (Z) implicitly assumes.
		// rigidity tapers from the rigid trunk base to the flexible tip, matching how the
		// existing "Sway" clip already bends joint1/joint2 more than joint0 (which stays fixed
		// there entirely) -- a real branch's own taper, not an arbitrary choice.
		constexpr float kJointRigidity[kJointCount] = {0.9f, 0.4f, 0.1f};
		asset.skeleton.resize(kJointCount);
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			ModelSkeletonJoint& joint = asset.skeleton[j];
			joint.inverseBind = inverseBind[j];
			joint.restTranslation = restTranslation[j];
			joint.restScale = 1.0f;
			joint.restRotation = glm::vec4(restRotations[j].x, restRotations[j].y, restRotations[j].z, restRotations[j].w);
			joint.restUpAxis = glm::vec3(0.0f, 1.0f, 0.0f);
			joint.rigidity = kJointRigidity[j];
			joint.parentIndex = (j == 0) ? kModelNoSkeletonSentinel : (j - 1);
		}

		// Geometry is authored in absolute rest-space (see AppendCylinderSegment's own comment),
		// so each segment's vertices need the offset to its OWN joint's rest-world origin -- the
		// translation column of that joint's restGlobals, which is exactly "the cumulative height
		// of every segment below it."
		std::vector<std::uint8_t> triangleBytes;
		for (std::uint32_t j = 0; j < kJointCount; ++j) {
			glm::vec3 restWorldOrigin = glm::vec3(restGlobals[j][3]);
			AppendSegmentMeshlet(
				kSegmentHeights[j],
				kSegmentRadii[j],
				j,
				restWorldOrigin,
				asset.vertices,
				asset.meshletVertices,
				triangleBytes,
				asset.meshlets
			);
		}
		std::size_t wordCount = (triangleBytes.size() + 3) / 4;
		asset.meshletTrianglesPacked.assign(wordCount, 0u);
		for (std::size_t i = 0; i < triangleBytes.size(); ++i) {
			asset.meshletTrianglesPacked[i >> 2] |= static_cast<std::uint32_t>(triangleBytes[i]) << ((i & 3u) * 8u);
		}

		auto skinMatrices = [&](const std::vector<glm::mat4>& globals) {
			std::vector<glm::mat4> out(kJointCount);
			for (std::uint32_t j = 0; j < kJointCount; ++j) {
				out[j] = globals[j] * inverseBind[j];
			}
			return out;
		};

		// Rest pose always occupies palette[0 .. jointCount), matching GltfModel::BakeClips's
		// own layout.
		asset.palette = skinMatrices(restGlobals);

		// One "Sway" clip: joint0 (the trunk base) stays fixed, joint1/joint2 bend back and
		// forth -- see this file's own header comment for why this is deliberately not Stage 4's
		// real wind model, just enough motion to prove per-model clip tables/palettes work.
		constexpr float sampleRateHz = 120.0f;
		constexpr float periodSeconds = 2.0f;
		constexpr float frameDt = 1.0f / sampleRateHz;
		int             frameCount = static_cast<int>(periodSeconds / frameDt) + 2;

		ModelClipEntry clip{};
		clip.paletteBase = static_cast<std::uint32_t>(asset.palette.size());
		clip.durationMs = static_cast<std::uint32_t>(periodSeconds * 1000.0f);
		clip.frameDtMs = frameDt * 1000.0f;

		glm::vec3 boundsMin(1e30f), boundsMax(-1e30f);
		for (int f = 0; f < frameCount; ++f) {
			float t = std::min(static_cast<float>(f) * frameDt, periodSeconds);
			float phase = (2.0f * kPi * t) / periodSeconds;

			glm::quat rot[kJointCount] = {
				glm::quat(1, 0, 0, 0),
				glm::angleAxis(glm::radians(12.0f) * std::sin(phase), glm::vec3(0.0f, 0.0f, 1.0f)),
				glm::angleAxis(glm::radians(20.0f) * std::sin(phase + 0.6f), glm::vec3(0.0f, 0.0f, 1.0f)),
			};
			std::vector<glm::mat4> globals;
			computeGlobals(rot, globals);
			std::vector<glm::mat4> frameSkin = skinMatrices(globals);
			asset.palette.insert(asset.palette.end(), frameSkin.begin(), frameSkin.end());

			for (const GltfVertex& v : asset.vertices) {
				glm::mat4 skin = frameSkin[v.joints.x]; // every vertex here is rigidly weight-1 to one joint
				glm::vec3 pos = glm::vec3(skin * glm::vec4(v.position, 1.0f));
				boundsMin = glm::min(boundsMin, pos);
				boundsMax = glm::max(boundsMax, pos);
			}
		}
		clip.frameCount = static_cast<std::uint32_t>(frameCount);
		clip.boundsCenter = (boundsMin + boundsMax) * 0.5f;
		clip.boundsRadius = glm::length(boundsMax - clip.boundsCenter);
		asset.clipTable.push_back(clip);

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
