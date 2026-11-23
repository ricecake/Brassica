#include "passes/model/ManifoldPalmAsset.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <manifold/manifold.h>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"

namespace brassica {

	namespace {

		glm::quat AlignZTo(glm::vec3 dir) {
			dir = glm::normalize(dir);
			constexpr glm::vec3 z(0.0f, 0.0f, 1.0f);
			float                d = glm::clamp(glm::dot(z, dir), -1.0f, 1.0f);
			if (d > 0.9999f) {
				return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
			}
			if (d < -0.9999f) {
				return glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			}
			glm::vec3 axis = glm::normalize(glm::cross(z, dir));
			return glm::angleAxis(std::acos(d), axis);
		}

		// One closed, thin diamond-lens segment -- same "two apexes over an outline ring"
		// construction ManifoldLeafAsset.cpp's own AppendLeafBlade uses, duplicated rather than
		// shared because a frond chains several of these end-to-end along a curving path (one
		// call per segment, each with its own length/width/origin/orientation) instead of a
		// single straight blade from a shared base point.
		void AppendFrondSegment(
			float length, float width, glm::vec3 origin, glm::quat orientation, std::vector<float>& vertProperties,
			std::vector<std::uint32_t>& triVerts
		) {
			float thickness = std::min(length, width) * 0.15f;
			float midZ = length * 0.5f;

			glm::vec3 localVerts[6] = {
				glm::vec3(0.0f, thickness * 0.5f, length * 0.42f),  // apex top
				glm::vec3(0.0f, -thickness * 0.5f, length * 0.42f), // apex bottom
				glm::vec3(0.0f, 0.0f, 0.0f),                        // base
				glm::vec3(-width * 0.5f, 0.0f, midZ),               // wide left
				glm::vec3(0.0f, 0.0f, length),                      // tip
				glm::vec3(width * 0.5f, 0.0f, midZ),                // wide right
			};

			auto base = static_cast<std::uint32_t>(vertProperties.size() / 3);
			for (glm::vec3 v : localVerts) {
				glm::vec3 world = orientation * v + origin;
				vertProperties.push_back(world.x);
				vertProperties.push_back(world.y);
				vertProperties.push_back(world.z);
			}

			std::uint32_t apexTop = base + 0;
			std::uint32_t apexBottom = base + 1;
			std::uint32_t ring[4] = {base + 2, base + 3, base + 4, base + 5};
			for (int i = 0; i < 4; ++i) {
				std::uint32_t a = ring[i];
				std::uint32_t b = ring[(i + 1) % 4];
				triVerts.push_back(apexTop);
				triVerts.push_back(a);
				triVerts.push_back(b);
				triVerts.push_back(apexBottom);
				triVerts.push_back(b);
				triVerts.push_back(a);
			}
		}

		// A long, tapering, drooping frond in the joint's own canonical local frame (base at
		// local z=0, growing to +Z -- see ManifoldJointSpec::customShape's own comment): built as
		// `segments` chained AppendFrondSegment blades following emitPalmTreeItem's own parabolic
		// droop (`dropY = -ft*ft*frondDroop*length`, ft the normalized progress along the frond)
		// instead of that shader's straight-then-dropped quad strips, so the curve is continuous
		// rather than 2 hard-angled hinge points. Since the attaching joint's own worldDir is
		// purely horizontal (fDir below has y=0), SegmentTransform's minimal rotation from local
		// +Z is a pure yaw around world Y -- local +Y maps exactly to world up, so drooping
		// toward local -Y as Z increases droops toward world down, not some roll-dependent
		// direction. Position-only (numProp=3), same reasoning AppendLeafBlade's own sibling
		// gives -- BuildSkinnedManifoldAsset calls CalculateNormals(0) once on the final unioned
		// result.
		manifold::Manifold BuildPalmFrond(float length, float width, float droop, int segments, std::uint32_t seed) {
			std::mt19937                          rng(seed);
			std::uniform_real_distribution<float> unit(0.0f, 1.0f);

			std::vector<float>         vertProperties;
			std::vector<std::uint32_t> triVerts;

			auto pointAt = [&](float t) {
				return glm::vec3(0.0f, -droop * length * t * t, length * t);
			};

			segments = std::max(segments, 1);
			glm::vec3 prevPos = pointAt(0.0f);
			for (int s = 0; s < segments; ++s) {
				float     t0 = static_cast<float>(s) / static_cast<float>(segments);
				float     t1 = static_cast<float>(s + 1) / static_cast<float>(segments);
				glm::vec3 nextPos = pointAt(t1);

				glm::vec3 delta = nextPos - prevPos;
				float     segLen = glm::length(delta);
				glm::quat orientation = (segLen > 1e-6f) ? AlignZTo(delta) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
				// Tapers from the base to a point at the tip, same shape emitPalmTreeItem's own
				// per-cross-section width falloff produced, plus a small per-segment jitter so a
				// cluster of fronds doesn't look mechanically identical.
				float midT = (t0 + t1) * 0.5f;
				float segWidth = width * (1.0f - 0.5f * midT) * (0.9f + unit(rng) * 0.2f);

				AppendFrondSegment(segLen, segWidth, prevPos, orientation, vertProperties, triVerts);
				prevPos = nextPos;
			}

			manifold::MeshGL mesh;
			mesh.numProp = 3;
			mesh.vertProperties = std::move(vertProperties);
			mesh.triVerts = std::move(triVerts);
			return manifold::Manifold(mesh);
		}

	} // namespace

	ModelAsset BuildPalmTree(const PalmGenParams& params) {
		std::mt19937                          rng(params.seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		std::vector<ManifoldJointSpec> joints;
		joints.reserve(static_cast<std::size_t>(1 + params.frondCount));

		// Trunk: one curved joint spanning the whole height, matching emitPalmTreeItem's own
		// `bend = t*t*trunkBendAmount` quadratic lean (its per-instance runtime `sway` wind term
		// isn't part of this static rest shape -- this engine's own wind animation is applied
		// separately, same convention every other procedural archetype already uses) -- a real
		// round tube via curveWaypoints instead of that shader's flat 2-vertex-wide ribbon.
		float     bendAngle = unit(rng) * glm::two_pi<float>();
		glm::vec3 curveAxis(std::cos(bendAngle), 0.0f, std::sin(bendAngle));
		glm::vec3 trunkEnd = glm::vec3(0.0f, params.trunkHeight, 0.0f) + curveAxis * params.trunkBendAmount;

		std::vector<glm::vec3> waypoints;
		for (float t : {0.25f, 0.5f, 0.75f}) {
			float bend = params.trunkBendAmount * t * t;
			waypoints.push_back(glm::vec3(0.0f, params.trunkHeight * t, 0.0f) + curveAxis * bend);
		}

		ManifoldJointSpec trunk{};
		trunk.start = glm::vec3(0.0f);
		trunk.end = trunkEnd;
		trunk.radiusLow = params.trunkRadiusLow;
		trunk.radiusHigh = params.trunkRadiusHigh;
		trunk.parentIndex = kModelNoSkeletonSentinel;
		trunk.rigidity = params.trunkBaseRigidity;
		trunk.curveWaypoints = waypoints;

		auto trunkJoint = static_cast<std::uint32_t>(joints.size());
		joints.push_back(trunk);

		// Fronds: frondCount joints, all children of the trunk tip, radiating outward and
		// drooping -- matching emitPalmTreeItem's own 5-frond crown. Each is a terminal
		// customShape joint, same convention Hub/Flower/Leaf already use.
		for (int f = 0; f < params.frondCount; ++f) {
			float fAngle = bendAngle + (glm::two_pi<float>() * static_cast<float>(f)) / static_cast<float>(params.frondCount) +
				(unit(rng) - 0.5f) * 0.3f;
			glm::vec3 fDir(std::cos(fAngle), 0.0f, std::sin(fAngle));

			ManifoldJointSpec frond{};
			frond.start = trunkEnd;
			frond.end = trunkEnd + fDir * 0.01f; // tiny offset, orientation-only
			frond.parentIndex = trunkJoint;
			frond.rigidity = params.frondRigidity;

			float         length = params.frondLength * (0.85f + unit(rng) * 0.3f);
			float         width = params.frondWidth * (0.85f + unit(rng) * 0.3f);
			float         droop = params.frondDroop;
			int           segments = params.frondSegments;
			std::uint32_t frondSeed = params.seed * 2654435761u + static_cast<std::uint32_t>(f) * 97u;
			frond.customShape = [length, width, droop, segments, frondSeed]() {
				return BuildPalmFrond(length, width, droop, segments, frondSeed);
			};
			if (params.frondMaterialIndex.has_value()) {
				frond.materialIndex = params.frondMaterialIndex;
			}

			joints.push_back(frond);
		}

		return BuildSkinnedManifoldAsset(joints, params.circularSegments, params.materialIndex);
	}

} // namespace brassica
