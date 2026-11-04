#include "model/ManifoldSkeletalMeshBuilder.hpp"

#include <algorithm>
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

		// Procedural plant generation v2: Catmull-Rom through 4 control points, ported from
		// external/boidish/src/spline.cpp's own CatmullRom (same formula, glm instead of that
		// project's own Vector3 type).
		glm::vec3 CatmullRom(float t, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
			return 0.5f *
				((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * (t * t) +
			     (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * (t * t * t));
		}

		// One sampled point along a curved joint's path, in the same absolute rest-space as
		// start/end: the curve position itself, plus a parallel-transported (normal, bitangent)
		// frame -- the plane the straight segment's own circular cross-section gets placed into
		// at that point, so the tube doesn't twist along its length.
		struct CurveFrame {
			glm::vec3 position;
			glm::vec3 normal;
			glm::vec3 bitangent;
		};

		glm::vec3 PerpendicularTo(glm::vec3 dir) {
			return std::abs(dir.y) < 0.999f ? glm::normalize(glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f)))
											 : glm::normalize(glm::cross(dir, glm::vec3(1.0f, 0.0f, 0.0f)));
		}

		// controlPoints is [joint.start, joint.curveWaypoints..., joint.end] (>= 2 points).
		// sampleCount entries, evenly spaced across the full chain -- sampleCount must match
		// nDivisions+1, the number of extruded rings BuildCurvedSegment below creates, so each
		// ring's warp lands on an exact table entry with no interpolation needed.
		std::vector<CurveFrame> BuildCurveFrames(const std::vector<glm::vec3>& controlPoints, int sampleCount) {
			int n = static_cast<int>(controlPoints.size());
			glm::vec3 phantomBefore = controlPoints[0] - (controlPoints[1] - controlPoints[0]);
			glm::vec3 phantomAfter = controlPoints[n - 1] + (controlPoints[n - 1] - controlPoints[n - 2]);
			auto      at = [&](int i) -> glm::vec3 {
                if (i < 0) return phantomBefore;
                if (i >= n) return phantomAfter;
                return controlPoints[i];
            };
			auto sampleAt = [&](float s) -> glm::vec3 {
				int seg = glm::clamp(static_cast<int>(std::floor(s)), 0, n - 2);
				float t = s - static_cast<float>(seg);
				return CatmullRom(t, at(seg - 1), at(seg), at(seg + 1), at(seg + 2));
			};

			std::vector<CurveFrame> frames(sampleCount);
			float                   sMax = static_cast<float>(n - 1);
			const float              dS = sMax * 0.001f;
			glm::vec3                lastNormal{};
			for (int i = 0; i < sampleCount; ++i) {
				float     s = sMax * static_cast<float>(i) / static_cast<float>(sampleCount - 1);
				glm::vec3 pos = sampleAt(s);

				glm::vec3 tangent;
				if (i < sampleCount - 1) {
					glm::vec3 next = sampleAt(std::min(s + dS, sMax));
					tangent = glm::dot(next - pos, next - pos) > 1e-10f ? glm::normalize(next - pos) : glm::vec3(0.0f, 1.0f, 0.0f);
				} else {
					glm::vec3 prev = sampleAt(std::max(s - dS, 0.0f));
					tangent = glm::dot(pos - prev, pos - prev) > 1e-10f ? glm::normalize(pos - prev) : glm::vec3(0.0f, 1.0f, 0.0f);
				}

				glm::vec3 normal;
				if (i == 0) {
					normal = PerpendicularTo(tangent);
				} else {
					glm::vec3 projected = lastNormal - tangent * glm::dot(tangent, lastNormal);
					normal = (glm::dot(projected, projected) > 1e-10f) ? glm::normalize(projected) : PerpendicularTo(tangent);
				}
				lastNormal = normal;

				frames[i] = {pos, normal, glm::normalize(glm::cross(tangent, normal))};
			}
			return frames;
		}

		// Heuristic, not a proof (sgreenhusted's own framing: "it will require self intersection
		// checks, but it shouldn't be a problem"): estimates local radius of curvature from the
		// turn angle and arc length between consecutive frames, and rejects the curve if that
		// radius of curvature ever drops below a safety margin over the tube's own local radius
		// -- a tube bent tighter than its own radius folds over itself. kMinCurvatureRadiusFactor
		// is deliberately conservative (1.5x, not 1.0x) since this is an estimate, not exact.
		bool CurveIsSafeToWarp(const std::vector<CurveFrame>& frames, float radiusLow, float radiusHigh) {
			constexpr float kMinCurvatureRadiusFactor = 1.5f;
			float           maxRadius = std::max(radiusLow, radiusHigh);
			for (std::size_t i = 0; i + 1 < frames.size(); ++i) {
				glm::vec3 delta = frames[i + 1].position - frames[i].position;
				float     arcLen = glm::length(delta);
				if (arcLen < 1e-6f) {
					continue;
				}
				glm::vec3 tangentA = glm::normalize(delta);
				glm::vec3 tangentBDelta =
					(i + 2 < frames.size()) ? frames[i + 2].position - frames[i + 1].position : delta;
				if (glm::dot(tangentBDelta, tangentBDelta) < 1e-10f) {
					continue;
				}
				glm::vec3 tangentB = glm::normalize(tangentBDelta);
				float     turnAngle = std::acos(glm::clamp(glm::dot(tangentA, tangentB), -1.0f, 1.0f));
				if (turnAngle < 1e-5f) {
					continue; // effectively straight here
				}
				float curvatureRadius = arcLen / turnAngle;
				if (curvatureRadius < maxRadius * kMinCurvatureRadiusFactor) {
					return false;
				}
			}
			return true;
		}

		// Builds the joint's segment bent along joint.curveWaypoints, ALWAYS positioned in
		// absolute rest-space (same space as start/end -- the caller must not apply
		// SegmentTransform to this function's result, unlike the plain-straight-cylinder case):
		// a straight tapered tube subdivided along its length (Manifold::Extrude directly, not
		// the Manifold::Cylinder convenience wrapper -- Cylinder uses nDivisions=0, i.e. exactly 2
		// rings with a straight side wall between them, confirmed against
		// external/manifold/src/constructors.cpp, which would leave the middle of the tube dead
		// straight no matter what a per-vertex Warp did to its two end rings), then Warp()'d so
		// each ring lands on the curve at the matching sample in frames. Falls back to a plain
		// straight segment (still positioned, via the same SegmentTransform every other joint
		// uses) if the curve is empty, radiusLow isn't positive (the cone case Manifold::Cylinder's
		// own flip logic handles, not needed for plant stems), or CurveIsSafeToWarp rejects the
		// bend.
		manifold::Manifold BuildCurvedSegment(
			glm::vec3 start, glm::vec3 end, const std::vector<glm::vec3>& waypoints, float radiusLow, float radiusHigh,
			int circularSegments
		) {
			double straightHeight = static_cast<double>(glm::length(end - start));
			auto   straightFallback = [&]() {
				manifold::Manifold cyl = manifold::Manifold::Cylinder(
					straightHeight, static_cast<double>(radiusLow), static_cast<double>(radiusHigh), circularSegments,
					false
				);
				return cyl.Transform(SegmentTransform(start, end - start));
			};

			if (waypoints.empty() || radiusLow <= 0.0f) {
				return straightFallback();
			}

			std::vector<glm::vec3> controlPoints;
			controlPoints.reserve(waypoints.size() + 2);
			controlPoints.push_back(start);
			controlPoints.insert(controlPoints.end(), waypoints.begin(), waypoints.end());
			controlPoints.push_back(end);

			int ringDivisions = std::max(8, static_cast<int>(waypoints.size()) * 6);
			std::vector<CurveFrame> frames = BuildCurveFrames(controlPoints, ringDivisions + 1);

			if (!CurveIsSafeToWarp(frames, radiusLow, radiusHigh)) {
				spdlog::warn("BuildCurvedSegment: bend too tight relative to tube radius, falling back to straight");
				return straightFallback();
			}

			int n = circularSegments > 2 ? circularSegments : 12;
			manifold::SimplePolygon circle(n);
			const double             dPhi = 360.0 / n;
			for (int i = 0; i < n; ++i) {
				circle[i] = manifold::vec2(radiusLow * manifold::cosd(dPhi * i), radiusLow * manifold::sind(dPhi * i));
			}
			manifold::Polygons crossSection{circle};
			manifold::vec2     scaleTop(static_cast<double>(radiusHigh) / static_cast<double>(radiusLow));

			manifold::Manifold straight =
				manifold::Manifold::Extrude(crossSection, straightHeight, ringDivisions, 0.0, scaleTop);

			manifold::Manifold warped = straight.Warp([&](manifold::vec3& v) {
				double heightD = straightHeight;
				int    ring = static_cast<int>(std::lround((v.z / heightD) * ringDivisions));
				ring = std::clamp(ring, 0, ringDivisions);
				const CurveFrame& f = frames[static_cast<std::size_t>(ring)];
				glm::vec3         displaced =
					f.position + static_cast<float>(v.x) * f.normal + static_cast<float>(v.y) * f.bitangent;
				v = manifold::vec3(displaced.x, displaced.y, displaced.z);
			});

			if (warped.Status() != manifold::Manifold::Error::NoError) {
				spdlog::warn(
					"BuildCurvedSegment: warp produced status {}, falling back to straight",
					static_cast<int>(warped.Status())
				);
				return straightFallback();
			}
			return warped;
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
			manifold::Manifold segment;
			if (joint.customShape) {
				// Built in the same canonical local frame a straight cylinder would occupy (base
				// at local z=0, growing to +Z) -- positioned the same way every other joint is.
				segment = joint.customShape().Transform(SegmentTransform(joint.start, joint.end - joint.start));
			} else if (!joint.curveWaypoints.empty()) {
				// Already positioned in absolute rest-space (including its own straight-fallback
				// paths) -- see BuildCurvedSegment's own comment for why no SegmentTransform
				// applies here.
				segment =
					BuildCurvedSegment(joint.start, joint.end, joint.curveWaypoints, joint.radiusLow, joint.radiusHigh, circularSegments);
			} else {
				// Unchanged from before curves/custom shapes existed -- byte-identical output.
				double height = static_cast<double>(glm::length(joint.end - joint.start));
				segment = manifold::Manifold::Cylinder(
					height, static_cast<double>(joint.radiusLow), static_cast<double>(joint.radiusHigh), circularSegments,
					false
				);
				segment = segment.Transform(SegmentTransform(joint.start, joint.end - joint.start));
			}
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
