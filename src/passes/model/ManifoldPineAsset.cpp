#include "passes/model/ManifoldPineAsset.hpp"

#include <algorithm>
#include <vector>

#include <manifold/manifold.h>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"

namespace brassica {

	ModelAsset BuildPineTree(const PineGenParams& params) {
		std::vector<ManifoldJointSpec> joints;
		joints.reserve(static_cast<std::size_t>(params.trunkSegments + params.tierCount));

		// Trunk: plain tapered straight chain, no branching at all -- same reasoning Palm's own
		// branchChance=0 trunk already relies on (nothing here ever needs curveWaypoints or
		// CurveIsSafeToWarp at all, so there's no curvature-safety exposure for the trunk).
		float segmentHeight = params.trunkHeight / static_cast<float>(params.trunkSegments);
		std::uint32_t lastTrunkJoint = kModelNoSkeletonSentinel;
		float         y = 0.0f;
		for (int i = 0; i < params.trunkSegments; ++i) {
			float t0 = static_cast<float>(i) / static_cast<float>(params.trunkSegments);
			float t1 = static_cast<float>(i + 1) / static_cast<float>(params.trunkSegments);
			float radiusLow = glm::mix(params.trunkRadiusLow, params.trunkRadiusHigh, t0);
			float radiusHigh = glm::mix(params.trunkRadiusLow, params.trunkRadiusHigh, t1);

			ManifoldJointSpec spec{};
			spec.start = glm::vec3(0.0f, y, 0.0f);
			spec.end = glm::vec3(0.0f, y + segmentHeight, 0.0f);
			spec.radiusLow = radiusLow;
			spec.radiusHigh = radiusHigh;
			spec.parentIndex = lastTrunkJoint;
			spec.rigidity = params.trunkBaseRigidity;
			// No per-joint materialIndex override here -- trunk joints correctly fall back to
			// BuildSkinnedManifoldAsset's own asset-level default (params.materialIndex, passed
			// below), same as every other bark/trunk joint in this pipeline.

			lastTrunkJoint = static_cast<std::uint32_t>(joints.size());
			joints.push_back(spec);
			y += segmentHeight;
		}

		// Tiers: stacked cones (manifold::Manifold::Cylinder with radiusHigh=0 is exactly a
		// cone -- no custom mesh-building needed), each a child of the one below it so wind
		// sway compounds going up, same as a real tree's branches sway more than its trunk.
		// Attachment heights start partway up the trunk and overlap by tierOverlap so there's
		// no visible gap between tiers, matching the old shader's own overlapping-tier ranges.
		float tierStep = params.tierHeight * (1.0f - params.tierOverlap);
		float tierStartY = params.trunkHeight * 0.4f;
		float tierRadius = params.tierBaseRadius;
		std::uint32_t parentJoint = lastTrunkJoint;
		for (int tier = 0; tier < params.tierCount; ++tier) {
			float tierY = tierStartY + tierStep * static_cast<float>(tier);
			float height = params.tierHeight;
			float radius = tierRadius;
			int   segments = params.circularSegments;

			ManifoldJointSpec spec{};
			spec.start = glm::vec3(0.0f, tierY, 0.0f);
			spec.end = glm::vec3(0.0f, tierY + 0.01f, 0.0f); // tiny offset, orientation-only
			spec.radiusLow = 0.0f;
			spec.radiusHigh = 0.0f;
			spec.parentIndex = parentJoint;
			spec.rigidity = glm::mix(params.tierBaseRigidity, params.tierBaseRigidity * 0.6f, static_cast<float>(tier) / static_cast<float>(std::max(params.tierCount - 1, 1)));
			spec.customShape = [height, radius, segments]() {
				return manifold::Manifold::Cylinder(static_cast<double>(height), static_cast<double>(radius), 0.0, segments, false);
			};
			if (params.needleMaterialIndex.has_value()) {
				spec.materialIndex = params.needleMaterialIndex;
			}

			parentJoint = static_cast<std::uint32_t>(joints.size());
			joints.push_back(spec);
			tierRadius *= params.tierRadiusFalloff;
		}

		return BuildSkinnedManifoldAsset(joints, params.circularSegments, params.materialIndex);
	}

} // namespace brassica
