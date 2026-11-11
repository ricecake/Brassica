#pragma once

#include <cstdint>

#include <manifold/manifold.h>

namespace brassica {

	// A small tuft of flat, pointed leaf blades splayed outward from a common attachment
	// point, in the same canonical local frame a straight tube segment occupies (base at
	// local z=0, growing to +Z -- see ManifoldJointSpec::customShape's own comment): meant to
	// be plugged in as a terminal tube tip's customShape, same role ManifoldFlowerAsset's
	// BuildFlowerHead plays for the Flower archetype. leafSize is roughly each blade's own
	// length; leafCount how many blades make up the tuft. seed only varies this cluster's own
	// per-blade yaw/pitch/scale jitter, not the plant's overall growth. widthRatio is each
	// blade's width as a fraction of its own length -- 0.45 (the default) reads as a rounded
	// leaf; a much smaller ratio (e.g. 0.15) reads as a long, thin frond instead, same blade
	// construction either way. attachRadius is the attaching tube's own tip radius -- each
	// blade's base is pushed outward past it (same margin reasoning as the Hub-flare 1.5x fix)
	// so BuildSkinnedManifoldAsset's CSG union doesn't silently clip the blade's base away as
	// interior volume. Position-only (numProp=3) -- BuildSkinnedManifoldAsset calls
	// CalculateNormals(0) once on the final unioned result, so this shape doesn't need its own
	// normal pass.
	manifold::Manifold BuildLeafCluster(
		float leafSize, int leafCount, std::uint32_t seed, float widthRatio = 0.45f, float attachRadius = 0.0f
	);

	// A different terminal-tip shape for the same Leaf customShape slot: sgreenhusted reported
	// the plain fan cluster above (uniformly-splayed blades from one shared point) as "small and
	// sparse," and asked for a semi-random collection of overlapping balls to give the canopy
	// real texture/volume, with random instances of the plain blade above stuck to the side of
	// those balls to break up the otherwise-smooth sphere-cluster silhouette. Same local frame
	// and CSG-absorption margin reasoning as BuildLeafCluster; same signature so
	// ProceduralPlantGenerator.cpp can switch between the two per PlantGenParams::leafBallCluster
	// without any other call-site change. Not used for Palm's frond crown (Palm sets
	// leafBallCluster=false) -- a ball cluster reads as a canopy clump, not a frond, so Palm
	// keeps the plain fan shape this function's sibling already provides.
	manifold::Manifold BuildLeafBallCluster(
		float leafSize, int leafCount, std::uint32_t seed, float widthRatio = 0.45f, float attachRadius = 0.0f
	);

} // namespace brassica
