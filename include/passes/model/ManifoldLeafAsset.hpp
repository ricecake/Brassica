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
	// construction either way. Position-only (numProp=3) -- BuildSkinnedManifoldAsset calls
	// CalculateNormals(0) once on the final unioned result, so this shape doesn't need its own
	// normal pass.
	manifold::Manifold BuildLeafCluster(float leafSize, int leafCount, std::uint32_t seed, float widthRatio = 0.45f);

} // namespace brassica
