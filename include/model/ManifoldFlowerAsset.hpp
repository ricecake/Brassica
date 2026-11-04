#pragma once

#include <manifold/manifold.h>

namespace brassica {

	// Flower head profile variants -- the exact 4 shapes shaders/foliage.mesh's own
	// emitFlowerItem already uses (Diamond/Bell/Star/Heart radial petal profiles), ported from
	// the live GPU shape to a real closed Manifold instead of an open flat fan: boidish's own
	// flower (6 flat Leaf quads around a Puffball sphere) was the thing sgreenhusted flagged as
	// jank/disconnected -- this replaces that idea entirely rather than porting it.
	enum class FlowerProfile { Diamond, Bell, Star, Heart };

	// Builds one flower head as a thin closed "lens" (a center apex above and below a ring of 8
	// petal vertices, same silhouette emitFlowerItem's own profile produces) in the same
	// canonical local frame a straight tube segment occupies (base at local z=0, growing to +Z --
	// see ManifoldJointSpec::customShape's own comment): the lens sits centered on local z=0,
	// meant to be plugged in as a terminal joint's customShape, not built standalone. headSize is
	// the petal ring's own radius. Position-only (numProp=3) -- the shared
	// BuildSkinnedManifoldAsset pipeline this feeds into already calls CalculateNormals(0) once
	// on the final unioned result, so this shape doesn't need its own normal pass.
	manifold::Manifold BuildFlowerHead(float headSize, FlowerProfile profile);

} // namespace brassica
