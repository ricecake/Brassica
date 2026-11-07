#pragma once

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// Stage 2's second producer: a minimal hand-built 3-joint jointed trunk (no file, no
	// textures, no fastgltf), proving ModelRegistry's interface is genuinely producer-agnostic
	// and not secretly glTF-shaped. Three rigidly-skinned cylinder segments in a parent-child
	// chain, one "Sway" clip bending the upper two joints, plus (Stage 4) a real
	// ModelSkeletonJoint skeleton (rigidity tapering trunk-to-tip, rest up-axis) so this is also
	// the live-wind-eval test subject the plan names.
	ModelAsset BuildProceduralTreeAsset();

} // namespace brassica
