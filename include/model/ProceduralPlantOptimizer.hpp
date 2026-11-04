#pragma once

#include "model/ProceduralPlantIR.hpp"

namespace brassica {

	// Simplifies a freshly-grown PlantIR in place, adapted from
	// external/boidish/src/procedural_optimizer.cpp's ConsolidateTubes/HandleJunctions. Turns "a
	// chain of straight segments" into "one joint's worth of curve waypoints, with a flare hub at
	// each real branch point" -- exactly the input shape ProceduralPlantGenerator's conversion
	// into ManifoldJointSpecs wants.
	class ProceduralPlantOptimizer {
	public:
		static void Optimize(PlantIR& ir);

	private:
		// Merges a Tube with its one-and-only Tube child into a single longer Tube. Perfectly
		// collinear runs merge with no trace left behind; a run that bends (however slightly)
		// leaves a ControlPoint at the bend so the curve-fitting step downstream still sees where
		// it was.
		static void ConsolidateTubes(PlantIR& ir);

		// Inserts a Hub wherever an element (after consolidation) has more than one child -- a
		// real branch point, which gets an organic flare sphere once converted to geometry.
		static void HandleJunctions(PlantIR& ir);
	};

} // namespace brassica
