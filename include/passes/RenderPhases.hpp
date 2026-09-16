#pragma once

#include "graph/Execution.hpp"

// Home for this engine's own named render phases -- see graph::Phase's own comment
// (graph/Execution.hpp) for why the generic graph layer only defines the generic
// Early/Default/Late bands (plus PreviousFrame/NextFrame) and deliberately leaves anything more
// specific (e.g. DeferredShading, ForwardTranslucent) to whoever's building on top of it.
//
// Empty for now: every current node (GradientNode/TerrainNode/DeferredNode, the two atmosphere
// LUT nodes) runs at Phase::Default and relies purely on resource-dependency edges for ordering,
// not phase separation -- nothing here needs a name yet. Add one the same way any future node
// would opt in: a `static constexpr graph::Phase kPhase = SomeName;` on the node (see
// graph::HasPhase's comment, Node.hpp), paired with an
// `inline constexpr graph::Phase SomeName = graph::Phase::Late;` (or a custom int32_t value
// spaced away from the existing bands) here.

namespace brassica {
	namespace SubPhase {
		inline constexpr graph::Phase HudOverlay = graph::Phase(2000);
		inline constexpr graph::Phase MenuOverlay = graph::Phase(3000);
	} // namespace SubPhase
} // namespace brassica
