#pragma once

#include "graph/Declaration.hpp"
#include "passes/ResourceKeys.hpp"

namespace brassica {

	// Use a group alias only when a node touches the WHOLE bundle under ONE uniform op. A partial
	// or mixed-op consumer (each particle node mixes Modify/Read across the particle-buffer
	// quartet, e.g. ParticleLivenessNode is Modify/Read/Modify/Modify) stays explicit -- forcing
	// it through a group would declare a dependency that doesn't exist.
	template <template <typename> class Op>
	using GBuffer = graph::Group<Op, GBufferPosition, GBufferNormal, GBufferAlbedo, GBufferDepth>;

} // namespace brassica
