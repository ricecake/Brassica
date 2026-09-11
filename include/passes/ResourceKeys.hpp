#pragma once

// Graph resource-key tags for the live render passes. Each is an empty, trivially-constructible
// struct satisfying brassica::graph::ResourceKey -- there is no runtime representation to get
// wrong, identity is purely at the type level (see include/graph/ResourceKey.hpp). Replaces
// include/passes/RenderResources.hpp's SwapchainData/GBufferData, which carried FrameGraphResource
// handles into a shared blackboard; the new model has no separate blackboard, cross-pass resource
// sharing is just these keys plus Declares<Create/Read/Modify<K>...> on each node.

namespace brassica {

	struct Swapchain {};
	struct LitSwapchain {};

	struct GradientBackground {};

	struct GBufferPosition {};

	struct GBufferNormal {};

	struct GBufferAlbedo {};

	struct GBufferDepth {};

	// The result of TerrainPass::BuildOrUpdateAccelerationStructure, registered into the graph
	// as an Imported AccelerationStructure resource -- see the AccelerationStructure
	// resource-kind plan. The build itself stays entirely out-of-band (TerrainPass's own
	// transient command pool/queue, unchanged); only the result flows through the graph.
	struct TerrainTLAS {};

	struct TransmittanceLUT {};

	struct MultiScatteringLUT {};

} // namespace brassica
