#pragma once

// Graph resource-key tags for the live render passes. Each is an empty, trivially-constructible
// struct satisfying brassica::graph::ResourceKey -- there is no runtime representation to get
// wrong, identity is purely at the type level (see include/graph/ResourceKey.hpp). Replaces
// include/passes/RenderResources.hpp's SwapchainData/GBufferData, which carried FrameGraphResource
// handles into a shared blackboard; the new model has no separate blackboard, cross-pass resource
// sharing is just these keys plus Declares<Create/Read/Modify<K>...> on each node.

#include "graph/Execution.hpp"
#include "vulkan/vulkan.hpp"

namespace brassica {

	struct CloudVolumeCascade0 {};

	struct CloudVolumeCascade1 {};

	struct CloudVolumeCascade2 {};

	inline graph::ResourceDesc CloudVolumeDesc() {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image3D,
			.width = 256,
			.height = 256,
			.depth = 256,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	struct Swapchain {};

	struct HdrColor {};

	struct AtmosphereRadiance {};

	struct GBufferPosition {};

	struct GBufferNormal {};

	struct GBufferAlbedo {};

	struct GBufferMaterial {};

	struct GBufferDepth {};

	struct ClusteredLighting {};

	struct AutoExposureBuffer {};

	struct BloomTextureMip0 {};
	struct BloomTextureMip1 {};
	struct BloomTextureMip2 {};
	struct BloomTextureMip3 {};
	struct BloomTextureMip4 {};

	struct LtmExpTextureMip0 {};
	struct LtmExpTextureMip1 {};
	struct LtmExpTextureMip2 {};
	struct LtmExpTextureMip3 {};
	struct LtmExpTextureMip4 {};

	struct LtmWgtTextureMip0 {};
	struct LtmWgtTextureMip1 {};
	struct LtmWgtTextureMip2 {};
	struct LtmWgtTextureMip3 {};
	struct LtmWgtTextureMip4 {};

	struct LtmFusedTexture {};

	// The result of TerrainPass::BuildOrUpdateAccelerationStructure, registered into the graph
	// as an Imported AccelerationStructure resource -- see the AccelerationStructure
	// resource-kind plan. The build itself stays entirely out-of-band (TerrainPass's own
	// transient command pool/queue, unchanged); only the result flows through the graph.
	struct TerrainTLAS {};

	// The terrain heightmap array -- a normal pass-owned persistent resource, Created<> by
	// TerrainGenNode (terrain/TerrainManager.hpp's TerrainClipmapDesc) like any other persistent
	// resource, not a hand-built C++ object imported into the graph. brassica::TerrainManager
	// (same header) is unrelated to this image now -- it's just the CPU-side manager (numLODs/
	// baseTexelSize config, Regenerate(), the async ground-height readback).
	struct TerrainClipmapTexture {};

	struct TerrainMinMaxTexture {};

	struct TerrainBiomeTexture {};

	struct TerrainWeatherBiomeTexture {};

	struct TerrainWeatherPingPongTexture {};

	struct TerrainWeatherMapATexture {};

	struct TerrainWeatherMapBTexture {};

	// Per-biome foliage properties (color/height/width/density/flowerRatio/...), one texel-row
	// per Whittaker biome index (shaders/helpers/whittaker.glsl, 0-9) -- FoliageManager-owned,
	// written host-side via a PredefinedTextureNode whenever the UI changes a biome's properties.
	struct FoliageBiomeTableTexture {};

	struct TransmittanceLUT {};

	struct MultiScatteringLUT {};

	struct SkyViewLUT {};

	struct ParticleBuffer {};

	struct ParticleTypeBuffer {};

	struct UnderwaterParticleAliveBuffer {};

	struct UnderwaterParticleIndirectBuffer {};

	struct AboveWaterParticleAliveBuffer {};

	struct AboveWaterParticleIndirectBuffer {};

	struct ParticleGridHeadsBuffer {};

	struct ParticleGridNextBuffer {};

	using ParticleAliveBuffer = AboveWaterParticleAliveBuffer;

	using ParticleIndirectBuffer = AboveWaterParticleIndirectBuffer;

	struct CylinderVertexBuffer {};

	struct CylinderIndexBuffer {};

	struct EntityInstanceBuffer {};

	struct EntityIndirectBuffer {};

	template <typename Tag = struct DefaultEntityTag>
	using LegacyEntityIndirectBuffer = EntityIndirectBuffer;

	struct MaterialBuffer {};

} // namespace brassica
