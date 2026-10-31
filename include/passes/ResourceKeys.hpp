#pragma once

// Graph resource-key tags for the live render passes -- identity is purely at the type level
// (include/graph/ResourceKey.hpp). Replaces include/passes/RenderResources.hpp's
// SwapchainData/GBufferData, which carried FrameGraphResource handles into a shared blackboard;
// the new model has no separate blackboard, cross-pass resource sharing is just these keys plus
// each node's own SetupBuilder calls (Create/Read/Modify<K>()) inside Setup. Most keys below are
// still plain empty tags -- a key gains a `Desc(const FrameContext&) const` member only once
// something has converted to the resource-type-ownership model (see graph/ResourceCatalog.hpp);
// an empty tag with no Desc() is just as valid as before, Provision falls back to whatever a
// node's realization supplies.

#include <algorithm>
#include <cmath>

#include "vulkan/vulkan.hpp"

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "lighting/Light.hpp"

namespace brassica {

	inline graph::ResourceDesc HiZTextureDesc(std::uint32_t width, std::uint32_t height, std::uint32_t mips = 0) {
		std::uint32_t calculatedMips = mips;
		if (calculatedMips == 0) {
			const std::uint32_t maxDim = (std::max)(width, height);
			calculatedMips = maxDim > 0
				? static_cast<std::uint32_t>(std::floor(std::log2(static_cast<float>(maxDim)))) + 1
				: 1;
		}
		calculatedMips = (std::min)(calculatedMips, 12u);

		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.mips = calculatedMips,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	// First production conversion to the type-owned-desc model (see graph/ResourceCatalog.hpp):
	// Desc() is the exact auto-compute path HiZTextureDesc(w, h) already took (mips=0), which
	// Spike A's desc census confirmed is arithmetically identical to HiZDownsampleNode's own
	// explicit-mips call -- both reduce to floor(log2(maxDim))+1, clamped to 12. kLifetime =
	// History documents that this type maintains an interframe double-buffered pair (today still
	// via the History<HiZTexture> key wrapper, not yet read from this field anywhere -- see
	// ResourceKey.hpp's Lifetime comment).
	struct HiZTexture {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::History;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return HiZTextureDesc(ctx.width, ctx.height);
		}
	};

	inline graph::ResourceDesc CloudShadowMapDesc(std::uint32_t width = 1024, std::uint32_t height = 1024) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.mips = 1,
			.layers = 5,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16G16B16A16Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			),
			// CloudShadowGenNode now only rewrites 1/4 of texels per frame (Bayer-staggered
			// update, shaders/cloud_shadow_gen.comp) -- the other 3/4 must survive untouched
			// across frames. Without this, PhysicalRegistry::ProvisionTexture's pooled
			// allocation never re-Acquires after frame 1 (same id/desc hits the early-return
			// path), so the aliasing pool's lifetime bookkeeping stays frozen at frame 1's
			// window and will eventually hand this exact memory to an unrelated resource while
			// still in use -- a real, confirmed-class bug (see Execution.hpp's .persistent doc
			// comment and the FoliageBiomeTableTexture precedent it was already fixed for).
			// This was already latently true before staggering, just invisible, since the old
			// full-rewrite-every-frame behavior erased any evidence of silent corruption.
			.persistent = true,
		};
	}

	struct CloudShadowMap {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return CloudShadowMapDesc(); }
	};

	inline graph::ResourceDesc TerrainShadowMapDesc(std::uint32_t width = 1024, std::uint32_t height = 1024) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.mips = 1,
			.layers = 3,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			),
			.persistent = true,
		};
	}

	struct TerrainShadowMap {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return TerrainShadowMapDesc(); }
	};

	inline graph::ResourceDesc CloudVolumeDesc() {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image3D,
			.width = 256,
			.height = 256,
			.depth = 256,
			// Mips 0..4 (256/128/64/32/16) -- CloudVolumeDownsampleNode fills 1..4 from mip 0.
			// Matches the clamp cloud_render.frag's sampleCloudVolume() uses when picking a
			// distance-based LOD (mirroring boidish's own clamp(volumeMip, 0.0, 4.0)); mips
			// beyond that would never be sampled, so there's no reason to generate them.
			.mips = 5,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	// Single tileable detail-noise volume (periodic in all 3 axes, world-space period
	// constants::Weather::kCloudVolumePeriod) replacing the former 3-cascade camera-snapped
	// system. Big-picture cloud shape now comes entirely from the weather coverage map; this
	// volume only supplies reusable shape/erosion detail, so it never needs camera-relative
	// recentering -- see CloudVolumeGenNode (mip 0) and CloudVolumeDownsampleNode (mips 1-4).
	// Lifetime::Persistent -- same reasoning as TransmittanceLUT/MultiScatteringLUT above:
	// CloudVolumeGenNode only bakes this once (dirty-flag gated), so "this frame's producer is
	// inactive" is the normal steady state, not "no valid data." Without this, every reader
	// (CloudShadowGenNode, CloudRenderNode) sees no active writer preceding it once the node
	// goes inactive after frame 1, and the zero-init path clears the whole volume to black every
	// frame -- exactly the vanishing-texture bug this was found by.
	struct CloudDetailVolume {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Persistent;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return CloudVolumeDesc(); }
	};

	inline graph::ResourceDesc VolumetricFroxelDesc() {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image3D,
			.width = 128,
			.height = 96,
			.depth = 64,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16G16B16A16Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	struct VolumetricCascade0Scattering {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	struct VolumetricCascade0Extinction {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	struct VolumetricCascade1Scattering {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	struct VolumetricCascade1Extinction {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	struct VolumetricCascade2Scattering {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	struct VolumetricCascade2Extinction {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricFroxelDesc(); }
	};

	inline graph::ResourceDesc VolumetricNoiseDesc() {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = 512,
			.height = 512,
			.depth = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16G16B16A16Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	struct VolumetricNoiseTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return VolumetricNoiseDesc(); }
	};

	struct Swapchain {};

	struct HdrColor {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat);
		}
	};

	struct AtmosphereRadiance {};

	struct GBufferPosition {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat);
		}
	};

	struct GBufferNormal {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat);
		}
	};

	struct GBufferAlbedo {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Srgb);
		}
	};

	struct GBufferMaterial {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm);
		}
	};

	struct GBufferDepth {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::DepthBufferDesc(ctx.width, ctx.height);
		}
	};

	struct ClusteredLighting {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::StorageBufferDesc(TOTAL_CLUSTERS * sizeof(ClusterGPU));
		}
	};

	// AutoExposureBuffer is imported (Engine::DrawFrame's RegisterImportedBuffer) -- Provision
	// skips an imported key's realization entirely regardless of its desc, so this stays a
	// plain tag; there's nothing for a Desc() to usefully describe.
	struct AutoExposureBuffer {};

	// Frame set binding 2 (shaders/lighting.glsl's LightsBuffer, uLights[]/uLightCount) -- the
	// one genuinely hybrid frame-set buffer: Engine::DrawFrame's CPU upload
	// (LightManager::GetLightsSSBOData) writes it fresh every frame *before* graph execution,
	// then ParticleBehaviorNode's firefly particles write real dynamic lights into it *on top*
	// (shaders/particle_behavior.comp's LIGHTS_BUFFER_WRITABLE path) during graph execution.
	// Imported, same reasoning and the same tag-only shape as AutoExposureBuffer just above --
	// Provision skips an imported key's realization regardless of desc.
	struct LightsBuffer {};

	// Frame set binding 7 (shaders/lighting.glsl's LightningStrikePool) -- a persistent,
	// 100%-GPU-owned strike pool. climate_ca.comp claims a ring-buffer slot and writes a fresh
	// strike's origin/seed/lifetime when weather conditions trigger one; lightning_age.comp ages
	// every slot and writes its reserved light-slot entry; lightning.task reads it to plot/cull.
	// Imported once at Engine init (include/types/LightningStrike.hpp's LightningStrikePoolLayout
	// sizes the allocation) -- its handle never changes, same as TerrainReadbackStaging below, not
	// AutoExposureBuffer/LightsBuffer's "different real handle per FRAME_OVERLAP slot" case.
	struct LightningStrikePool {};

	// TonemapComputeNode's bloom/local-tonemap mip chain: mip N is ctx.width/2 (then height)
	// halved N more times, floored to 1 -- same formula at every one of the 16 keys below, just
	// a different (and, for the Ltm pair, duplicated) mip index. kMip is a type-level constant,
	// not a catalog config field: unlike TerrainMinMaxTexture's numLODs, nothing publishes this
	// per-instance, it's baked into which struct you are.
	inline graph::ResourceDesc TonemapMipDesc(const graph::FrameContext& ctx, std::uint32_t mip) {
		std::uint32_t w = std::max(1u, (ctx.width / 2) >> mip);
		std::uint32_t h = std::max(1u, (ctx.height / 2) >> mip);
		return graph::ComputeStorageImageDesc(w, h, vk::Format::eR16G16B16A16Sfloat);
	}

	struct BloomTextureMip0 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 0); }
	};

	struct BloomTextureMip1 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 1); }
	};

	struct BloomTextureMip2 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 2); }
	};

	struct BloomTextureMip3 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 3); }
	};

	struct BloomTextureMip4 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 4); }
	};

	struct LtmExpTextureMip0 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 0); }
	};

	struct LtmExpTextureMip1 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 1); }
	};

	struct LtmExpTextureMip2 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 2); }
	};

	struct LtmExpTextureMip3 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 3); }
	};

	struct LtmExpTextureMip4 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 4); }
	};

	struct LtmWgtTextureMip0 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 0); }
	};

	struct LtmWgtTextureMip1 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 1); }
	};

	struct LtmWgtTextureMip2 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 2); }
	};

	struct LtmWgtTextureMip3 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 3); }
	};

	struct LtmWgtTextureMip4 {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 4); }
	};

	// Same dims as mip 0 of the chain above (ltm_fuse.comp fuses down to one mip-0-sized
	// texture).
	struct LtmFusedTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const { return TonemapMipDesc(ctx, 0); }
	};

	// The result of TerrainPass::BuildOrUpdateAccelerationStructure, registered into the graph
	// as an Imported AccelerationStructure resource -- see the AccelerationStructure
	// resource-kind plan. The build itself stays entirely out-of-band (TerrainPass's own
	// transient command pool/queue, unchanged); only the result flows through the graph.
	struct TerrainTLAS {};

	// mips = 1: this image has no mip chain of its own -- only TerrainMinMaxDesc below does.
	inline graph::ResourceDesc TerrainClipmapDesc(std::uint32_t numLODs) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = constants::Class::Terrain::MapDim,
			.height = constants::Class::Terrain::MapDim,
			.mips = 1,
			.layers = numLODs,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc
			),
			.persistent = true,
		};
	}

	// floor(log2(MapDim)) + 1 -- the full mip chain down to 1x1 for a 1024-wide image (mips
	// 0-10), well under SPD's 12-mip-per-dispatch limit. RG32F, not RGBA32F: only min (x) and
	// max (y) are meaningful, and the per-mip storage views this needs (PhysicalTexture::
	// CreateView, PhysicalResource.hpp) are what the broken mips=1 claim on this key used to mask
	// -- nothing ever read past mip 0 before.
	inline graph::ResourceDesc TerrainMinMaxDesc(std::uint32_t numLODs) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = constants::Class::Terrain::MapDim,
			.height = constants::Class::Terrain::MapDim,
			.mips = 11,
			.layers = numLODs,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc
			),
			.persistent = true,
		};
	}

	inline graph::ResourceDesc TerrainBiomeDesc(std::uint32_t numLODs) { return TerrainClipmapDesc(numLODs); }

	// The terrain heightmap array -- a normal pass-owned persistent resource, Created<> by
	// TerrainGenNode, like any other persistent resource, not a hand-built C++ object imported
	// into the graph. numLODs is published by TerrainGenNode::SetFrameParams (it tracks
	// terrainGridParams.x, the one piece of runtime config this desc needs beyond FrameContext).
	// brassica::TerrainManager (terrain/TerrainManager.hpp) is unrelated to this image now --
	// it's just the CPU-side manager (numLODs/baseTexelSize config, Regenerate(), the async
	// ground-height readback).
	struct TerrainClipmapTexture {
		std::uint32_t numLODs = constants::Class::Terrain::DefaultMaxLODs;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return TerrainClipmapDesc(numLODs); }
	};

	struct TerrainMinMaxTexture {
		std::uint32_t numLODs = constants::Class::Terrain::DefaultMaxLODs;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return TerrainMinMaxDesc(numLODs); }
	};

	struct TerrainBiomeTexture {
		std::uint32_t numLODs = constants::Class::Terrain::DefaultMaxLODs;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const { return TerrainBiomeDesc(numLODs); }
	};

	// TerrainManager's host-visible readback staging buffer (TerrainReadbackNode's
	// Modify<>, terrain/TerrainManager.hpp) -- imported, same tag-only shape as
	// AutoExposureBuffer/LightsBuffer: the real allocation is owned and sized by TerrainManager
	// itself (1MB, comfortably covering the 128x128 RGBA32F window this is actually used for),
	// registered once at Engine::InitVulkan time since (unlike those per-FRAME_OVERLAP-slot
	// buffers) this one handle never changes.
	struct TerrainReadbackStaging {};

	// Moved from TerrainBiomeNode.hpp (same reasoning as TerrainClipmapDesc/TerrainMinMaxDesc's
	// move from terrain/TerrainManager.hpp in S2b batch 2): a key's Desc() lives here, so the
	// constants/helper it calls have to live where ResourceKeys.hpp can see them without
	// including the node header back (that would be circular -- TerrainBiomeNode.hpp already
	// includes this file). TerrainBiomeNode.hpp still uses kSimTextureDim/kCoverageTextureDim/
	// kBiomeTextureDim/kBlendDitherDivisor/kUpdateInterval directly in its push constants and
	// dispatch math; nothing there changes, it just reaches them through this header's own
	// existing include.
	namespace constants::Weather {
		constexpr std::uint32_t kSimTextureDim = 2048;
		constexpr std::uint32_t kCoverageTextureDim = 2048;
		constexpr std::uint32_t kBiomeTextureDim = 2048;
		constexpr std::uint32_t kBlendDitherDivisor = 8;
		constexpr std::uint32_t kUpdateInterval = 300;
		constexpr std::uint32_t kCellsPerUpdate = 512;

		// World-space period (metres) of CloudDetailVolume's periodic tiling, in all 3 axes.
		// Matches the former near cascade's footprint so existing noise-frequency tuning still
		// reads at roughly the same visual scale.
		constexpr float kCloudVolumePeriod = 20000.0f;
	} // namespace constants::Weather

	inline graph::ResourceDesc WeatherBiomeImageDesc(std::uint32_t dim) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = dim,
			.height = dim,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
			),
			.persistent = true,
		};
	}

	// TerrainBiomeNode's blended climate/biome-classification output (TerrainBiomeMapA/B's
	// ping-pong target) -- deliberately a distinct key from TerrainBiomeTexture above, which
	// TerrainGenNode owns and writes per-texel into the terrain clipmap's own layer/mip shape
	// (TerrainBiomeDesc). The two used to collide on one key (both Create<>'d it with different,
	// incompatible descs -- TerrainClipmapDesc's numLODs-layer shape vs WeatherBiomeImageDesc's
	// single-layer kBiomeTextureDim shape -- which silently reprovisioned/retired the resource
	// every frame and left its bindless descriptor racing between the two shapes, confirmed via
	// GPU-assisted validation as "descriptor index never updated").
	struct TerrainBiomeMainTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim);
		}
	};

	struct TerrainWeatherBiomeTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim);
		}
	};

	struct TerrainWeatherPingPongTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kSimTextureDim);
		}
	};

	struct TerrainWeatherMapATexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim);
		}
	};

	struct TerrainWeatherMapBTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim);
		}
	};

	struct TerrainBiomeMapATexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim);
		}
	};

	struct TerrainBiomeMapBTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim);
		}
	};

	// Per-biome foliage properties (color/height/width/density/flowerRatio/...), one texel-row
	// per Whittaker biome index (shaders/helpers/whittaker.glsl, 0-9) -- FoliageManager-owned,
	// written host-side via a PredefinedTextureNode whenever the UI changes a biome's properties.
	struct FoliageBiomeTableTexture {};

	// Lifetime::Persistent -- TransmittanceLUTNode/MultiScatteringLUTNode (AtmosphereLUTNode.hpp)
	// only regenerate these when AtmospherePushConstants actually changes (AtmosphereRegenerationState's
	// throttle), so "this frame's producer is inactive" is the *normal*, expected case once the
	// atmosphere settles, not "there's no valid data" -- unlike a genuine feature toggle (GTAO,
	// Foliage, bloom), where inactive really does mean "no contribution wanted." Without this,
	// §E's zero-init list (RelationalCompiler.hpp) can't tell the two apart and wrongly clears a
	// perfectly good cached LUT to black the moment the throttle kicks in -- found via a real
	// hardware test: a stable sun angle produced a persistently black sky (SkyViewLUTNode's own
	// plain Read<TransmittanceLUT>()/Read<MultiScatteringLUT>() got zeroed every frame once the
	// throttle stopped regenerating them), while the independently-drawn sun disc kept rendering
	// correctly. detail::IsPersistentKey (RelationalCompiler.hpp) is what actually reads this.
	struct TransmittanceLUT {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Persistent;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat);
		}
	};

	struct MultiScatteringLUT {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Persistent;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ComputeStorageImageDesc(32, 32, vk::Format::eR32G32B32A32Sfloat);
		}
	};

	struct SkyViewLUT {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat);
		}
	};

	inline graph::ResourceDesc GTAOTextureDesc(std::uint32_t width, std::uint32_t height) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.mips = 1,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR16Sfloat),
			// eTransferDst: lets ZeroInitializeInactiveReads (PhysicalExecutionBackend.hpp, S5)
			// clear this texture via vkCmdClearColorImage when GTAO is disabled -- found via a
			// real hardware test: a GTAO-off frame hit a real "image not compatible with
			// VK_IMAGE_USAGE_TRANSFER_DST_BIT" validation warning on this exact clear, since
			// nothing before S5 ever needed to transfer-write into this texture.
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	struct GTAOTexture {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return GTAOTextureDesc(ctx.width, ctx.height);
		}
	};

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

	// Matches shaders/foliage_archetype.glsl / foliage_instance.glsl exactly.
	inline constexpr std::uint32_t kFoliageArchetypeMaxVerts = 128;
	inline constexpr std::uint32_t kFoliageArchetypeMaxTris = 128;
	inline constexpr std::uint32_t kFoliageTreeArchetypeCount = 8;
	inline constexpr std::uint32_t kFoliageArchetypeCount = 16;
	inline constexpr std::uint32_t kFoliageInstanceCapacity = 8192;

	// Catalog-converted (Desc() lives on the key, like TerrainClipmapTexture/TerrainMinMaxTexture
	// above) rather than each Create<>/Read<>() call site repeating its own .Desc(): with no
	// per-site desc to disagree about, every toucher can carry a real Read<>/Modify<>() instead
	// of the buffer_reference-consumer workaround of an explicit After<>() plus a hand-rolled
	// barrier (EntityNode.hpp's EntityInstanceBuffer pattern -- itself flagged there as
	// pre-catalog-conversion, not the thing to copy for a new resource). Per-vertex byte size
	// verified via spirv-dis against the real std430 layout (vec3+vec3+vec2+vec4+uint -> 80
	// bytes/vertex, not the naively-expected 56) rather than hand-computed -- see
	// feedback_push_constant_struct_editing.
	inline graph::ResourceDesc FoliageArchetypeVertexBufferDesc() {
		graph::ResourceDesc desc = graph::StorageBufferDesc(
			static_cast<std::uint64_t>(kFoliageArchetypeCount) * kFoliageArchetypeMaxVerts * 80
		);
		desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
		desc.persistent = true;
		return desc;
	}

	inline graph::ResourceDesc FoliageArchetypeIndexBufferDesc() {
		graph::ResourceDesc desc = graph::StorageBufferDesc(
			static_cast<std::uint64_t>(kFoliageArchetypeCount) * kFoliageArchetypeMaxTris * 16
		);
		desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
		desc.persistent = true;
		return desc;
	}

	inline graph::ResourceDesc FoliageArchetypeMetaBufferDesc() {
		graph::ResourceDesc desc =
			graph::StorageBufferDesc(static_cast<std::uint64_t>(kFoliageArchetypeCount) * 8);
		desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
		desc.persistent = true;
		return desc;
	}

	// Persistent, GPU-baked (not host-written) tree/bush archetype shapes -- see
	// shaders/foliage_archetype.glsl for the layout and FoliageBakeNode.hpp for how/when these
	// get populated. Still read via buffer_reference (device address in a push constant, not a
	// bindless SSBO index -- no generic bindless-buffer GLSL array exists in this codebase yet,
	// see shaders/bindless.glsl); only the graph-side declaration changed, not the GPU-side access.
	struct FoliageArchetypeVertexBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return FoliageArchetypeVertexBufferDesc();
		}
	};
	struct FoliageArchetypeIndexBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return FoliageArchetypeIndexBufferDesc();
		}
	};
	struct FoliageArchetypeMetaBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return FoliageArchetypeMetaBufferDesc();
		}
	};

	// Scratch color target: FoliageBakeNode's mesh-shader pipeline needs a real Write-access
	// Image2D realization to be valid (DynamicRenderingWrapper::Begin requires at least one), but
	// the bake's real output is the buffers above, not anything rasterized here. Sized to the
	// real frame (FoliageBakeNode::Setup), not a tiny throwaway extent -- BindGraphics sets the
	// viewport/scissor from the frame's own width/height, not from this attachment's extent.
	struct FoliageBakeDummyTarget {};

	// Per-frame tree/bush instance list collected by foliage.mesh, and the indirect-draw args
	// buffer whose groupCountX doubles as the live append counter -- see
	// shaders/foliage_instance.glsl for the layout. Rebuilt every frame (unlike the archetype
	// buffers above), zeroed each frame before collection via ClearBufferToZero. Catalog-converted
	// for the same reason as the archetype buffers above.
	struct FoliageInstanceBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			graph::ResourceDesc desc =
				graph::StorageBufferDesc(static_cast<std::uint64_t>(kFoliageInstanceCapacity) * 32);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			desc.persistent = true;
			return desc;
		}
	};
	struct FoliageIndirectArgsBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			graph::ResourceDesc desc = graph::StorageBufferDesc(16); // real struct is 12 bytes; 16 for safety margin
			desc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eIndirectBuffer |
				// eTransferSrc: lets a debug/test readback (vkCmdCopyBuffer out of this buffer)
				// actually work -- added while writing test_foliage_node.cpp's own
				// FoliageIndirectResetNode regression test, which needs to verify this exact
				// buffer's real post-reset contents. Harmless to leave on a 16-byte buffer, and
				// this is specifically the kind of silent, hard-to-observe buffer that benefits
				// from being readable.
				vk::BufferUsageFlagBits::eTransferSrc
			);
			desc.persistent = true;
			return desc;
		}
	};

	// Stage 1 of the GLTF/GLB pipeline (GltfModel/GltfModelManager): one asset's vertex/meshlet
	// data, host-uploaded once at load time via graph::PredefinedBufferNode -- same mechanism as
	// MaterialBuffer above, not catalog-converted. Unlike the foliage archetype buffers above,
	// byte size depends on which asset was loaded, so there's no FrameContext-only Desc() a bare
	// tag struct could compute; GltfModelManager's PredefinedBufferNode members are each key's
	// sole Create<>(), with the real (data-size-derived, eShaderDeviceAddress-flagged) desc passed
	// explicitly to SetData() there. Every reader's Read<>() of these stays bare (no .Desc()) --
	// same reasoning as ParticleGridBuildNode's bare Read<ParticleTypeBuffer>() above, since
	// there's exactly one real producer per key here, not ParticleBuffer's dual-Modify case where
	// two touchers must agree on an identical explicit .Desc().
	struct GltfVertexBuffer {};
	struct GltfMeshletVertexBuffer {};
	struct GltfMeshletTriangleBuffer {};
	struct GltfMeshletBuffer {};

	// Stage 4 (GPU skinning): one mat4 per joint, in skin-joint order (GltfModel::EvaluatePose's
	// own output order), recomputed and re-uploaded every frame as the current animation plays --
	// unlike the other four GLTF buffers above, this one's *contents* change every frame even
	// though its size doesn't, same PredefinedBufferNode mechanism either way. Same non-catalog
	// reasoning as the others: joint count depends on which asset's skin was loaded.
	struct GltfJointMatrixBuffer {};

} // namespace brassica
