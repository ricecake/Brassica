#pragma once

// Clouds-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass of
// the same restructure -- same Desc() output, just expressed as a list of constants instead of
// a free-function body.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	// CloudShadowGenNode now only rewrites 1/4 of texels per frame (Bayer-staggered update,
	// shaders/cloud_shadow_gen.comp) -- the other 3/4 must survive untouched across frames.
	// Lifetime::Persistent is what makes that safe: PhysicalRegistry::ProvisionTexture's pooled
	// allocation never re-Acquires once a desc-match keeps reusing the same handle across frames,
	// so without it the aliasing pool's lifetime bookkeeping stays frozen at frame 1's window and
	// will eventually hand this exact memory to an unrelated resource while still in use -- a
	// real, confirmed-class bug (see Execution.hpp's ResourceDesc::persistent doc comment and the
	// FoliageBiomeTableTexture precedent it was already fixed for). This was already latently
	// true before staggering, just invisible, since the old full-rewrite-every-frame behavior
	// erased any evidence of silent corruption.
	struct CloudShadowMap: FixedImage2D<CloudShadowMap, graph::Lifetime::Persistent> {
		static constexpr std::uint32_t kWidth = 1024;
		static constexpr std::uint32_t kHeight = 1024;
		static constexpr std::uint32_t kLayers = 5;
		static constexpr vk::Format    kFormat = vk::Format::eR16G16B16A16Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	// Single tileable detail-noise volume (periodic in all 3 axes, world-space period
	// constants::Weather::kCloudVolumePeriod) replacing the former 3-cascade camera-snapped
	// system. Big-picture cloud shape now comes entirely from the weather coverage map; this
	// volume only supplies reusable shape/erosion detail, so it never needs camera-relative
	// recentering -- see CloudVolumeGenNode (mip 0) and CloudVolumeDownsampleNode (mips 1-4).
	// Mips 0..4 (256/128/64/32/16) -- CloudVolumeDownsampleNode fills 1..4 from mip 0. Matches
	// the clamp cloud_render.frag's sampleCloudVolume() uses when picking a distance-based LOD
	// (mirroring boidish's own clamp(volumeMip, 0.0, 4.0)); mips beyond that would never be
	// sampled, so there's no reason to generate them.
	// Lifetime::Persistent -- same reasoning as TransmittanceLUT/MultiScatteringLUT
	// (passes/atmosphere/resources.hpp): CloudVolumeGenNode only bakes this once (dirty-flag
	// gated), so "this frame's producer is inactive" is the normal steady state, not "no valid
	// data." Without this, every reader (CloudShadowGenNode, CloudRenderNode) sees no active
	// writer preceding it once the node goes inactive after frame 1, and the zero-init path
	// clears the whole volume to black every frame -- exactly the vanishing-texture bug this was
	// found by.
	struct CloudDetailVolume: FixedImage3D<CloudDetailVolume, graph::Lifetime::Persistent> {
		static constexpr std::uint32_t kWidth = 256;
		static constexpr std::uint32_t kHeight = 256;
		static constexpr std::uint32_t kDepth = 256;
		static constexpr std::uint32_t kMips = 5;
		static constexpr vk::Format    kFormat = vk::Format::eR16Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	namespace detail {
		// Not a base itself -- CloudColor/CloudTransmittance/CloudDepth/CloudVelocity all share
		// this exact screen-sized rgba16f storage-image shape (ScreenImage2D itself,
		// resources/ResourceTypes.hpp); eTransferDst is what makes them safe to pair with
		// graph::Lifetime::History later (same usage set graph::ComputeStorageImageDesc used to
		// build for them by hand).
		struct CloudRaymarchOutputShape {
			static constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;
			static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		};
	} // namespace detail

	// Cloud raymarch outputs (CloudRenderNode, now a compute pass -- shaders/cloud_raymarch.comp)
	// -- CloudCompositeNode blends CloudColor/CloudTransmittance into HdrColor; CloudDepth/
	// CloudVelocity exist for the temporal reprojection pass (tile-based TAA plan, Stage B) to
	// consume.
	struct CloudColor: ScreenImage2D<CloudColor>, detail::CloudRaymarchOutputShape {};

	struct CloudTransmittance: ScreenImage2D<CloudTransmittance>, detail::CloudRaymarchOutputShape {};

	// Representative world-space hit distance (marchClouds()'s weightedDistSum/totalWeight,
	// boidish's own reprojection-depth formula) -- single channel, but the shared rgba16f shape
	// above is fine for a formatless-write target (see bindless.glsl's uImagesGenericWrite
	// comment); only .x is ever read.
	struct CloudDepth: ScreenImage2D<CloudDepth>, detail::CloudRaymarchOutputShape {};

	// Non-camera (wind-advection) screen-space velocity only -- camera motion is reprojected
	// separately in Stage B via uPreviousViewProjMatrix, matching boidish's own split between
	// cam_only_vel and a stored non-camera velocity texture.
	struct CloudVelocity: ScreenImage2D<CloudVelocity>, detail::CloudRaymarchOutputShape {};

	// Stage B (temporal reprojection + variance-clipped accumulation, CloudTemporalNode,
	// shaders/cloud_temporal.comp) -- these are both this frame's write target AND (via
	// ReadPrevious<K>()/graph::History<K>) last frame's read target, the same self-referential
	// ping-pong FoliageNode's ReadPrevious<HiZTexture>() precedent proves out (just against a
	// different node's output there, vs. this node's own prior output here -- confirmed safe via
	// PhysicalRegistry::ProvisionTemporalPairs's frameIndex-parity swap, which doesn't care which
	// node wrote either half of the pair). CloudCompositeNode reads these now, not CloudColor/
	// CloudTransmittance directly. Lifetime::Persistent for the same "inactive producer is the
	// normal steady state" reason as CloudDetailVolume above -- plain CRTP form, no desc.persistent
	// override needed: History<K> (ResourceKey.hpp) now forwards the base's kLifetime via
	// LifetimeOfK<T>(), so ZeroInitializeInactiveReads' Persistent exemption
	// (PhysicalExecutionBackend.hpp) correctly fires for a ReadPrevious<K>() of these too. (It
	// didn't used to -- see git history/feedback_history_lifetime_persistent_gotcha for the real
	// hardware bug this was found by, before History<K> forwarded lifetime at all.)
	struct CloudAccumulatedColor: ScreenImage2D<CloudAccumulatedColor, graph::Lifetime::Persistent>,
								   detail::CloudRaymarchOutputShape {};

	struct CloudAccumulatedTransmittance: ScreenImage2D<CloudAccumulatedTransmittance, graph::Lifetime::Persistent>,
										   detail::CloudRaymarchOutputShape {};

	// SVGF-style running moments: x = luma, y = luma^2 (Stage C's variance estimate is y - x*x),
	// z = historyLength (decayed on disocclusion/rejection, not just incremented), w = this
	// pixel's accumulated depth (carried forward for next frame's disocclusion comparison --
	// CloudDepth itself is always fresh every frame in Stages A-C, so there's no separate
	// "history depth" resource needed beyond this channel).
	struct CloudMoments: ScreenImage2D<CloudMoments, graph::Lifetime::Persistent>, detail::CloudRaymarchOutputShape {};

} // namespace brassica
