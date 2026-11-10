#pragma once

// Cross-feature graph resource-key tags (design plan's restructure, S1): keys delivered through
// the global frame-set/bindless plumbing rather than owned by one render feature -- the final
// swapchain target, the GBuffer/HdrColor intermediates every feature reads or writes, and the
// frame-set-bound globals (ClusteredLighting/AutoExposureBuffer). A feature-owned
// key that merely happens to be read by another feature (e.g. TerrainTLAS, GTAOTexture) stays in
// its owning feature's own passes/<feature>/resources.hpp instead -- see that header's own
// comment for the distinction. Split out of the old monolithic passes/ResourceKeys.hpp (S1);
// S2 then catalog-converted every key below and retrofitted them onto the declarative CRTP bases
// (resources/ResourceTypes.hpp) -- see passes/clouds/resources.hpp's comment for why.

#include "vulkan/vulkan.hpp"

#include "passes/lighting/Light.hpp"
#include "resources/ResourceTypes.hpp"

namespace brassica {

	// Provision() skips an imported key's realization entirely regardless of desc (
	// IsImported() early-continue, PhysicalRegistry.hpp), so this Desc() is never consulted for
	// provisioning -- only for EffectiveDesc's same-stage layout check (PhysicalRegistry.hpp),
	// which otherwise has to guess kind=Image2D from the ResourceDesc{} default. Correct for an
	// image like this one by accident; see ClusteredLighting/AutoExposureBuffer below for
	// where that same accident was actually wrong (kind=Buffer).
	struct Swapchain: ImportedImage2D {};

	struct HdrColor: ScreenImage2D<HdrColor> {
		static constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eColorAttachment |
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
	};

	struct GBufferPosition: ScreenImage2D<GBufferPosition> {
		static constexpr vk::Format kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = HdrColor::kUsage;
	};

	struct GBufferNormal: ScreenImage2D<GBufferNormal> {
		static constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = HdrColor::kUsage;
	};

	struct GBufferAlbedo: ScreenImage2D<GBufferAlbedo> {
		static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Srgb;
		static constexpr vk::ImageUsageFlags kUsage = HdrColor::kUsage;
	};

	struct GBufferMaterial: ScreenImage2D<GBufferMaterial> {
		static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;
		static constexpr vk::ImageUsageFlags kUsage = HdrColor::kUsage;
	};

	// eD32SfloatS8Uint, not plain eD32Sfloat: a shader-written stencil value (render::StencilMode)
	// needs a real stencil aspect to write into -- see PhysicalTexture's depth-only sampled view
	// (PhysicalResource.hpp) for the consequence this has on every existing ctx.Index<GBufferDepth>()
	// sampled read, which can no longer use the same combined-aspect view the attachment binding
	// uses (a sampled descriptor must have exactly one aspect).
	struct GBufferDepth: ScreenImage2D<GBufferDepth> {
		static constexpr vk::Format kFormat = vk::Format::eD32SfloatS8Uint;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment |
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	// Frame set binding 3 (shaders/lighting.glsl's uClusters) -- real allocation and import still
	// live in Engine.cpp's clusterGridBuffers[]/RegisterImportedBuffer (S0's minimal fix for the
	// ClusteredLighting phantom-resource bug); the real fix that makes this a genuine Transient
	// catalog key instead is S7's. ImportedBuffer (kind=Buffer), not a bare tag: a bare `{}`
	// defaults to ResourceDesc{}'s kind=Image2D, which is wrong for a buffer and is exactly the
	// placeholder-desc trap the same-stage layout check (PhysicalRegistry.hpp) warns about.
	struct ClusteredLighting: ImportedBuffer {};

	// AutoExposureBuffer is imported (Engine::DrawFrame's RegisterImportedBuffer) -- Provision
	// skips an imported key's realization entirely regardless of its desc, so the only thing this
	// Desc() needs to get right is the kind, same reasoning as ClusteredLighting above. Real
	// catalog conversion (Lifetime::Persistent, no Create()) is S7's.
	struct AutoExposureBuffer: ImportedBuffer {};

	// Split from the old single hybrid LightsBuffer (CPU upload + GPU firefly/lightning writes
	// landing on top of the same buffer every frame, the "weird round trips" TODO.md names) into
	// two real, bindless-indexed catalog buffers with no CPU/GPU overlap at all:
	//
	// CpuLightsBuffer: host-authored only (sun, moon, whatever LightManager::AddLight adds).
	// Sole producer is LightManager's own PredefinedBufferNode (LightManager::RegisterBufferNode),
	// uploaded only when LightManager's dirty tracking says something actually changed. Never
	// written by any shader.
	//
	// GpuLightsBuffer: GPU-authored only. GpuLightsResetNode Creates it fresh every frame
	// (zeroing contents, resetting the claim counter); ParticleBehaviorNode's fireflies then
	// atomic-claim a slot in [0, GPU_LIGHT_DYNAMIC_CAPACITY), and LightningAgeNode writes its own
	// static reserved range above that -- both Modify<>() it, ordered after the reset via the
	// normal Create-before-accessor rule. The CPU never touches this buffer at all.
	struct CpuLightsBuffer: ExternalBuffer<CpuLightsBuffer> {
		static constexpr std::uint64_t        kByteSize = sizeof(CpuLightsSSBOData);
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

	struct GpuLightsBuffer: FixedBuffer<GpuLightsBuffer> {
		static constexpr std::uint64_t        kByteSize = sizeof(GpuLightsSSBOData);
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

} // namespace brassica
