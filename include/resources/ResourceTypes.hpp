#pragma once

// CRTP bases for the declarative resource-key shapes (design plan's restructure, S2) --
// sgreenhusted's call on notes.md's original sketch: even where a key's Desc() already has a
// DRY, shared-free-function body (e.g. CloudShadowMapDesc(), TonemapMipDesc()), a short list of
// `kXxx` constants reads faster and is easier to add to/maintain than a function-call body,
// which is the actual goal here -- not deduplication, which those free functions already had.
//
// Naming: statics are `kXxx` (never a bare instance member with the same name a derived type's
// own static might shadow -- see notes.md's original CRTP sketch and its name-shadowing bug).
// Instance members are reserved for genuine per-frame config an owning manager publishes (e.g.
// TerrainLODImage2D's numLODs below) -- nothing here is Vulkan-free; this header lives beside
// resources/core.hpp, not graph/, same as the catalog itself.

#include <algorithm>
#include <cstdint>

#include "vulkan/vulkan.hpp"

#include "constants.h"
#include "graph/Execution.hpp"

namespace brassica {

	// A fixed-size 2D image: every dimension/format/usage is a type-level constant. Derived
	// provides kWidth, kHeight, kFormat (vk::Format), kUsage (vk::ImageUsageFlags); kMips/kLayers
	// default to 1 (overridable) since most fixed images are single-mip, single-layer.
	template <class Derived, graph::Lifetime L = graph::Lifetime::Transient>
	struct FixedImage2D {
		static constexpr graph::Lifetime kLifetime = L;
		static constexpr std::uint32_t   kMips = 1;
		static constexpr std::uint32_t   kLayers = 1;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = Derived::kWidth,
				.height = Derived::kHeight,
				.mips = Derived::kMips,
				.layers = Derived::kLayers,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
			};
		}
	};

	// A fixed-size 3D volume image. Derived provides kWidth/kHeight/kDepth, kFormat, kUsage.
	template <class Derived, graph::Lifetime L = graph::Lifetime::Transient>
	struct FixedImage3D {
		static constexpr graph::Lifetime kLifetime = L;
		static constexpr std::uint32_t   kMips = 1;
		static constexpr std::uint32_t   kLayers = 1;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image3D,
				.width = Derived::kWidth,
				.height = Derived::kHeight,
				.depth = Derived::kDepth,
				.mips = Derived::kMips,
				.layers = Derived::kLayers,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
			};
		}
	};

	// A screen-sized 2D image: width/height always track the frame's real render target size
	// exactly (no scale factor -- nothing currently needs one; add kScaleNum/kScaleDen here if a
	// future key does). Derived provides kFormat, kUsage.
	template <class Derived, graph::Lifetime L = graph::Lifetime::Transient>
	struct ScreenImage2D {
		static constexpr graph::Lifetime kLifetime = L;
		static constexpr std::uint32_t   kMips = 1;
		static constexpr std::uint32_t   kLayers = 1;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = ctx.width,
				.height = ctx.height,
				.mips = Derived::kMips,
				.layers = Derived::kLayers,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
			};
		}
	};

	// One mip of TonemapComputeNode's bloom/local-tonemap chain: mip N is ctx.width/2 (then
	// height) halved N more times, floored to 1 -- same formula every one of the 16 keys using
	// this uses, just a different (type-level) mip index. Derived provides kMipIndex, kFormat,
	// kUsage.
	template <class Derived, graph::Lifetime L = graph::Lifetime::Transient>
	struct ScreenMipImage2D {
		static constexpr graph::Lifetime kLifetime = L;
		static constexpr std::uint32_t   kMips = 1;
		static constexpr std::uint32_t   kLayers = 1;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			const std::uint32_t w = std::max(1u, (ctx.width / 2) >> Derived::kMipIndex);
			const std::uint32_t h = std::max(1u, (ctx.height / 2) >> Derived::kMipIndex);
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = w,
				.height = h,
				.mips = Derived::kMips,
				.layers = Derived::kLayers,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
			};
		}
	};

	// The terrain clipmap/min-max/biome-index family: MapDim x MapDim, numLODs layers -- numLODs
	// is genuine per-frame config (TerrainGenNode::SetFrameParams publishes it onto this exact
	// instance, same as every other key's config field), not a type-level constant, which is why
	// this base takes a real member instead of a kLayers constant like the others above. Derived
	// provides kMips, kFormat, kUsage.
	//
	// Deliberately Lifetime::Transient, not Persistent, despite wanting the same alias-pool
	// bypass Persistent gives (this image must survive indefinitely once allocated -- see
	// ResourceDesc::persistent's own comment) -- confirmed the hard way: TerrainGenNode Creates
	// this every frame and TerrainMinMaxDownsampleNode Modifies it later the SAME frame, a real
	// same-frame producer/consumer pair, not "last frame's cached content is an acceptable
	// substitute" (which is what Lifetime::Persistent actually means -- see
	// RelationalCompiler.hpp's Create-before-accessor comment: persistent keys skip that rule
	// entirely, which silently turned this into two unordered writers the moment it was tried).
	// `.persistent = true` on the desc itself is the correct, narrower tool here: it drives only
	// the alias-pool bypass and the zero-init-clear exemption (PhysicalRegistry.hpp/
	// PhysicalExecutionBackend.hpp), not RelationalCompiler's scheduling-edge logic.
	template <class Derived>
	struct TerrainLODImage2D {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Transient;

		std::uint32_t numLODs = constants::Class::Terrain::DefaultMaxLODs;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = constants::Class::Terrain::MapDim,
				.height = constants::Class::Terrain::MapDim,
				.mips = Derived::kMips,
				.layers = numLODs,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.persistent = true,
			};
		}
	};

	// TerrainBiomeNode's 7 ping-pong climate-sim textures: kDim x kDim, single layer, same
	// format/usage for all seven (only the dimension constant differs, and even that's 2048 for
	// every one of them today -- kept as separate type-level constants per key rather than one
	// shared value, since nothing says they have to stay equal). Same Lifetime::Transient +
	// .persistent=true reasoning as TerrainLODImage2D just above: TerrainBiomeNode Modifies all
	// seven every frame, and at least TerrainWeatherBiomeTexture has real same-frame readers
	// (CloudShadowGenNode, CloudRenderNode, FoliageNode, DeferredNode) that need a real
	// Create-before-accessor edge, not Lifetime::Persistent's "last frame's content is fine."
	// Derived provides kDim.
	template <class Derived>
	struct WeatherBiomeImage2D {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Transient;
		static constexpr std::uint32_t   kLayers = 1;
		static constexpr vk::Format      kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = Derived::kDim,
				.height = Derived::kDim,
				.layers = kLayers,
				.formatCode = static_cast<std::uint32_t>(kFormat),
				.usageMask = static_cast<std::uint32_t>(kUsage),
				.persistent = true,
			};
		}
	};

	// A fixed-size buffer: byteSize/usage/hostAccess are all type-level constants. Derived
	// provides kByteSize, kUsage (vk::BufferUsageFlags), kHostAccess (graph::HostAccess,
	// defaults to None).
	template <class Derived, graph::Lifetime L = graph::Lifetime::Transient>
	struct FixedBuffer {
		static constexpr graph::Lifetime kLifetime = L;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::None;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Buffer,
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.byteSize = Derived::kByteSize,
				.hostAccess = Derived::kHostAccess,
			};
		}
	};

	// A buffer whose byteSize is genuine per-frame config (e.g. maxParticles * sizeof(Particle)),
	// published onto the catalog instance by whichever node owns that sizing field -- same
	// pattern as TerrainLODImage2D's numLODs, just for a buffer instead of an image. Lifetime::
	// Transient, not Persistent, same reasoning as TerrainLODImage2D/WeatherBiomeImage2D: every
	// current user has a genuine same-frame producer/consumer pair (e.g. ParticleLivenessNode
	// Creates AboveWaterParticleAliveBuffer, ParticleBehaviorNode Reads it the same frame) --
	// Lifetime::Persistent would skip RelationalCompiler's Create-before-accessor rule entirely,
	// the exact scheduling bug this migration already hit once on the terrain keys.
	// desc.persistent=true gets the narrower effect actually wanted: alias-pool bypass (this
	// buffer's content must survive frame to frame once allocated) + the zero-init-clear
	// exemption, nothing about scheduling. Derived provides kUsage; kHostAccess defaults to None
	// (GPU-only, no host up/download -- unlike ExternalBuffer/ConfiguredExternalBuffer above).
	template <class Derived>
	struct ConfiguredBuffer {
		static constexpr graph::Lifetime   kLifetime = graph::Lifetime::Transient;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::None;

		std::uint64_t byteSize = 0;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Buffer,
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.byteSize = byteSize,
				.hostAccess = kHostAccess,
				.persistent = true,
			};
		}
	};

	// Delivered through the global frame-set/bindless plumbing or a dedicated out-of-band
	// import, not provisioned by Provision() at all (PhysicalResourceRegistry::Provision's own
	// IsImported() early-continue skips it regardless of what this Desc() returns). This Desc()
	// only ever matters for EffectiveDesc's same-stage layout check -- see
	// resources/core.hpp's Swapchain for the longer version of this comment. No width/height/
	// format/usage is meaningful, same reasoning as AccelerationStructureDesc().
	template <graph::ResourceDesc::Kind K>
	struct Imported {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::Imported;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{.kind = K};
		}
	};

	using ImportedImage2D = Imported<graph::ResourceDesc::Kind::Image2D>;
	using ImportedBuffer = Imported<graph::ResourceDesc::Kind::Buffer>;
	using ImportedAccelerationStructure = Imported<graph::ResourceDesc::Kind::AccelerationStructure>;

	// A fixed-size 2D image whose real content comes from a host upload with no per-frame
	// regeneration concept (a PredefinedTextureNode), not a graph node that sometimes runs and
	// sometimes doesn't -- Lifetime::External, not Persistent; see
	// passes/foliage/resources.hpp's FoliageBiomeTableTexture for the full reasoning. Derived
	// provides kWidth, kHeight, kFormat, kUsage; kMips/kLayers default to 1.
	template <class Derived>
	struct ExternalImage2D {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::External;
		static constexpr std::uint32_t   kMips = 1;
		static constexpr std::uint32_t   kLayers = 1;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::Staged;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = Derived::kWidth,
				.height = Derived::kHeight,
				.mips = Derived::kMips,
				.layers = Derived::kLayers,
				.formatCode = static_cast<std::uint32_t>(Derived::kFormat),
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.hostAccess = Derived::kHostAccess,
				// Belt-and-suspenders alongside Lifetime::External's own widened exemption
				// (ZeroInitializeInactiveReads/ProvisionTexture, PhysicalExecutionBackend.hpp/
				// PhysicalRegistry.hpp) -- redundant today, deliberately, since this is exactly
				// the 0469e96 bug class and I can't re-verify the clear-skip path on real
				// hardware myself this session.
				.persistent = true,
			};
		}
	};

	// A fixed-size buffer whose real content comes from a host upload (a PredefinedBufferNode) --
	// same Lifetime::External reasoning as ExternalImage2D above. Derived provides kByteSize,
	// kUsage; kHostAccess defaults to Staged (every current user matches
	// graph::StagedStorageBufferDesc's shape).
	template <class Derived>
	struct ExternalBuffer {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::External;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::Staged;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Buffer,
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.byteSize = Derived::kByteSize,
				.hostAccess = Derived::kHostAccess,
				// See ExternalImage2D's identical comment on why this stays set alongside
				// Lifetime::External's own widened exemption.
				.persistent = true,
			};
		}
	};

	// Same as ExternalBuffer, but byteSize is genuine runtime config an owning manager publishes
	// (e.g. MaterialManager::RegisterBufferNode, once the model-loading system can report a real
	// material count up front) rather than a type-level constant -- see passes/material/
	// resources.hpp's MaterialBuffer. Derived provides kUsage; kHostAccess defaults to Staged.
	template <class Derived>
	struct ConfiguredExternalBuffer {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::External;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::Staged;

		std::uint64_t byteSize = 0;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			return graph::ResourceDesc{
				.kind = graph::ResourceDesc::Kind::Buffer,
				.usageMask = static_cast<std::uint32_t>(Derived::kUsage),
				.byteSize = byteSize,
				.hostAccess = Derived::kHostAccess,
				// See ExternalImage2D's identical comment on why this stays set alongside
				// Lifetime::External's own widened exemption.
				.persistent = true,
			};
		}
	};

} // namespace brassica
