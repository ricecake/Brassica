#pragma once

// Foliage-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "passes/foliage/IFoliageManager.hpp"
#include "resources/ResourceTypes.hpp"

#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"

namespace brassica {

	// Per-biome foliage properties (color/height/width/density/flowerRatio/...), one texel-row
	// per Whittaker biome index (shaders/helpers/whittaker.glsl, 0-9) -- FoliageManager-owned,
	// written host-side via a PredefinedTextureNode whenever the UI changes a biome's properties
	// (FoliageManager::SyncTableNode, which keeps building and passing its own copy of this exact
	// desc to SetData() -- harmless, since Provision()'s catalog-always-wins rule means this
	// Desc() is what actually reaches ProvisionTexture regardless, and the two are identical so
	// no disagreement warning fires either). 5 texels wide x kFoliageBiomeCount rows, RGBA32F.
	//
	// ExternalImage2D (Lifetime::External, resources/ResourceTypes.hpp): this key's real
	// producer is a host upload with no per-frame regeneration concept at all (unlike a
	// throttled LUT), not a graph node that sometimes runs and sometimes doesn't. The base's own
	// `.persistent = true` is what actually protects this texture's real content
	// (ZeroInitializeInactiveReads, PhysicalExecutionBackend.hpp, exempts any physical resource
	// with persistent=true -- or, now, Lifetime::Persistent/External -- and HasDefinedContents()
	// -- see 0469e96, the real hardware bug this exact exemption fixed).
	struct FoliageBiomeTableTexture: ExternalImage2D<FoliageBiomeTableTexture> {
		static constexpr std::uint32_t kWidth = 5;
		static constexpr std::uint32_t kHeight = kFoliageBiomeCount;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

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
	struct FoliageBakeDummyTarget: ScreenImage2D<FoliageBakeDummyTarget> {
		static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eColorAttachment |
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
	};

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

} // namespace brassica
