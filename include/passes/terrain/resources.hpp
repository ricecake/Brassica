#pragma once

// Terrain-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "graph/PhysicalResource.hpp"
#include "resources/ResourceTypes.hpp"

namespace brassica {

	// The result of TerrainPass::BuildOrUpdateAccelerationStructure, registered into the graph
	// as an Imported AccelerationStructure resource. The build itself stays entirely out-of-band
	// (TerrainPass's own transient command pool/queue, unchanged); only the result flows through
	// the graph. Both real touchers (TerrainGenNode, DeferredNode) used to need their own
	// explicit .Desc(graph::AccelerationStructureDesc()) escape-hatch call, since Provision()'s
	// AccelerationStructure detection used to switch on a realization's own raw, pre-resolution
	// desc.kind; now checked against the registry's own m_accelStructs map directly instead
	// (PhysicalRegistry.hpp's Provision), so this key needs no escape hatch anywhere.
	struct TerrainTLAS: ImportedAccelerationStructure {};

	// mips = 1: this image has no mip chain of its own -- only TerrainMinMaxTexture below does.
	struct TerrainClipmapShape {
		static constexpr std::uint32_t kMips = 1;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eSampled |
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
	};

	// The terrain heightmap array -- a normal pass-owned persistent resource, Created<> by
	// TerrainGenNode every frame, like any other Lifetime::Persistent resource (re-asserting the
	// same desc every frame is a no-op past the first -- ProvisionTexture's desc-match early
	// return leaves the real resource untouched). numLODs is published by TerrainGenNode::
	// SetFrameParams (it tracks terrainGridParams.x, the one piece of runtime config this desc
	// needs beyond FrameContext). brassica::TerrainManager (passes/terrain/TerrainManager.hpp) is
	// unrelated to this image -- it's just the CPU-side manager (numLODs/baseTexelSize config,
	// Regenerate(), the async ground-height readback).
	struct TerrainClipmapTexture: TerrainLODImage2D<TerrainClipmapTexture>, TerrainClipmapShape {};

	// floor(log2(MapDim)) + 1 -- the full mip chain down to 1x1 for a 1024-wide image (mips
	// 0-10), well under SPD's 12-mip-per-dispatch limit. RG32F, not RGBA32F: only min (x) and
	// max (y) are meaningful, and the per-mip storage views this needs (PhysicalTexture::
	// CreateView, PhysicalResource.hpp) are what the broken mips=1 claim on this key used to mask
	// -- nothing ever read past mip 0 before.
	struct TerrainMinMaxTexture: TerrainLODImage2D<TerrainMinMaxTexture> {
		static constexpr std::uint32_t kMips = 11;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = TerrainClipmapShape::kUsage;
	};

	// The terrain's own per-LOD biome index map, Created<> by TerrainGenNode -- distinct from
	// TerrainBiomeMainTexture (passes/biome/resources.hpp), which TerrainBiomeNode owns and
	// writes in a completely different (single-layer, ping-ponged) shape.
	struct TerrainBiomeTexture: TerrainLODImage2D<TerrainBiomeTexture>, TerrainClipmapShape {};

	// TerrainManager's host-visible readback staging buffer (TerrainReadbackNode's Modify<>,
	// passes/terrain/TerrainManager.hpp) -- imported, same reasoning as AutoExposureBuffer/
	// LightsBuffer (resources/core.hpp): the real allocation is owned and sized by
	// TerrainManager itself (1MB, comfortably covering the 128x128 RGBA32F window this is
	// actually used for), registered once at Engine::InitVulkan time since (unlike those
	// per-FRAME_OVERLAP-slot buffers) this one handle never changes.
	struct TerrainReadbackStaging: ImportedBuffer {};

	struct TerrainShadowMap: FixedImage2D<TerrainShadowMap, graph::Lifetime::Persistent> {
		static constexpr std::uint32_t kWidth = 1024;
		static constexpr std::uint32_t kHeight = 1024;
		static constexpr std::uint32_t kLayers = 3;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eStorage |
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	// GPU Chunk vertex representation: position (xyz) + height (w), normal (xyz) + biome (w), and material/substrate attributes
	struct TerrainChunkVertexGPU {
		glm::vec4 position; // xyz = position, w = height
		glm::vec4 normal;   // xyz = normal, w = biome
		glm::vec4 material; // x = waterLevel, y = ridgeMap, z = substrate, w = unused
	};

	static_assert(sizeof(TerrainChunkVertexGPU) == 48, "TerrainChunkVertexGPU size must be 48 bytes");

	// Terrain Page Table entry: physicalSlot (4B), coordX (4B), coordZ (4B), lod (4B) = 16 bytes
	struct TerrainPageTableEntryGPU {
		std::uint32_t physicalSlot{0xFFFFFFFFu};
		std::int32_t  coordX{0x7FFFFFFF};
		std::int32_t  coordZ{0x7FFFFFFF};
		std::uint32_t lod{0xFFFFFFFFu};
	};
	static_assert(sizeof(TerrainPageTableEntryGPU) == 16, "TerrainPageTableEntryGPU size must be 16 bytes");

	// Terrain Page Table buffer: maps virtual chunk IDs (maxVirtualChunks) to physical slot indices and coordinates
	struct TerrainPageTableBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			graph::ResourceDesc desc = graph::StorageBufferDesc(
				static_cast<std::uint64_t>(constants::Class::Terrain::ChunkManager::MaxVirtualChunks) * sizeof(TerrainPageTableEntryGPU)
			);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			desc.persistent = true;
			return desc;
		}
	};

	// Terrain Chunk LOD Buffer: holds target LOD evaluation per virtual chunk slot
	struct TerrainChunkLodBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			graph::ResourceDesc desc = graph::StorageBufferDesc(
				static_cast<std::uint64_t>(constants::Class::Terrain::ChunkManager::MaxVirtualChunks) * sizeof(std::uint32_t)
			);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			desc.persistent = true;
			return desc;
		}
	};

	// Terrain Indirection Buffer: tracks allocation counters, free slot stack, and active chunk instances
	struct TerrainIndirectionBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			const std::uint64_t byteSize = 64 +
				static_cast<std::uint64_t>(constants::Class::Terrain::ChunkManager::MaxAllocatedChunks) * sizeof(std::uint32_t) * 2;
			graph::ResourceDesc desc = graph::StorageBufferDesc(byteSize);
			desc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			desc.persistent = true;
			return desc;
		}
	};

	// Terrain Chunk Requirements Buffer: stores requested chunks and target LOD identified by LOD selection pass
	struct TerrainChunkRequirementsBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			const std::uint64_t byteSize = 16 +
				static_cast<std::uint64_t>(constants::Class::Terrain::ChunkManager::MaxVirtualChunks) * sizeof(std::uint32_t);
			graph::ResourceDesc desc = graph::StorageBufferDesc(byteSize);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			desc.persistent = true;
			return desc;
		}
	};

	// Terrain Chunk Vertex Buffer: holds vertices for all allocated physical chunk slots
	struct TerrainChunkVertexBuffer {
		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext&) const {
			const std::uint64_t byteSize = static_cast<std::uint64_t>(constants::Class::Terrain::ChunkManager::MaxAllocatedChunks) *
				constants::Class::Terrain::ChunkManager::VerticesPerChunk * sizeof(TerrainChunkVertexGPU);
			graph::ResourceDesc desc = graph::StorageBufferDesc(byteSize);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			desc.persistent = true;
			return desc;
		}
	};

} // namespace brassica
