#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Node.hpp"
#include "passes/model/resources.hpp"
#include "passes/RenderPhases.hpp"
#include "vk_mem_alloc.h"
#include "VulkanCompat.hpp"

namespace brassica::graph {
	class Graph;
	class ResourceCatalog;
}

namespace brassica {

	class ModelRegistry;

	// One static triangle BLAS per registered decor model template -- ModelRegistry's own
	// model-local flat triangle index buffer (ModelRegistry::GetFlatTriangleIndices) paired with
	// GltfVertexBuffer's position field, read via a raw device address rather than a bindless
	// index. Built once per ModelRegistry content change (RebuildIfNeeded no-ops otherwise), and
	// referenced directly -- zero per-instance geometry duplication -- by far/rest-pose decor
	// instances when DecorTLAS's instance buffer is authored (model_decor_tlas_instances.comp).
	// Near/currently-animated instances get a separate per-slot refit BLAS fed by a skinned-
	// vertex-position compute pass; that pool is not managed here.
	//
	// Mirrors the deleted TerrainAccelerationStructure's ownership shape (plain members, own
	// command buffer/scratch/instance buffers, nothing routed through the frame graph except
	// bindless indices this class never needs) with one real difference: GltfVertexBuffer is a
	// graph-managed, growable ConfiguredExternalBuffer, not a buffer this class owns outright the
	// way terrain's AABB buffer was -- so RebuildIfNeeded must be driven from a real node's
	// Execute() (DecorTemplateBlasNode), with a real Read<GltfVertexBuffer>() graph edge, rather
	// than from Engine::Init() once.
	class DecorAccelerationStructure {
	public:
		DecorAccelerationStructure() = default;
		~DecorAccelerationStructure() { Destroy(); }

		DecorAccelerationStructure(const DecorAccelerationStructure&) = delete;
		DecorAccelerationStructure& operator=(const DecorAccelerationStructure&) = delete;

		void Init(
			vk::Device                      device,
			VmaAllocator                     allocator,
			const DispatchLoaderDynamic*     dls,
			std::span<const std::uint32_t>  concurrentFamilies
		) {
			m_device = device;
			m_allocator = allocator;
			m_dls = dls;
			m_concurrentFamilies.assign(concurrentFamilies.begin(), concurrentFamilies.end());
			InitTLAS();
		}

		void Destroy();

		// No-ops unless registry.GetVersion() has advanced since the last real rebuild. Copies
		// registry.GetVertices() into this class's own vertex-position buffer rather than reading
		// the shared catalog GltfVertexBuffer via device address -- see GetVertices()'s own
		// comment (ModelRegistry.hpp) for why. Also refreshes DecorTemplateBlasAddressBuffer's
		// pending upload (RegisterBufferNode) whenever templates actually rebuild.
		void RebuildIfNeeded(vk::CommandBuffer cmd, const ModelRegistry& registry);

		// 0 for a modelIndex with no built template yet (e.g. this frame's registry change
		// hasn't been picked up by RebuildIfNeeded() yet) -- callers should treat 0 as "skip this
		// instance," the same sentinel-by-absence convention terrain's old GetTLAS() used.
		[[nodiscard]] vk::DeviceAddress GetTemplateBlasAddress(std::uint32_t modelIndex) const {
			return modelIndex < m_templates.size() ? m_templates[modelIndex].blasAddress : 0;
		}

		// Same per-frame-rebuild shape as ModelRegistry::RegisterBufferNodes -- call every frame
		// alongside it; DecorTemplateBlasAddressBuffer only actually re-uploads on a real
		// template rebuild (RebuildIfNeeded's own SetData call), same as every other
		// PredefinedBufferNode here.
		void RegisterBufferNode(graph::Graph& frameGraph, graph::ResourceCatalog& catalog, const ModelRegistry& registry);

		// Rebuilds the TLAS's instance geometry from instanceBufferAddress (a DecorTLASInstanceBuffer
		// device address, kDecorCapacity hand-packed VkAccelerationStructureInstanceKHR entries --
		// dead slots must have instanceMask == 0, not be omitted, since this always builds exactly
		// kDecorCapacity primitives). The TLAS object itself is created once, eagerly, in Init()
		// (fixed size, kDecorCapacity instances) and never recreated -- only its buffer's contents
		// change frame to frame, same "same handle, rebuilt in place" shape
		// TerrainAccelerationStructure's own BLAS/TLAS used.
		void BuildTLAS(vk::CommandBuffer cmd, vk::DeviceAddress instanceBufferAddress);

		[[nodiscard]] vk::AccelerationStructureKHR GetTLAS() const { return m_tlas; }

	private:
		struct BufferResource {
			vk::Buffer        buffer{nullptr};
			VmaAllocation     allocation{VK_NULL_HANDLE};
			vk::DeviceAddress deviceAddress{0};
			void*             mapped{nullptr};
		};

		struct TemplateEntry {
			vk::AccelerationStructureKHR blas{nullptr};
			BufferResource               blasBuffer;
			vk::DeviceAddress            blasAddress{0};
		};

		void* CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, BufferResource& res, bool hostMapped);
		void  DestroyBuffer(BufferResource& res);
		void  DestroyTemplates();
		void  InitTLAS();

		vk::Device                  m_device{nullptr};
		VmaAllocator                m_allocator{VK_NULL_HANDLE};
		const DispatchLoaderDynamic* m_dls{nullptr};
		std::vector<std::uint32_t>  m_concurrentFamilies;

		BufferResource              m_vertexBuffer;
		BufferResource              m_flatIndexBuffer;
		BufferResource              m_scratchBuffer;
		std::vector<TemplateEntry>  m_templates;
		std::uint64_t               m_lastSeenVersion{~0ull};

		vk::AccelerationStructureKHR m_tlas{nullptr};
		BufferResource               m_tlasBuffer;
		BufferResource               m_tlasScratchBuffer;

		// lo/hi uint32 pairs, one per registered model, kept in sync with m_templates by
		// RebuildIfNeeded -- the pending-upload source for DecorTemplateBlasAddressBuffer.
		std::vector<glm::uvec2> m_blasAddressesForGpu;
		graph::PredefinedBufferNode<DecorTemplateBlasAddressBuffer, glm::uvec2, SubPhase::Prepare>
			m_blasAddressBufNode;
	};

} // namespace brassica
