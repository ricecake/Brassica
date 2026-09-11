#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "vk_mem_alloc.h"
#include "VulkanCompat.hpp"

namespace brassica {

	// What's left of the old TerrainPass once its pipeline/shader/descriptor-set ownership moved
	// to TerrainNode + Engine::pipelineLibrary (Node/Pass unification, Stage 6): the terrain
	// BLAS/TLAS build, unchanged from before that migration -- own transient command pool/queue,
	// own camera-movement throttle, own synchronous device.waitIdle(). Not a "pass" anymore (it
	// owns no pipeline), hence the rename and the move into terrain/ alongside TerrainClipmap.
	//
	// Default-constructed, then wired via Init() once a real instance/device exist -- mirrors
	// PhysicalResourceRegistry/PipelineLibrary's own default-then-wire pattern (Engine.hpp).
	class TerrainAccelerationStructure {
	public:
		TerrainAccelerationStructure() = default;

		~TerrainAccelerationStructure() { DestroyAccelerationStructures(); }

		TerrainAccelerationStructure(const TerrainAccelerationStructure&) = delete;
		TerrainAccelerationStructure& operator=(const TerrainAccelerationStructure&) = delete;

		void Init(vk::Instance instance, vk::Device device) {
			this->device = device;
			dls.init(instance, device);
		}

		void BuildOrUpdate(VmaAllocator allocator, const glm::vec3& cameraPos, float baseTexelSize, uint32_t numLODs);

		// Explicit, not just left to the destructor -- must run while device is still a live
		// handle, called from Engine::Cleanup() before device.destroy() (mirroring
		// PhysicalResourceRegistry::Reset()'s own placement there).
		void DestroyAccelerationStructures();

		[[nodiscard]] vk::AccelerationStructureKHR GetTLAS() const { return tlas; }

		[[nodiscard]] const DispatchLoaderDynamic& GetDls() const { return dls; }

	private:
		struct BufferResource {
			vk::Buffer        buffer{nullptr};
			VmaAllocation     allocation{VK_NULL_HANDLE};
			vk::DeviceAddress deviceAddress{0};
		};

		vk::Device            device{nullptr};
		DispatchLoaderDynamic dls;
		VmaAllocator          lastAllocator{VK_NULL_HANDLE};

		BufferResource               aabbBuffer;
		BufferResource               blasBuffer;
		vk::AccelerationStructureKHR blas{nullptr};
		BufferResource               instanceBuffer;
		BufferResource               tlasBuffer;
		vk::AccelerationStructureKHR tlas{nullptr};
		BufferResource               scratchBuffer;
		glm::vec3                    lastASCameraPos{1e9f, 1e9f, 1e9f};
	};

} // namespace brassica
