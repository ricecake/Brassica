#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "vk_mem_alloc.h"
#include "VulkanCompat.hpp"

namespace brassica {

	class ComputeShader;

	namespace render {
		class PipelineLibrary;
	}

	class TerrainAccelerationStructure {
	public:
		TerrainAccelerationStructure() = default;

		~TerrainAccelerationStructure() { DestroyAccelerationStructures(); }

		TerrainAccelerationStructure(const TerrainAccelerationStructure&) = delete;
		TerrainAccelerationStructure& operator=(const TerrainAccelerationStructure&) = delete;

		void Init(vk::Instance instance, vk::Device device, VmaAllocator alloc = VK_NULL_HANDLE) {
			this->device = device;
			this->allocator = alloc;
			dls.init(instance, device);
			if (allocator != VK_NULL_HANDLE) {
				InitAccelerationStructures();
			}
		}

		void InitAccelerationStructures();

		void SetAllocator(VmaAllocator alloc) {
			this->allocator = alloc;
			if (allocator != VK_NULL_HANDLE && !tlas) {
				InitAccelerationStructures();
			}
		}

		void BuildOrUpdate(
			vk::CommandBuffer        cmd,
			const glm::vec3&         cameraPos,
			float                    baseTexelSize,
			uint32_t                 numLODs,
			render::PipelineLibrary* pipelineLibrary,
			ComputeShader*           aabbShader,
			vk::DescriptorSet        frameSet,
			vk::DescriptorSet        globalSet,
			vk::DescriptorSetLayout  frameSetLayout,
			vk::DescriptorSetLayout  globalSetLayout,
			const glm::uvec4&        gridParams,
			const glm::uvec4&        lodOffsets0_3,
			const glm::uvec4&        lodOffsets4_7,
			const glm::uvec4&        lodOffsets8_11 = glm::uvec4(0u)
		);

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
		VmaAllocator          allocator{VK_NULL_HANDLE};
		DispatchLoaderDynamic dls;

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
