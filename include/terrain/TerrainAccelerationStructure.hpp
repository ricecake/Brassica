#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "IManager.hpp"
#include "vk_mem_alloc.h"
#include "VulkanCompat.hpp"

namespace brassica {

	class ComputeShader;

	namespace render {
		class PipelineLibrary;
	}

	struct TerrainAccelerationStructureState {
		bool enabled{true};

		auto GetReflection() const {
			return std::make_tuple(MakeField(
				"enabled",
				"Enable Terrain Acceleration Structure",
				&TerrainAccelerationStructureState::enabled
			));
		}
	};

	class TerrainAccelerationStructure
		: public ManagerBase<TerrainAccelerationStructure, TerrainAccelerationStructureState> {
	public:
		using State = TerrainAccelerationStructureState;

		TerrainAccelerationStructure() = default;

		~TerrainAccelerationStructure() override {
			if (m_initialized) {
				Shutdown();
			} else {
				DestroyAccelerationStructures();
			}
		}

		void Initialize() override { m_initialized = true; }

		void Shutdown() override {
			DestroyAccelerationStructures();
			m_initialized = false;
		}

		std::string GetManagerName() const override { return "TerrainAccelerationStructure"; }

		State GetState() const override { return State{m_enabled}; }

		void SetState(const State& state) override { m_enabled = state.enabled; }

		TerrainAccelerationStructure(const TerrainAccelerationStructure&) = delete;
		TerrainAccelerationStructure& operator=(const TerrainAccelerationStructure&) = delete;

		// concurrentFamilies: stored (not forwarded by-value as a one-shot param) because
		// InitAccelerationStructures can run again later via SetAllocator, below, with no other
		// way to learn it a second time. See BuildBufferCreateInfo's comment (PhysicalResource.hpp)
		// for what this is for -- TerrainTLAS and its backing buffer are the design plan's S4
		// out-of-registry CONCURRENT sites that live here.
		void Init(
			vk::Instance                   instance,
			vk::Device                     device,
			VmaAllocator                   alloc = VK_NULL_HANDLE,
			std::span<const std::uint32_t> concurrentFamilies = {}
		) {
			this->device = device;
			this->allocator = alloc;
			m_concurrentFamilies.assign(concurrentFamilies.begin(), concurrentFamilies.end());
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
			const glm::uvec4&        gridParams
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

		vk::Device                 device{nullptr};
		VmaAllocator               allocator{VK_NULL_HANDLE};
		DispatchLoaderDynamic      dls;
		std::vector<std::uint32_t> m_concurrentFamilies;

		BufferResource               aabbBuffer;
		BufferResource               blasBuffer;
		vk::AccelerationStructureKHR blas{nullptr};
		BufferResource               instanceBuffer;
		BufferResource               tlasBuffer;
		vk::AccelerationStructureKHR tlas{nullptr};
		BufferResource               scratchBuffer;
		glm::vec3                    lastASCameraPos{1e9f, 1e9f, 1e9f};
		bool                         m_enabled{true};
	};

} // namespace brassica
