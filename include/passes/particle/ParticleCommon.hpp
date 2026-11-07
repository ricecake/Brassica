#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"

#include "graph/PhysicalRegistry.hpp"
#include "passes/particle/resources.hpp"

// Shared infrastructure for the particle node cluster (passes/particle/nodes/*.hpp) -- split out
// of the old monolithic ParticleSystemNode.hpp (S1, design plan's restructure) specifically
// because it's genuinely shared across 6 of the 7 node files, not just incidentally duplicated:
// every node but ParticleTypeBufferNode calls detail::RefreshParticleDescriptorSet, and the two
// render nodes share ParticleRenderPushConstants.
namespace brassica {

	// Forward-declared only: Before<WaterNode>()/After<WaterNode>() and the Composite-phase
	// writer-chain relations in UnderwaterParticleRenderNode.hpp/AboveWaterParticleRenderNode.hpp
	// need just the type's name -- see ClusterLightAssignmentNode.hpp's identical reasoning.
	struct WaterNode;
	struct VolumetricRenderNode;
	struct CirrusNode;
	struct CloudCompositeNode;

	struct ParticleRenderPushConstants {
		std::uint32_t isUnderwater{0};
	};

	inline void UpdateParticleDescriptorSet(
		vk::Device                             device,
		vk::DescriptorSet                      particleSet,
		const graph::PhysicalResourceRegistry* registry
	) {
		if (!registry || !particleSet || !device)
			return;

		auto pBuf = registry->GetBuffer<ParticleBuffer>();
		auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
		auto pAboveAliveBuf = registry->GetBuffer<AboveWaterParticleAliveBuffer>();
		auto pAboveIndirectBuf = registry->GetBuffer<AboveWaterParticleIndirectBuffer>();
		auto pUnderAliveBuf = registry->GetBuffer<UnderwaterParticleAliveBuffer>();
		auto pUnderIndirectBuf = registry->GetBuffer<UnderwaterParticleIndirectBuffer>();
		auto pGridHeadsBuf = registry->GetBuffer<ParticleGridHeadsBuffer>();
		auto pGridNextBuf = registry->GetBuffer<ParticleGridNextBuffer>();

		if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf ||
		    !pGridHeadsBuf || !pGridNextBuf)
			return;

		vk::Buffer b0 = pBuf->GetBuffer();
		vk::Buffer b1 = pTypeBuf->GetBuffer();
		vk::Buffer b2 = pAboveAliveBuf->GetBuffer();
		vk::Buffer b3 = pAboveIndirectBuf->GetBuffer();
		vk::Buffer b4 = pUnderAliveBuf->GetBuffer();
		vk::Buffer b5 = pUnderIndirectBuf->GetBuffer();
		vk::Buffer b6 = pGridHeadsBuf->GetBuffer();
		vk::Buffer b7 = pGridNextBuf->GetBuffer();

		if (!b0 || !b1 || !b2 || !b3 || !b4 || !b5 || !b6 || !b7)
			return;

		std::array<vk::DescriptorBufferInfo, 8> bufferInfos{
			vk::DescriptorBufferInfo{b0, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b1, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b2, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b3, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b4, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b5, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b6, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b7, 0, VK_WHOLE_SIZE}
		};

		std::array<vk::WriteDescriptorSet, 8> writes{};
		for (uint32_t i = 0; i < 8; ++i) {
			writes[i]
				.setDstSet(particleSet)
				.setDstBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferInfos[i]);
		}

		device.updateDescriptorSets(writes, nullptr);
	}

	namespace detail {

		struct ParticleDescriptorSet {
			vk::DescriptorSetLayout layout{nullptr};
			vk::DescriptorPool      pool{nullptr};
			vk::DescriptorSet       set{nullptr};
		};

		inline ParticleDescriptorSet CreateParticleDescriptorSet(vk::Device device) {
			ParticleDescriptorSet result;

			std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
			for (uint32_t i = 0; i < 8; ++i) {
				bindings[i]
					.setBinding(i)
					.setDescriptorType(vk::DescriptorType::eStorageBuffer)
					.setDescriptorCount(1)
					.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			}
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(bindings);
			result.layout = device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(1);
			result.pool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(result.pool);
			allocInfo.setSetLayouts(result.layout);
			result.set = device.allocateDescriptorSets(allocInfo).front();

			return result;
		}

		inline void DestroyParticleDescriptorSet(vk::Device device, ParticleDescriptorSet& s) {
			if (s.pool) {
				device.destroyDescriptorPool(s.pool);
				s.pool = nullptr;
			}
			if (s.layout) {
				device.destroyDescriptorSetLayout(s.layout);
				s.layout = nullptr;
			}
			s.set = nullptr;
		}

		struct ParticleDescriptorCache {
			vk::DescriptorSet         set{nullptr};
			std::array<vk::Buffer, 8> buffers{};
		};

		inline void RefreshParticleDescriptorSet(
			ParticleDescriptorCache&               cache,
			vk::Device                             device,
			vk::DescriptorSet                      particleSet,
			const graph::PhysicalResourceRegistry* registry
		) {
			if (!registry || !particleSet) {
				return;
			}
			auto pBuf = registry->GetBuffer<ParticleBuffer>();
			auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
			auto pAboveAliveBuf = registry->GetBuffer<AboveWaterParticleAliveBuffer>();
			auto pAboveIndirectBuf = registry->GetBuffer<AboveWaterParticleIndirectBuffer>();
			auto pUnderAliveBuf = registry->GetBuffer<UnderwaterParticleAliveBuffer>();
			auto pUnderIndirectBuf = registry->GetBuffer<UnderwaterParticleIndirectBuffer>();
			auto pGridHeadsBuf = registry->GetBuffer<ParticleGridHeadsBuffer>();
			auto pGridNextBuf = registry->GetBuffer<ParticleGridNextBuffer>();

			if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf ||
			    !pGridHeadsBuf || !pGridNextBuf) {
				return;
			}

			std::array<vk::Buffer, 8> buffers{
				pBuf->GetBuffer(),
				pTypeBuf->GetBuffer(),
				pAboveAliveBuf->GetBuffer(),
				pAboveIndirectBuf->GetBuffer(),
				pUnderAliveBuf->GetBuffer(),
				pUnderIndirectBuf->GetBuffer(),
				pGridHeadsBuf->GetBuffer(),
				pGridNextBuf->GetBuffer(),
			};
			if (cache.set == particleSet && cache.buffers == buffers) {
				return;
			}

			UpdateParticleDescriptorSet(device, particleSet, registry);
			cache.set = particleSet;
			cache.buffers = buffers;
		}

	} // namespace detail

} // namespace brassica
