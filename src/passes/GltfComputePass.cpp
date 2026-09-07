#include "passes/GltfComputePass.hpp"
#include "spdlog/spdlog.h"
#include <array>
#include <string>

namespace brassica {

	GltfComputePass::GltfComputePass(vk::Device dev, ShaderWatcher* watcher)
		: ComputePass("GltfComputePass", dev) {
		InitPipeline(watcher);
	}

	void GltfComputePass::InitPipeline(ShaderWatcher* watcher) {
		if (!computeShader.CompileComputeFromFile(device, "shaders/gltf_instance.comp")) {
			spdlog::error("Failed to compile shaders/gltf_instance.comp");
		}

		SetShader(&computeShader);

		// Layout Set 0: 5 SSBO bindings
		std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
		// 0: InstanceBuffer
		bindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		// 1: PrimitiveBuffer
		bindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		// 2: IndirectDrawBuffer
		bindings[2].setBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		// 3: VisibleInstanceBuffer
		bindings[3].setBinding(3).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		// 4: DrawCountBuffer
		bindings[4].setBinding(4).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		computeSetLayout = device.createDescriptorSetLayout(layoutInfo);

		vk::PushConstantRange pcRange{};
		pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
		pcRange.setOffset(0);
		pcRange.setSize(sizeof(GltfComputePushConstants));

		InitComputePipeline(std::span(&computeSetLayout, 1), std::span(&pcRange, 1), watcher);
	}

	void GltfComputePass::RegisterPass(
		FrameGraph& fg,
		FrameGraphBlackboard& blackboard,
		vk::DescriptorSet computeSet,
		const GltfComputePushConstants& pushConstants,
		vk::Buffer indirectDrawBuf,
		vk::Buffer visibleInstanceBuf,
		vk::Buffer drawCountBuf,
		size_t indirectDrawSize,
		size_t visibleInstanceSize
	) {
		std::string indirectName = "GLTF_IndirectDraws_" + std::to_string(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(indirectDrawBuf)));
		std::string visibleName = "GLTF_VisibleInstances_" + std::to_string(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(visibleInstanceBuf)));
		std::string countName = "GLTF_DrawCount_" + std::to_string(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(drawCountBuf)));

		FrameGraphResource importedIndirect = fg.import(indirectName, {indirectDrawSize}, FrameGraphSSBO{indirectDrawBuf});
		FrameGraphResource importedVisible = fg.import(visibleName, {visibleInstanceSize}, FrameGraphSSBO{visibleInstanceBuf});
		FrameGraphResource importedCount = fg.import(countName, {sizeof(uint32_t)}, FrameGraphSSBO{drawCountBuf});

		const auto& passData = fg.addCallbackPass<GltfComputePassData>(
			"GltfComputePass",
			[&](FrameGraph::Builder& builder, GltfComputePassData& data) {
				data.indirectDrawTarget = builder.write(importedIndirect, static_cast<uint32_t>(BufferUsage::Indirect));
				data.visibleInstancesTarget = builder.write(importedVisible, static_cast<uint32_t>(BufferUsage::StorageWrite));
				data.drawCountTarget = builder.write(importedCount, static_cast<uint32_t>(BufferUsage::StorageWrite));
				builder.setSideEffect();
			},
			[this, computeSet, pushConstants, drawCountBuf](const GltfComputePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				if (pushConstants.totalInstances == 0) return;

				// Reset atomic draw count buffer to 0 before compute execution
				cmd.fillBuffer(drawCountBuf, 0, sizeof(uint32_t), 0);

				vk::BufferMemoryBarrier barrier{};
				barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
				barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
				barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
				barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
				barrier.setBuffer(drawCountBuf);
				barrier.setOffset(0);
				barrier.setSize(sizeof(uint32_t));

				cmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eTransfer,
					vk::PipelineStageFlagBits::eComputeShader,
					{}, nullptr, barrier, nullptr
				);

				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
				if (computeSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, computeSet, nullptr);
				}

				cmd.pushConstants<GltfComputePushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

				uint32_t groupCount = (pushConstants.totalInstances + 63) / 64;
				Dispatch(cmd, groupCount, 1, 1);
			}
		);

		blackboard.add<GltfComputePassData>() = passData;
	}

} // namespace brassica
