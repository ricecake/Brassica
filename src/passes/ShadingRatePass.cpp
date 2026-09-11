#include "passes/ShadingRatePass.hpp"

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	ShadingRatePass::ShadingRatePass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	): ComputePass("ShadingRatePass", dev) {
		// Set 1 layout: binding 0 = Normal, binding 1 = Depth, binding 2 = Shading Rate storage image
		vk::DescriptorSetLayoutBinding bindings[3];

		bindings[0].setBinding(0);
		bindings[0].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		bindings[0].setDescriptorCount(1);
		bindings[0].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		bindings[1].setBinding(1);
		bindings[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		bindings[1].setDescriptorCount(1);
		bindings[1].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		bindings[2].setBinding(2);
		bindings[2].setDescriptorType(vk::DescriptorType::eStorageImage);
		bindings[2].setDescriptorCount(1);
		bindings[2].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		set1Layout = device.createDescriptorSetLayout(layoutInfo);

		vk::DescriptorPoolSize poolSizes[2];
		poolSizes[0].setType(vk::DescriptorType::eCombinedImageSampler);
		poolSizes[0].setDescriptorCount(2);
		poolSizes[1].setType(vk::DescriptorType::eStorageImage);
		poolSizes[1].setDescriptorCount(1);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(1);
		poolInfo.setPoolSizes(poolSizes);
		descriptorPool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(set1Layout);
		descriptorSet = device.allocateDescriptorSets(allocInfo).front();

		compShader.CompileComputeFromFile(device, "shaders/shading_rate.comp");
		SetShader(&compShader);

		vk::DescriptorSetLayout setLayouts[] = {globalSet0Layout, set1Layout};
		InitComputePipeline(setLayouts, {}, watcher, pCache);
	}

	ShadingRatePass::~ShadingRatePass() {
		DestroyPipeline();
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (set1Layout) {
			device.destroyDescriptorSetLayout(set1Layout);
			set1Layout = nullptr;
		}
		compShader.Destroy(device);
	}

	void ShadingRatePass::UpdateDescriptors(
		vk::ImageView gBufferNormalView,
		vk::Sampler   gBufferNormalSampler,
		vk::ImageView gBufferDepthView,
		vk::Sampler   gBufferDepthSampler,
		vk::ImageView shadingRateMapView
	) {
		vk::DescriptorImageInfo normalInfo{gBufferNormalSampler, gBufferNormalView, vk::ImageLayout::eShaderReadOnlyOptimal};
		vk::DescriptorImageInfo depthInfo{gBufferDepthSampler, gBufferDepthView, vk::ImageLayout::eShaderReadOnlyOptimal};
		vk::DescriptorImageInfo shadingRateInfo{nullptr, shadingRateMapView, vk::ImageLayout::eGeneral};

		vk::WriteDescriptorSet writes[3];

		writes[0].setDstSet(descriptorSet);
		writes[0].setDstBinding(0);
		writes[0].setDstArrayElement(0);
		writes[0].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		writes[0].setImageInfo(normalInfo);

		writes[1].setDstSet(descriptorSet);
		writes[1].setDstBinding(1);
		writes[1].setDstArrayElement(0);
		writes[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		writes[1].setImageInfo(depthInfo);

		writes[2].setDstSet(descriptorSet);
		writes[2].setDstBinding(2);
		writes[2].setDstArrayElement(0);
		writes[2].setDescriptorType(vk::DescriptorType::eStorageImage);
		writes[2].setImageInfo(shadingRateInfo);

		device.updateDescriptorSets(writes, nullptr);
	}

} // namespace brassica
