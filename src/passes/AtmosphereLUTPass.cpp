#include "passes/AtmosphereLUTPass.hpp"

#include <array>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	AtmosphereLUTPass::AtmosphereLUTPass(vk::Device dev, ShaderWatcher* watcher): Pass("AtmosphereLUTPass", dev) {
		CreateDescriptorResources(dev);
		InitPipeline(dev, watcher);
	}

	AtmosphereLUTPass::~AtmosphereLUTPass() {
		DestroyPipeline();
		CleanupDescriptorResources();
	}

	void AtmosphereLUTPass::CreateDescriptorResources(vk::Device dev) {
		if (!dev)
			return;

		// 1. Transmittance Set Layout: Binding 0 -> Storage Image
		vk::DescriptorSetLayoutBinding transBinding{};
		transBinding.setBinding(0);
		transBinding.setDescriptorType(vk::DescriptorType::eStorageImage);
		transBinding.setDescriptorCount(1);
		transBinding.setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo transLayoutInfo{};
		transLayoutInfo.setBindings(transBinding);
		transmittanceSetLayout = dev.createDescriptorSetLayout(transLayoutInfo);

		// 2. MultiScattering Set Layout: Binding 0 -> Storage Image, Binding 1 -> Combined Image Sampler
		std::array<vk::DescriptorSetLayoutBinding, 2> multiBindings{};
		multiBindings[0].setBinding(0);
		multiBindings[0].setDescriptorType(vk::DescriptorType::eStorageImage);
		multiBindings[0].setDescriptorCount(1);
		multiBindings[0].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		multiBindings[1].setBinding(1);
		multiBindings[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		multiBindings[1].setDescriptorCount(1);
		multiBindings[1].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo multiLayoutInfo{};
		multiLayoutInfo.setBindings(multiBindings);
		multiScatteringSetLayout = dev.createDescriptorSetLayout(multiLayoutInfo);

		// 3. Pool
		std::array<vk::DescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eStorageImage).setDescriptorCount(2 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(1 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(2 * FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// 4. Allocate Sets
		std::vector<vk::DescriptorSetLayout> transLayouts(FRAME_OVERLAP, transmittanceSetLayout);
		vk::DescriptorSetAllocateInfo        transAllocInfo{};
		transAllocInfo.setDescriptorPool(descriptorPool);
		transAllocInfo.setSetLayouts(transLayouts);

		auto allocatedTransSets = dev.allocateDescriptorSets(transAllocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			transmittanceSets[i] = allocatedTransSets[i];
		}

		std::vector<vk::DescriptorSetLayout> multiLayouts(FRAME_OVERLAP, multiScatteringSetLayout);
		vk::DescriptorSetAllocateInfo        multiAllocInfo{};
		multiAllocInfo.setDescriptorPool(descriptorPool);
		multiAllocInfo.setSetLayouts(multiLayouts);

		auto allocatedMultiSets = dev.allocateDescriptorSets(multiAllocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			multiScatteringSets[i] = allocatedMultiSets[i];
		}

		// 5. Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void AtmosphereLUTPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (transmittanceSetLayout) {
			device.destroyDescriptorSetLayout(transmittanceSetLayout);
			transmittanceSetLayout = nullptr;
		}
		if (multiScatteringSetLayout) {
			device.destroyDescriptorSetLayout(multiScatteringSetLayout);
			multiScatteringSetLayout = nullptr;
		}
	}

	void AtmosphereLUTPass::InitPipeline(vk::Device dev, ShaderWatcher* watcher) {
		DestroyPipeline();

		if (!transmittanceShader.CompileComputeFromFile(dev, "shaders/atmosphere/transmittance_lut.comp")) {
			spdlog::error("Failed to compile shaders/atmosphere/transmittance_lut.comp");
			return;
		}

		if (!multiScatteringShader.CompileComputeFromFile(dev, "shaders/atmosphere/multiscattering_lut.comp")) {
			spdlog::error("Failed to compile shaders/atmosphere/multiscattering_lut.comp");
			return;
		}

		if (!dev)
			return;

		vk::PushConstantRange pushRange{};
		pushRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
		pushRange.setOffset(0);
		pushRange.setSize(sizeof(AtmospherePushConstants));

		// Transmittance Pipeline Layout & Pipeline
		vk::PipelineLayoutCreateInfo transPipelineLayoutInfo{};
		transPipelineLayoutInfo.setSetLayouts(transmittanceSetLayout);
		transPipelineLayoutInfo.setPushConstantRanges(pushRange);
		transmittancePipelineLayout = dev.createPipelineLayout(transPipelineLayoutInfo);

		vk::ComputePipelineCreateInfo transPipelineInfo{};
		transPipelineInfo.setStage(transmittanceShader.GetStageCreateInfo());
		transPipelineInfo.setLayout(transmittancePipelineLayout);

		auto transResult = dev.createComputePipeline(nullptr, transPipelineInfo);
		if (transResult.result == vk::Result::eSuccess) {
			transmittancePipeline = transResult.value;
		} else {
			spdlog::error("Failed to create transmittance compute pipeline");
		}

		// MultiScattering Pipeline Layout & Pipeline
		vk::PipelineLayoutCreateInfo multiPipelineLayoutInfo{};
		multiPipelineLayoutInfo.setSetLayouts(multiScatteringSetLayout);
		multiPipelineLayoutInfo.setPushConstantRanges(pushRange);
		multiScatteringPipelineLayout = dev.createPipelineLayout(multiPipelineLayoutInfo);

		vk::ComputePipelineCreateInfo multiPipelineInfo{};
		multiPipelineInfo.setStage(multiScatteringShader.GetStageCreateInfo());
		multiPipelineInfo.setLayout(multiScatteringPipelineLayout);

		auto multiResult = dev.createComputePipeline(nullptr, multiPipelineInfo);
		if (multiResult.result == vk::Result::eSuccess) {
			multiScatteringPipeline = multiResult.value;
		} else {
			spdlog::error("Failed to create multiscattering compute pipeline");
		}

		if (watcher) {
			watcher->RegisterShader(&transmittanceShader, [this, dev]() {
				spdlog::info("Hot-reloading transmittance_lut.comp...");
				InitPipeline(dev, nullptr);
			});
			watcher->RegisterShader(&multiScatteringShader, [this, dev]() {
				spdlog::info("Hot-reloading multiscattering_lut.comp...");
				InitPipeline(dev, nullptr);
			});
		}
	}

	void AtmosphereLUTPass::DestroyPipeline() {
		if (transmittancePipeline) {
			device.destroyPipeline(transmittancePipeline);
			transmittancePipeline = nullptr;
		}
		if (multiScatteringPipeline) {
			device.destroyPipeline(multiScatteringPipeline);
			multiScatteringPipeline = nullptr;
		}
		if (transmittancePipelineLayout) {
			device.destroyPipelineLayout(transmittancePipelineLayout);
			transmittancePipelineLayout = nullptr;
		}
		if (multiScatteringPipelineLayout) {
			device.destroyPipelineLayout(multiScatteringPipelineLayout);
			multiScatteringPipelineLayout = nullptr;
		}
		transmittanceShader.Destroy(device);
		multiScatteringShader.Destroy(device);
	}

} // namespace brassica
