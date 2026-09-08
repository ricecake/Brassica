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

		vk::DescriptorSetLayoutBinding transBinding{};
		transBinding.setBinding(0);
		transBinding.setDescriptorType(vk::DescriptorType::eStorageImage);
		transBinding.setDescriptorCount(1);
		transBinding.setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo transLayoutInfo{};
		transLayoutInfo.setBindings(transBinding);
		transmittanceSetLayout = dev.createDescriptorSetLayout(transLayoutInfo);

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

		std::array<vk::DescriptorSetLayoutBinding, 3> skyBindings{};
		skyBindings[0].setBinding(0);
		skyBindings[0].setDescriptorType(vk::DescriptorType::eStorageImage);
		skyBindings[0].setDescriptorCount(1);
		skyBindings[0].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		skyBindings[1].setBinding(1);
		skyBindings[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		skyBindings[1].setDescriptorCount(1);
		skyBindings[1].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		skyBindings[2].setBinding(2);
		skyBindings[2].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		skyBindings[2].setDescriptorCount(1);
		skyBindings[2].setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo skyLayoutInfo{};
		skyLayoutInfo.setBindings(skyBindings);
		skyViewSetLayout = dev.createDescriptorSetLayout(skyLayoutInfo);

		std::array<vk::DescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eStorageImage).setDescriptorCount(3 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(3 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(3 * FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

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

		std::vector<vk::DescriptorSetLayout> skyLayouts(FRAME_OVERLAP, skyViewSetLayout);
		vk::DescriptorSetAllocateInfo        skyAllocInfo{};
		skyAllocInfo.setDescriptorPool(descriptorPool);
		skyAllocInfo.setSetLayouts(skyLayouts);

		auto allocatedSkySets = dev.allocateDescriptorSets(skyAllocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			skyViewSets[i] = allocatedSkySets[i];
		}

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
		if (skyViewSetLayout) {
			device.destroyDescriptorSetLayout(skyViewSetLayout);
			skyViewSetLayout = nullptr;
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

		if (!skyViewShader.CompileComputeFromFile(dev, "shaders/atmosphere/sky_view_lut.comp")) {
			spdlog::error("Failed to compile shaders/atmosphere/sky_view_lut.comp");
			return;
		}

		if (!dev)
			return;

		vk::PushConstantRange transPushRange{};
		transPushRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
		transPushRange.setOffset(0);
		transPushRange.setSize(sizeof(AtmospherePushConstants));

		vk::PipelineLayoutCreateInfo transPipelineLayoutInfo{};
		transPipelineLayoutInfo.setSetLayouts(transmittanceSetLayout);
		transPipelineLayoutInfo.setPushConstantRanges(transPushRange);
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

		vk::PipelineLayoutCreateInfo multiPipelineLayoutInfo{};
		multiPipelineLayoutInfo.setSetLayouts(multiScatteringSetLayout);
		multiPipelineLayoutInfo.setPushConstantRanges(transPushRange);
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

		vk::PushConstantRange skyPushRange{};
		skyPushRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
		skyPushRange.setOffset(0);
		skyPushRange.setSize(sizeof(SkyViewPushConstants));

		vk::PipelineLayoutCreateInfo skyPipelineLayoutInfo{};
		skyPipelineLayoutInfo.setSetLayouts(skyViewSetLayout);
		skyPipelineLayoutInfo.setPushConstantRanges(skyPushRange);
		skyViewPipelineLayout = dev.createPipelineLayout(skyPipelineLayoutInfo);

		vk::ComputePipelineCreateInfo skyPipelineInfo{};
		skyPipelineInfo.setStage(skyViewShader.GetStageCreateInfo());
		skyPipelineInfo.setLayout(skyViewPipelineLayout);

		auto skyResult = dev.createComputePipeline(nullptr, skyPipelineInfo);
		if (skyResult.result == vk::Result::eSuccess) {
			skyViewPipeline = skyResult.value;
		} else {
			spdlog::error("Failed to create skyview compute pipeline");
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
			watcher->RegisterShader(&skyViewShader, [this, dev]() {
				spdlog::info("Hot-reloading sky_view_lut.comp...");
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
		if (skyViewPipeline) {
			device.destroyPipeline(skyViewPipeline);
			skyViewPipeline = nullptr;
		}
		if (transmittancePipelineLayout) {
			device.destroyPipelineLayout(transmittancePipelineLayout);
			transmittancePipelineLayout = nullptr;
		}
		if (multiScatteringPipelineLayout) {
			device.destroyPipelineLayout(multiScatteringPipelineLayout);
			multiScatteringPipelineLayout = nullptr;
		}
		if (skyViewPipelineLayout) {
			device.destroyPipelineLayout(skyViewPipelineLayout);
			skyViewPipelineLayout = nullptr;
		}
		transmittanceShader.Destroy(device);
		multiScatteringShader.Destroy(device);
		skyViewShader.Destroy(device);
	}

	AtmosphereLUTData AtmosphereLUTPass::RegisterPass(
		FrameGraph&                    fg,
		FrameGraphBlackboard&          blackboard,
		uint32_t                       activeFrame,
		const AtmospherePushConstants& push,
		const SkyViewPushConstants&    skyPush
	) {
		AtmosphereLUTData data{};

		data.transmittanceLUT = fg.addCallbackPass<FrameGraphResource>(
			"AtmosphereTransmittancePass",
			[&](FrameGraph::Builder& builder, FrameGraphResource& resource) {
				resource = builder.create<FrameGraphTexture2D>(
					"TransmittanceLUT",
					FrameGraphTexture2D::Desc{
						.extent = {256, 64},
						.format = vk::Format::eR32G32B32A32Sfloat,
						.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
					}
				);
				resource = builder.write(resource, static_cast<uint32_t>(TextureUsage::StorageWrite));
			},
			[this,
			 activeFrame,
			 push](const FrameGraphResource& resource, FrameGraphPassResources& resources, void* ctx) {
				if (!transmittancePipeline)
					return;

				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             transTex = resources.get<FrameGraphTexture2D>(resource);

				vk::DescriptorSet currentSet = transmittanceSets[activeFrame % FRAME_OVERLAP];

				vk::DescriptorImageInfo imageInfo{};
				imageInfo.setImageView(transTex.imageView);
				imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

				vk::WriteDescriptorSet descriptorWrite{};
				descriptorWrite.setDstSet(currentSet);
				descriptorWrite.setDstBinding(0);
				descriptorWrite.setDstArrayElement(0);
				descriptorWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
				descriptorWrite.setImageInfo(imageInfo);

				device.updateDescriptorSets(descriptorWrite, nullptr);

				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, transmittancePipeline);
				cmd.pushConstants(
					transmittancePipelineLayout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(AtmospherePushConstants),
					&push
				);
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute,
					transmittancePipelineLayout,
					0,
					currentSet,
					nullptr
				);

				cmd.dispatch((256 + 7) / 8, (64 + 7) / 8, 1);
			}
		);

		data.multiScatteringLUT = fg.addCallbackPass<FrameGraphResource>(
			"AtmosphereMultiScatteringPass",
			[&](FrameGraph::Builder& builder, FrameGraphResource& resource) {
				builder.read(data.transmittanceLUT, static_cast<uint32_t>(TextureUsage::SampledShaderRead));

				resource = builder.create<FrameGraphTexture2D>(
					"MultiScatteringLUT",
					FrameGraphTexture2D::Desc{
						.extent = {32, 32},
						.format = vk::Format::eR32G32B32A32Sfloat,
						.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
					}
				);
				resource = builder.write(resource, static_cast<uint32_t>(TextureUsage::StorageWrite));
			},
			[this,
			 activeFrame,
			 data,
			 push](const FrameGraphResource& resource, FrameGraphPassResources& resources, void* ctx) {
				if (!multiScatteringPipeline)
					return;

				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             transTex = resources.get<FrameGraphTexture2D>(data.transmittanceLUT);
				auto&             multiTex = resources.get<FrameGraphTexture2D>(resource);

				vk::DescriptorSet currentSet = multiScatteringSets[activeFrame % FRAME_OVERLAP];

				std::array<vk::DescriptorImageInfo, 2> imageInfos{};
				imageInfos[0].setImageView(multiTex.imageView);
				imageInfos[0].setImageLayout(vk::ImageLayout::eGeneral);

				imageInfos[1].setSampler(sampler);
				imageInfos[1].setImageView(transTex.imageView);
				imageInfos[1].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
				descriptorWrites[0].setDstSet(currentSet);
				descriptorWrites[0].setDstBinding(0);
				descriptorWrites[0].setDstArrayElement(0);
				descriptorWrites[0].setDescriptorType(vk::DescriptorType::eStorageImage);
				descriptorWrites[0].setImageInfo(imageInfos[0]);

				descriptorWrites[1].setDstSet(currentSet);
				descriptorWrites[1].setDstBinding(1);
				descriptorWrites[1].setDstArrayElement(0);
				descriptorWrites[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[1].setImageInfo(imageInfos[1]);

				device.updateDescriptorSets(descriptorWrites, nullptr);

				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, multiScatteringPipeline);
				cmd.pushConstants(
					multiScatteringPipelineLayout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(AtmospherePushConstants),
					&push
				);
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute,
					multiScatteringPipelineLayout,
					0,
					currentSet,
					nullptr
				);

				cmd.dispatch(1, 1, 1);
			}
		);

		data.skyViewLUT = fg.addCallbackPass<FrameGraphResource>(
			"AtmosphereSkyViewPass",
			[&](FrameGraph::Builder& builder, FrameGraphResource& resource) {
				builder.read(data.transmittanceLUT, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
				builder.read(data.multiScatteringLUT, static_cast<uint32_t>(TextureUsage::SampledShaderRead));

				resource = builder.create<FrameGraphTexture2D>(
					"SkyViewLUT",
					FrameGraphTexture2D::Desc{
						.extent = {192, 108},
						.format = vk::Format::eR32G32B32A32Sfloat,
						.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
					}
				);
				resource = builder.write(resource, static_cast<uint32_t>(TextureUsage::StorageWrite));
			},
			[this,
			 activeFrame,
			 data,
			 skyPush](const FrameGraphResource& resource, FrameGraphPassResources& resources, void* ctx) {
				if (!skyViewPipeline)
					return;

				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             transTex = resources.get<FrameGraphTexture2D>(data.transmittanceLUT);
				auto&             multiTex = resources.get<FrameGraphTexture2D>(data.multiScatteringLUT);
				auto&             skyTex = resources.get<FrameGraphTexture2D>(resource);

				vk::DescriptorSet currentSet = skyViewSets[activeFrame % FRAME_OVERLAP];

				std::array<vk::DescriptorImageInfo, 3> imageInfos{};
				imageInfos[0].setImageView(skyTex.imageView);
				imageInfos[0].setImageLayout(vk::ImageLayout::eGeneral);

				imageInfos[1].setSampler(sampler);
				imageInfos[1].setImageView(transTex.imageView);
				imageInfos[1].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				imageInfos[2].setSampler(sampler);
				imageInfos[2].setImageView(multiTex.imageView);
				imageInfos[2].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};
				descriptorWrites[0].setDstSet(currentSet);
				descriptorWrites[0].setDstBinding(0);
				descriptorWrites[0].setDstArrayElement(0);
				descriptorWrites[0].setDescriptorType(vk::DescriptorType::eStorageImage);
				descriptorWrites[0].setImageInfo(imageInfos[0]);

				descriptorWrites[1].setDstSet(currentSet);
				descriptorWrites[1].setDstBinding(1);
				descriptorWrites[1].setDstArrayElement(0);
				descriptorWrites[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[1].setImageInfo(imageInfos[1]);

				descriptorWrites[2].setDstSet(currentSet);
				descriptorWrites[2].setDstBinding(2);
				descriptorWrites[2].setDstArrayElement(0);
				descriptorWrites[2].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[2].setImageInfo(imageInfos[2]);

				device.updateDescriptorSets(descriptorWrites, nullptr);

				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, skyViewPipeline);
				cmd.pushConstants(
					skyViewPipelineLayout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(SkyViewPushConstants),
					&skyPush
				);
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute,
					skyViewPipelineLayout,
					0,
					currentSet,
					nullptr
				);

				cmd.dispatch((192 + 7) / 8, (108 + 7) / 8, 1);
			}
		);

		blackboard.add<AtmosphereLUTData>() = data;
		return data;
	}

} // namespace brassica
