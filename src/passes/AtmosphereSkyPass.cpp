#include "passes/AtmosphereSkyPass.hpp"

#include <array>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	AtmosphereSkyPass::AtmosphereSkyPass(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	): RenderPass("AtmosphereSkyPass", dev, vk::Format::eR16G16B16A16Sfloat) {
		if (instance && dev) {
			dls.init(static_cast<VkInstance>(instance), static_cast<VkDevice>(dev));
		}
		CreateDescriptorResources(dev);
		InitPipeline(instance, dev, globalSet0Layout, watcher, pCache);
	}

	AtmosphereSkyPass::~AtmosphereSkyPass() {
		DestroyPipeline();
		CleanupDescriptorResources();
	}

	void AtmosphereSkyPass::CreateDescriptorResources(vk::Device dev) {
		if (!dev)
			return;

		std::array<vk::DescriptorSetLayoutBinding, 2> skyBindings{};
		skyBindings[0].setBinding(0);
		skyBindings[0].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		skyBindings[0].setDescriptorCount(1);
		skyBindings[0].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		skyBindings[1].setBinding(1);
		skyBindings[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		skyBindings[1].setDescriptorCount(1);
		skyBindings[1].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo skyLayoutInfo{};
		skyLayoutInfo.setBindings(skyBindings);
		skySetLayout = dev.createDescriptorSetLayout(skyLayoutInfo);

		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eCombinedImageSampler);
		poolSize.setDescriptorCount(2 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, skySetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocated = dev.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			skySets[i] = allocated[i];
		}

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void AtmosphereSkyPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (skySetLayout) {
			device.destroyDescriptorSetLayout(skySetLayout);
			skySetLayout = nullptr;
		}
	}

	void AtmosphereSkyPass::InitPipeline(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	) {
		DestroyPipeline();

		if (!taskShader.CompileTaskFromFile(dev, "shaders/atmosphere/sky.task")) {
			spdlog::error("Failed to compile shaders/atmosphere/sky.task");
			return;
		}

		if (!meshShader.CompileMeshFromFile(dev, "shaders/atmosphere/sky.mesh")) {
			spdlog::error("Failed to compile shaders/atmosphere/sky.mesh");
			return;
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/atmosphere/sky.frag")) {
			spdlog::error("Failed to compile shaders/atmosphere/sky.frag");
			return;
		}

		SetShaders(&meshShader, &fragShader);

		vk::PushConstantRange pushRange{};
		pushRange.setStageFlags(
			vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment
		);
		pushRange.setOffset(0);
		pushRange.setSize(sizeof(AtmosphereSkyPushConstants));

		storedSetLayouts = {globalSet0Layout, skySetLayout};
		storedPushConstants = {pushRange};

		InitRenderPipeline(
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eUndefined,
			storedSetLayouts,
			storedPushConstants,
			watcher,
			false,
			false,
			vk::CompareOp::eAlways,
			vk::CullModeFlagBits::eNone,
			pCache
		);

		if (watcher) {
			watcher->RegisterShader(&taskShader, [this, instance, dev, globalSet0Layout, pCache]() {
				spdlog::info("Hot-reloading sky.task...");
				InitPipeline(instance, dev, globalSet0Layout, nullptr, pCache);
			});
			watcher->RegisterShader(&meshShader, [this, instance, dev, globalSet0Layout, pCache]() {
				spdlog::info("Hot-reloading sky.mesh...");
				InitPipeline(instance, dev, globalSet0Layout, nullptr, pCache);
			});
			watcher->RegisterShader(&fragShader, [this, instance, dev, globalSet0Layout, pCache]() {
				spdlog::info("Hot-reloading sky.frag...");
				InitPipeline(instance, dev, globalSet0Layout, nullptr, pCache);
			});
		}
	}

	void AtmosphereSkyPass::DestroyPipeline() {
		taskShader.Destroy(device);
		meshShader.Destroy(device);
		fragShader.Destroy(device);
	}

	AtmosphereSkyPassData AtmosphereSkyPass::RegisterPass(
		FrameGraph&                       fg,
		FrameGraphBlackboard&             blackboard,
		vk::Extent2D                      extent,
		vk::DescriptorSet                 globalDescriptorSet,
		uint32_t                          activeFrame,
		const AtmosphereSkyPushConstants& push
	) {
		AtmosphereSkyPassData data{};

		auto* lutData = blackboard.try_get<AtmosphereLUTData>();

		data.background = fg.addCallbackPass<FrameGraphResource>(
			"AtmosphereSkyPass",
			[&](FrameGraph::Builder& builder, FrameGraphResource& resource) {
				if (lutData) {
					builder.read(lutData->skyViewLUT, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
					builder.read(lutData->transmittanceLUT, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
				}

				resource = builder.create<FrameGraphTexture2D>(
					"Background",
					FrameGraphTexture2D::Desc{
						.extent = {extent.width, extent.height},
						.format = vk::Format::eR16G16B16A16Sfloat,
						.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
					}
				);
				resource = builder.write(resource, static_cast<uint32_t>(TextureUsage::ColorAttachment));
			},
			[this,
			 extent,
			 globalDescriptorSet,
			 activeFrame,
			 lutData,
			 push](const FrameGraphResource& resource, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             bgTex = resources.get<FrameGraphTexture2D>(resource);

				if (lutData) {
					auto& skyViewTex = resources.get<FrameGraphTexture2D>(lutData->skyViewLUT);
					auto& transTex = resources.get<FrameGraphTexture2D>(lutData->transmittanceLUT);

					vk::DescriptorSet currentSet = skySets[activeFrame % FRAME_OVERLAP];

					std::array<vk::DescriptorImageInfo, 2> imageInfos{};
					imageInfos[0].setSampler(sampler);
					imageInfos[0].setImageView(skyViewTex.imageView);
					imageInfos[0].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

					imageInfos[1].setSampler(sampler);
					imageInfos[1].setImageView(transTex.imageView);
					imageInfos[1].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

					std::array<vk::WriteDescriptorSet, 2> writes{};
					writes[0].setDstSet(currentSet);
					writes[0].setDstBinding(0);
					writes[0].setDstArrayElement(0);
					writes[0].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
					writes[0].setImageInfo(imageInfos[0]);

					writes[1].setDstSet(currentSet);
					writes[1].setDstBinding(1);
					writes[1].setDstArrayElement(0);
					writes[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
					writes[1].setImageInfo(imageInfos[1]);

					device.updateDescriptorSets(writes, nullptr);
				}

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(bgTex.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(vk::ClearValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));

				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

				std::array<vk::DescriptorSet, 2> sets = {globalDescriptorSet, skySets[activeFrame % FRAME_OVERLAP]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, sets, nullptr);

				cmd.pushConstants(
					pipelineLayout,
					vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
						vk::ShaderStageFlagBits::eFragment,
					0,
					sizeof(AtmosphereSkyPushConstants),
					&push
				);

				cmd.drawMeshTasksEXT(1, 1, 1, dls);

				EndRendering(cmd);
			}
		);

		blackboard.add<AtmosphereSkyPassData>() = data;
		return data;
	}

} // namespace brassica
