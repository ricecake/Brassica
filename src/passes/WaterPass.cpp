#include "passes/WaterPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	WaterPass::WaterPass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	):
		RenderPass("WaterPass", dev, colorFmt) {
		CreateDescriptorResources(dev);
		InitPipeline(dev, globalSet0Layout, colorFmt, watcher, pCache);
	}

	WaterPass::~WaterPass() {
		CleanupDescriptorResources();
	}

	void WaterPass::CreateDescriptorResources(vk::Device dev) {
		// Set 1 Layout:
		// Binding 0: Position sampler (GBufferPosition)
		// Binding 1: Depth sampler (GBufferDepth)
		// Binding 2: Albedo sampler (GBufferAlbedo)
		// Binding 3: Terrain Clipmap Texture Array sampler
		std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
		for (uint32_t i = 0; i < 4; ++i) {
			bindings[i].setBinding(i);
			bindings[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
			bindings[i].setDescriptorCount(1);
			bindings[i].setStageFlags(vk::ShaderStageFlagBits::eFragment);
		}

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		waterSetLayout = dev.createDescriptorSetLayout(layoutInfo);

		// Pool
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(4 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// Allocate Sets
		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, waterSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = dev.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			waterDescriptorSets[i] = allocatedSets[i];
		}

		// Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void WaterPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (waterSetLayout) {
			device.destroyDescriptorSetLayout(waterSetLayout);
			waterSetLayout = nullptr;
		}
	}

	void WaterPass::InitPipeline(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/water.vert")) {
			spdlog::error("Failed to compile water.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/water.frag")) {
			spdlog::error("Failed to compile water.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);

		std::array<vk::DescriptorSetLayout, 2> setLayouts = {globalSet0Layout, waterSetLayout};
		vk::PushConstantRange                  pushConstantRange{};
		pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eFragment);
		pushConstantRange.setOffset(0);
		pushConstantRange.setSize(sizeof(TerrainPushConstants));

		storedPushConstants.assign({pushConstantRange});

		InitRenderPipeline(
			colorFmt,
			vk::Format::eUndefined,
			setLayouts,
			storedPushConstants,
			watcher,
			false,
			false,
			vk::CompareOp::eLess,
			vk::CullModeFlagBits::eNone,
			pCache,
			true // enableBlend
		);
	}

} // namespace brassica
