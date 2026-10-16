#include "passes/DeferredPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	DeferredPass::DeferredPass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	):
		RenderPass("DeferredPass", dev, colorFmt) {
		CreateDescriptorResources(dev);
		InitPipeline(dev, globalSet0Layout, colorFmt, watcher, pCache);
	}

	DeferredPass::~DeferredPass() {
		CleanupDescriptorResources();
	}

	void DeferredPass::CreateDescriptorResources(vk::Device dev) {
		// Set 1 Layout:
		// Binding 0: Position sampler
		// Binding 1: Normal sampler
		// Binding 2: Albedo sampler
		// Binding 3: Background sampler
		// Binding 4: Terrain Clipmap Texture Array sampler
		// Binding 5: Acceleration Structure (TLAS)
		// Binding 6: Terrain AABB Storage Buffer
		std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};
		for (uint32_t i = 0; i < 5; ++i) {
			bindings[i].setBinding(i);
			bindings[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
			bindings[i].setDescriptorCount(1);
			bindings[i].setStageFlags(vk::ShaderStageFlagBits::eFragment);
		}
		bindings[5].setBinding(5);
		bindings[5].setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
		bindings[5].setDescriptorCount(1);
		bindings[5].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		bindings[6].setBinding(6);
		bindings[6].setDescriptorType(vk::DescriptorType::eStorageBuffer);
		bindings[6].setDescriptorCount(1);
		bindings[6].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		gbufferSetLayout = dev.createDescriptorSetLayout(layoutInfo);

		// Pool
		std::array<vk::DescriptorPoolSize, 3> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(5 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1 * FRAME_OVERLAP);
		poolSizes[2].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// Allocate Sets
		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, gbufferSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = dev.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			gbufferDescriptorSets[i] = allocatedSets[i];
		}

		// Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void DeferredPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (gbufferSetLayout) {
			device.destroyDescriptorSetLayout(gbufferSetLayout);
			gbufferSetLayout = nullptr;
		}
	}

	void DeferredPass::InitPipeline(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/deferred.vert")) {
			spdlog::error("Failed to compile deferred.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/deferred.frag")) {
			spdlog::error("Failed to compile deferred.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);

		std::array<vk::DescriptorSetLayout, 2> setLayouts = {globalSet0Layout, gbufferSetLayout};
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
			pCache
		);
	}

} // namespace brassica
