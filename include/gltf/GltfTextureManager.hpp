#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	struct LoadedTextureResource {
		vk::Image image{nullptr};
		vk::ImageView imageView{nullptr};
		VmaAllocation allocation{VK_NULL_HANDLE};
		int32_t globalIndex{-1};
	};

	class GltfTextureManager {
	public:
		static constexpr uint32_t MAX_TEXTURES = 1024;

		GltfTextureManager() = default;
		~GltfTextureManager();

		void Init(vk::Device device, VmaAllocator allocator, vk::Queue graphicsQueue, uint32_t queueFamilyIndex);
		void Cleanup();

		int32_t LoadGltfImage(
			const fastgltf::Asset& asset,
			const fastgltf::Image& image,
			const std::string& baseDir
		);

		int32_t CreateTextureFromPixels(
			const uint8_t* pixels,
			int width,
			int height,
			int channels,
			vk::Format format = vk::Format::eR8G8B8A8Unorm
		);

		vk::DescriptorSetLayout GetDescriptorSetLayout() const { return descriptorSetLayout; }
		vk::DescriptorSet GetDescriptorSet() const { return descriptorSet; }
		vk::Sampler GetSampler() const { return sampler; }

	private:
		vk::Device device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};
		vk::Queue graphicsQueue{nullptr};
		uint32_t queueFamilyIndex{0};

		vk::Sampler sampler{nullptr};
		vk::DescriptorSetLayout descriptorSetLayout{nullptr};
		vk::DescriptorPool descriptorPool{nullptr};
		vk::DescriptorSet descriptorSet{nullptr};

		std::vector<LoadedTextureResource> textures;

		void CreateDefaultTexture();
	};

} // namespace brassica
