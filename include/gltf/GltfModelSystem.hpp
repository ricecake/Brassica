#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "entt/entt.hpp"
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

#include "gltf/GltfModel.hpp"
#include "gltf/GltfAnimation.hpp"
#include "gltf/GltfTextureManager.hpp"
#include "passes/GltfComputePass.hpp"
#include "passes/GltfRenderPass.hpp"
#include "types/CameraData.hpp"
#include "ecs/Components.hpp"

namespace brassica {

	struct InstanceGPUData {
		glm::mat4 worldMatrix{1.0f};
		uint32_t modelIndex{0};
		uint32_t primitiveIndex{0};
		uint32_t jointMatrixOffset{0};
		uint32_t jointCount{0};
	};

	struct VisibleInstanceGPUData {
		glm::mat4 worldMatrix{1.0f};
		uint32_t primitiveIndex{0};
		uint32_t materialIndex{0};
		uint32_t jointMatrixOffset{0};
		uint32_t jointCount{0};
	};

	struct PerModelFrameResources {
		vk::Buffer instanceBuffer{nullptr};
		VmaAllocation instanceAllocation{VK_NULL_HANDLE};
		size_t instanceBufferSize{0};

		vk::Buffer jointMatrixBuffer{nullptr};
		VmaAllocation jointMatrixAllocation{VK_NULL_HANDLE};
		size_t jointMatrixBufferSize{0};

		vk::Buffer indirectDrawBuffer{nullptr};
		VmaAllocation indirectDrawAllocation{VK_NULL_HANDLE};
		size_t indirectDrawBufferSize{0};

		vk::Buffer visibleInstanceBuffer{nullptr};
		VmaAllocation visibleInstanceAllocation{VK_NULL_HANDLE};
		size_t visibleInstanceBufferSize{0};

		vk::Buffer drawCountBuffer{nullptr};
		VmaAllocation drawCountAllocation{VK_NULL_HANDLE};

		vk::DescriptorSet computeDescriptorSet{nullptr};
		vk::DescriptorSet modelDescriptorSet{nullptr};
	};

	class GltfModelSystem {
	public:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		GltfModelSystem() = default;
		~GltfModelSystem();

		void Init(
			vk::Instance instance,
			vk::Device device,
			VmaAllocator allocator,
			vk::DescriptorSetLayout globalSet0Layout,
			GltfTextureManager* textureManager,
			ShaderWatcher* watcher = nullptr
		);

		void Cleanup();

		std::shared_ptr<GltfModel> LoadModel(const std::string& filepath);

		void Update(entt::registry& registry, const CameraData& camera, float deltaTime);

		void Render(
			entt::registry& registry,
			FrameGraph& fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D extent,
			vk::DescriptorSet globalDescriptorSet,
			const CameraData& camera,
			uint32_t activeFrame
		);

		GltfTextureManager* GetTextureManager() const { return textureManager; }

	private:
		vk::Device device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};
		GltfTextureManager* textureManager{nullptr};

		std::unique_ptr<GltfComputePass> computePass;
		std::unique_ptr<GltfRenderPass> renderPass;

		vk::DescriptorPool descriptorPool{nullptr};

		std::unordered_map<std::string, std::shared_ptr<GltfModel>> modelCache;

		// Frame resources mapped by model pointer and activeFrame index
		std::unordered_map<GltfModel*, std::array<PerModelFrameResources, FRAME_OVERLAP>> modelResources;

		void EnsureBufferCapacity(
			PerModelFrameResources& res,
			size_t instanceCount,
			size_t jointCount
		);

		void CreateDescriptorPool();
	};

} // namespace brassica
