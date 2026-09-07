#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	class GltfTextureManager;

	struct GltfVertex {
		glm::vec3 position{0.0f};
		float _pad0{0.0f};
		glm::vec3 normal{0.0f, 1.0f, 0.0f};
		float _pad1{0.0f};
		glm::vec4 tangent{0.0f};
		glm::vec2 texcoord{0.0f};
		glm::uvec4 joints{0};
		glm::vec4 weights{0.0f};
	};

	struct GltfMeshlet {
		uint32_t vertexOffset{0};
		uint32_t triangleOffset{0};
		uint32_t vertexCount{0};
		uint32_t triangleCount{0};
		glm::vec3 center{0.0f};
		float radius{0.0f};
	};

	struct GltfMaterialGPU {
		glm::vec4 baseColorFactor{1.0f};
		float metallicFactor{1.0f};
		float roughnessFactor{1.0f};
		int32_t baseColorTextureIndex{-1};
		int32_t normalTextureIndex{-1};
		int32_t metallicRoughnessTextureIndex{-1};
		int32_t _pad0{0};
		int32_t _pad1{0};
		int32_t _pad2{0};
	};

	struct GltfPrimitiveGPU {
		uint32_t firstMeshlet{0};
		uint32_t meshletCount{0};
		uint32_t materialIndex{0};
		uint32_t firstVertex{0};
		uint32_t vertexCount{0};
		uint32_t _pad0{0};
		uint32_t _pad1{0};
		uint32_t _pad2{0};
	};

	struct GltfNode {
		std::string name;
		int32_t parent{-1};
		std::vector<int32_t> children;
		glm::vec3 translation{0.0f};
		glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
		glm::vec3 scale{1.0f};
		glm::mat4 localMatrix{1.0f};
		glm::mat4 globalMatrix{1.0f};
		int32_t meshIndex{-1};
		int32_t skinIndex{-1};
	};

	struct GltfSkin {
		std::string name;
		int32_t skeletonRoot{-1};
		std::vector<int32_t> joints;
		std::vector<glm::mat4> inverseBindMatrices;
	};

	enum class GltfTargetPath { Translation, Rotation, Scale };
	enum class GltfInterpolationType { Linear, Step, Cubicspline };

	struct GltfAnimationChannel {
		int32_t targetNode{-1};
		GltfTargetPath path{GltfTargetPath::Translation};
		std::vector<float> samplersInput;
		std::vector<glm::vec4> samplersOutput;
		GltfInterpolationType interpolation{GltfInterpolationType::Linear};
	};

	struct GltfAnimation {
		std::string name;
		float duration{0.0f};
		std::vector<GltfAnimationChannel> channels;
	};

	class GltfModel {
	public:
		GltfModel() = default;
		~GltfModel();

		GltfModel(const GltfModel&) = delete;
		GltfModel& operator=(const GltfModel&) = delete;

		GltfModel(GltfModel&& other) noexcept;
		GltfModel& operator=(GltfModel&& other) noexcept;

		bool LoadFromFile(
			vk::Device device,
			VmaAllocator allocator,
			GltfTextureManager* textureManager,
			const std::string& filepath
		);

		void Destroy();

		// Accessors
		const std::vector<GltfNode>& GetNodes() const { return nodes; }
		std::vector<GltfNode>& GetNodes() { return nodes; }
		const std::vector<GltfSkin>& GetSkins() const { return skins; }
		const std::vector<GltfAnimation>& GetAnimations() const { return animations; }
		const std::vector<GltfPrimitiveGPU>& GetPrimitives() const { return primitives; }

		uint32_t GetTotalMeshlets() const { return static_cast<uint32_t>(meshlets.size()); }
		uint32_t GetTotalVertices() const { return static_cast<uint32_t>(vertices.size()); }

		vk::Buffer GetVertexBuffer() const { return vertexBuffer; }
		vk::Buffer GetMeshletBuffer() const { return meshletBuffer; }
		vk::Buffer GetMeshletVerticesBuffer() const { return meshletVerticesBuffer; }
		vk::Buffer GetMeshletTrianglesBuffer() const { return meshletTrianglesBuffer; }
		vk::Buffer GetPrimitiveBuffer() const { return primitiveBuffer; }
		vk::Buffer GetMaterialBuffer() const { return materialBuffer; }

	private:
		vk::Device device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};

		std::vector<GltfVertex> vertices;
		std::vector<GltfMeshlet> meshlets;
		std::vector<uint32_t> meshletVertices;
		std::vector<uint8_t> meshletTriangles;
		std::vector<GltfPrimitiveGPU> primitives;
		std::vector<GltfMaterialGPU> materials;

		std::vector<GltfNode> nodes;
		std::vector<GltfSkin> skins;
		std::vector<GltfAnimation> animations;

		vk::Buffer vertexBuffer{nullptr};
		VmaAllocation vertexAllocation{VK_NULL_HANDLE};

		vk::Buffer meshletBuffer{nullptr};
		VmaAllocation meshletAllocation{VK_NULL_HANDLE};

		vk::Buffer meshletVerticesBuffer{nullptr};
		VmaAllocation meshletVerticesAllocation{VK_NULL_HANDLE};

		vk::Buffer meshletTrianglesBuffer{nullptr};
		VmaAllocation meshletTrianglesAllocation{VK_NULL_HANDLE};

		vk::Buffer primitiveBuffer{nullptr};
		VmaAllocation primitiveAllocation{VK_NULL_HANDLE};

		vk::Buffer materialBuffer{nullptr};
		VmaAllocation materialAllocation{VK_NULL_HANDLE};

		void CreateGpuBuffers();
	};

} // namespace brassica
