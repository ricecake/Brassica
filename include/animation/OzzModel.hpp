#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <meshoptimizer.h>

#include <ozz/animation/offline/animation_builder.h>
#include <ozz/animation/offline/raw_animation.h>
#include <ozz/animation/offline/raw_skeleton.h>
#include <ozz/animation/offline/skeleton_builder.h>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/memory/unique_ptr.h>
#include <ozz/geometry/runtime/skinning_job.h>

#include <glm/glm.hpp>

namespace brassica {

	struct AnimationClip {
		std::string                                   name;
		ozz::unique_ptr<ozz::animation::Animation>    animation;
		float                                         duration{0.0f};
	};

	struct ModelVertex {
		glm::vec4 position; // xyz = position, w = 1.0
		glm::vec4 normal;   // xyz = normal, w = 0.0
	};

	struct GPUMeshlet {
		std::uint32_t vertexOffset;
		std::uint32_t triangleOffset;
		std::uint32_t vertexCount;
		std::uint32_t triangleCount;
	};

	struct OzzModelInstance {
		float       playbackTime{0.0f};
		std::size_t currentAnimIndex{0};

		std::vector<ozz::math::SoaTransform> localTransforms;
		std::vector<ozz::math::Float4x4>     modelMatrices;
		std::vector<ozz::math::Float4x4>     skinningMatrices;

		std::vector<glm::vec3>   skinnedPositions;
		std::vector<glm::vec3>   skinnedNormals;
		std::vector<ModelVertex> skinnedVertices;
	};

	class OzzModel {
	public:
		OzzModel() = default;
		explicit OzzModel(const std::filesystem::path& glbPath);
		~OzzModel();

		bool LoadFromGLB(const std::filesystem::path& glbPath);

		OzzModelInstance CreateInstance() const;
		void UpdateInstance(OzzModelInstance& instance, float dt, std::size_t animIndex) const;
		void UpdateInstance(OzzModelInstance& instance, float dt, const std::string& animName) const;

		[[nodiscard]] const std::vector<glm::vec3>&     GetRestPositions() const { return m_restPositions; }
		[[nodiscard]] const std::vector<glm::vec3>&     GetRestNormals() const { return m_restNormals; }
		[[nodiscard]] const std::vector<ModelVertex>&   GetRestVertices() const { return m_restVertices; }
		[[nodiscard]] const std::vector<std::uint32_t>& GetIndices() const { return m_indices; }

		[[nodiscard]] const std::vector<GPUMeshlet>&     GetMeshlets() const { return m_meshlets; }
		[[nodiscard]] const std::vector<std::uint32_t>& GetMeshletVertices() const { return m_meshletVertices; }
		[[nodiscard]] const std::vector<std::uint8_t>&  GetMeshletTriangles() const { return m_meshletTriangles; }

		[[nodiscard]] std::size_t GetVertexCount() const { return m_restPositions.size(); }
		[[nodiscard]] std::size_t GetTriangleCount() const { return m_indices.size() / 3; }
		[[nodiscard]] std::size_t GetMeshletCount() const { return m_meshlets.size(); }

		[[nodiscard]] const ozz::animation::Skeleton* GetSkeleton() const { return m_skeleton.get(); }
		[[nodiscard]] std::size_t                     GetAnimationCount() const { return m_animations.size(); }
		[[nodiscard]] const std::vector<AnimationClip>& GetAnimations() const { return m_animations; }
		[[nodiscard]] int                             FindAnimationIndex(const std::string& name) const;

	private:
		void ProcessGLTF(const fastgltf::Asset& asset);
		void OptimizeMeshAndBuildMeshlets();

		ozz::unique_ptr<ozz::animation::Skeleton> m_skeleton;
		std::vector<AnimationClip>                 m_animations;
		std::unordered_map<std::string, std::size_t> m_animNameToIndex;

		std::vector<ozz::math::Float4x4>          m_inverseBindPoses;

		// Rest pose & Skinning data
		std::vector<glm::vec3>     m_restPositions;
		std::vector<glm::vec3>     m_restNormals;
		std::vector<ModelVertex>   m_restVertices;
		std::vector<std::uint16_t> m_jointIndices;
		std::vector<float>         m_jointWeights;
		std::vector<std::uint32_t> m_indices;

		// Meshlet data
		std::vector<GPUMeshlet>    m_meshlets;
		std::vector<std::uint32_t> m_meshletVertices;
		std::vector<std::uint8_t>  m_meshletTriangles;
	};

} // namespace brassica
