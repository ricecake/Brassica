#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

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

	class OzzModel {
	public:
		struct AnimationClip {
			std::string                                   name;
			ozz::unique_ptr<ozz::animation::Animation>    animation;
			float                                         duration{0.0f};
		};

		OzzModel() = default;
		explicit OzzModel(const std::filesystem::path& glbPath);
		~OzzModel();

		bool LoadFromGLB(const std::filesystem::path& glbPath);

		void Update(float dt, std::size_t animIndex = 0);
		void Update(float dt, const std::string& animName);

		[[nodiscard]] const std::vector<glm::vec3>&     GetPositions() const { return m_skinnedPositions; }
		[[nodiscard]] const std::vector<glm::vec3>&     GetNormals() const { return m_skinnedNormals; }
		[[nodiscard]] const std::vector<std::uint32_t>& GetIndices() const { return m_indices; }

		[[nodiscard]] std::size_t GetVertexCount() const { return m_skinnedPositions.size(); }
		[[nodiscard]] std::size_t GetTriangleCount() const { return m_indices.size() / 3; }

		[[nodiscard]] const ozz::animation::Skeleton* GetSkeleton() const { return m_skeleton.get(); }
		[[nodiscard]] std::size_t                     GetAnimationCount() const { return m_animations.size(); }
		[[nodiscard]] const std::vector<AnimationClip>& GetAnimations() const { return m_animations; }
		[[nodiscard]] int                             FindAnimationIndex(const std::string& name) const;

		[[nodiscard]] float GetPlaybackTime() const { return m_playbackTime; }

	private:
		void ProcessGLTF(const fastgltf::Asset& asset);

		ozz::unique_ptr<ozz::animation::Skeleton> m_skeleton;
		std::vector<AnimationClip>                 m_animations;
		std::unordered_map<std::string, std::size_t> m_animNameToIndex;

		std::vector<ozz::math::SoaTransform>      m_localTransforms;
		std::vector<ozz::math::Float4x4>          m_modelMatrices;
		std::vector<ozz::math::Float4x4>          m_inverseBindPoses;
		std::vector<ozz::math::Float4x4>          m_skinningMatrices;

		// Rest pose & Skinning data
		std::vector<glm::vec3>     m_restPositions;
		std::vector<glm::vec3>     m_restNormals;
		std::vector<std::uint16_t> m_jointIndices;
		std::vector<float>         m_jointWeights;
		std::vector<std::uint32_t> m_indices;

		// Skinned output
		std::vector<glm::vec3>     m_skinnedPositions;
		std::vector<glm::vec3>     m_skinnedNormals;

		// Playback state
		float                      m_playbackTime{0.0f};
		std::size_t                m_currentAnimIndex{0};
	};

} // namespace brassica
