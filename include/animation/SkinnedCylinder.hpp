#pragma once

#include <memory>
#include <vector>

#include <geometrycentral/surface/surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <ozz/animation/offline/raw_skeleton.h>
#include <ozz/animation/offline/skeleton_builder.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/maths/vec_float.h>
#include <ozz/base/memory/unique_ptr.h>
#include <ozz/geometry/runtime/skinning_job.h>

#include <glm/glm.hpp>

namespace brassica {

	class SkinnedCylinder {
	public:
		struct Parameters {
			float       radius{1.0f};
			float       height{4.0f};
			std::size_t radialSegments{12};
			std::size_t heightSegments{8};
		};

		SkinnedCylinder() : SkinnedCylinder(Parameters{}) {}
		explicit SkinnedCylinder(const Parameters& params);
		~SkinnedCylinder();

		// Evaluates skeletal animation at specified time and updates skinned vertex positions and normals
		void Update(float timeSeconds);

		// Accessors
		[[nodiscard]] const std::vector<glm::vec3>&     GetPositions() const { return m_skinnedPositions; }
		[[nodiscard]] const std::vector<glm::vec3>&     GetNormals() const { return m_skinnedNormals; }
		[[nodiscard]] const std::vector<std::uint32_t>& GetIndices() const { return m_indices; }

		[[nodiscard]] geometrycentral::surface::SurfaceMesh* GetSurfaceMesh() const { return m_mesh.get(); }
		[[nodiscard]] geometrycentral::surface::VertexPositionGeometry* GetGeometry() const { return m_geometry.get(); }
		[[nodiscard]] const ozz::animation::Skeleton* GetSkeleton() const { return m_skeleton.get(); }

		[[nodiscard]] std::size_t GetVertexCount() const { return m_skinnedPositions.size(); }
		[[nodiscard]] std::size_t GetTriangleCount() const { return m_indices.size() / 3; }

	private:
		void GenerateGeometryCentralMesh(const Parameters& params);
		void BuildOzzSkeletonAndWeights(const Parameters& params);

		std::unique_ptr<geometrycentral::surface::SurfaceMesh>            m_mesh;
		std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> m_geometry;

		ozz::unique_ptr<ozz::animation::Skeleton> m_skeleton;
		std::vector<ozz::math::SoaTransform>      m_localTransforms;
		std::vector<ozz::math::Float4x4>          m_modelMatrices;
		std::vector<ozz::math::Float4x4>          m_inverseBindPoses;
		std::vector<ozz::math::Float4x4>          m_skinningMatrices;

		// Skinning inputs
		std::vector<glm::vec3>     m_restPositions;
		std::vector<glm::vec3>     m_restNormals;
		std::vector<std::uint16_t> m_jointIndices;
		std::vector<float>         m_jointWeights;
		std::vector<std::uint32_t> m_indices;

		// Skinning outputs
		std::vector<glm::vec3> m_skinnedPositions;
		std::vector<glm::vec3> m_skinnedNormals;

		Parameters m_params;
	};

} // namespace brassica
