#include "animation/SkinnedCylinder.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <geometrycentral/surface/surface_mesh_factories.h>

#include <ozz/base/maths/simd_math.h>

namespace brassica {

	SkinnedCylinder::SkinnedCylinder(const Parameters& params)
		: m_params(params) {
		GenerateGeometryCentralMesh(m_params);
		BuildOzzSkeletonAndWeights(m_params);
		Update(0.0f);
	}

	SkinnedCylinder::~SkinnedCylinder() {
		m_skeleton.reset();
		m_geometry.reset();
		m_mesh.reset();
	}

	void SkinnedCylinder::GenerateGeometryCentralMesh(const Parameters& params) {
		std::vector<geometrycentral::Vector3> positions;
		std::vector<std::vector<std::size_t>> polygons;

		const std::size_t radialSegs = std::max<std::size_t>(3, params.radialSegments);
		const std::size_t heightSegs = std::max<std::size_t>(1, params.heightSegments);
		const float       radius = params.radius;
		const float       height = params.height;

		// Generate ring vertices
		for (std::size_t j = 0; j <= heightSegs; ++j) {
			float y = -height * 0.5f + static_cast<float>(j) * (height / static_cast<float>(heightSegs));
			for (std::size_t i = 0; i < radialSegs; ++i) {
				float angle = static_cast<float>(i) * (2.0f * std::numbers::pi_v<float> / static_cast<float>(radialSegs));
				float x = radius * std::cos(angle);
				float z = radius * std::sin(angle);
				positions.push_back(geometrycentral::Vector3{x, y, z});
			}
		}

		std::size_t topCenterIdx = positions.size();
		positions.push_back(geometrycentral::Vector3{0.0f, height * 0.5f, 0.0f});

		std::size_t bottomCenterIdx = positions.size();
		positions.push_back(geometrycentral::Vector3{0.0f, -height * 0.5f, 0.0f});

		// Quad faces for sides
		for (std::size_t j = 0; j < heightSegs; ++j) {
			for (std::size_t i = 0; i < radialSegs; ++i) {
				std::size_t nextI = (i + 1) % radialSegs;
				std::size_t v0 = j * radialSegs + i;
				std::size_t v1 = j * radialSegs + nextI;
				std::size_t v2 = (j + 1) * radialSegs + nextI;
				std::size_t v3 = (j + 1) * radialSegs + i;
				polygons.push_back({v0, v1, v2, v3});
			}
		}

		// Top cap triangles
		std::size_t topRingStart = heightSegs * radialSegs;
		for (std::size_t i = 0; i < radialSegs; ++i) {
			std::size_t nextI = (i + 1) % radialSegs;
			std::size_t v0 = topRingStart + i;
			std::size_t v1 = topRingStart + nextI;
			polygons.push_back({topCenterIdx, v0, v1});
		}

		// Bottom cap triangles
		std::size_t bottomRingStart = 0;
		for (std::size_t i = 0; i < radialSegs; ++i) {
			std::size_t nextI = (i + 1) % radialSegs;
			std::size_t v0 = bottomRingStart + i;
			std::size_t v1 = bottomRingStart + nextI;
			polygons.push_back({bottomCenterIdx, v1, v0});
		}

		m_geometry.reset();
		m_mesh.reset();

		auto manifoldMesh = std::make_unique<geometrycentral::surface::ManifoldSurfaceMesh>(polygons);
		auto geometry = std::make_unique<geometrycentral::surface::VertexPositionGeometry>(*manifoldMesh);
		for (auto v : manifoldMesh->vertices()) {
			geometry->vertexPositions[v] = positions[v.getIndex()];
		}

		m_mesh = std::move(manifoldMesh);
		m_geometry = std::move(geometry);
		m_geometry->requireVertexNormals();

		m_restPositions.clear();
		m_restNormals.clear();
		m_indices.clear();

		for (auto v : m_mesh->vertices()) {
			geometrycentral::Vector3 p = m_geometry->vertexPositions[v];
			geometrycentral::Vector3 n = m_geometry->vertexNormals[v];
			m_restPositions.emplace_back(p.x, p.y, p.z);
			m_restNormals.emplace_back(n.x, n.y, n.z);
		}

		for (auto f : m_mesh->faces()) {
			std::vector<std::uint32_t> faceIndices;
			for (auto v : f.adjacentVertices()) {
				faceIndices.push_back(static_cast<std::uint32_t>(v.getIndex()));
			}
			if (faceIndices.size() == 3) {
				m_indices.push_back(faceIndices[0]);
				m_indices.push_back(faceIndices[1]);
				m_indices.push_back(faceIndices[2]);
			} else if (faceIndices.size() == 4) {
				m_indices.push_back(faceIndices[0]);
				m_indices.push_back(faceIndices[1]);
				m_indices.push_back(faceIndices[2]);

				m_indices.push_back(faceIndices[0]);
				m_indices.push_back(faceIndices[2]);
				m_indices.push_back(faceIndices[3]);
			}
		}

		m_skinnedPositions = m_restPositions;
		m_skinnedNormals = m_restNormals;
	}

	void SkinnedCylinder::BuildOzzSkeletonAndWeights(const Parameters& params) {
		ozz::animation::offline::RawSkeleton rawSkeleton;

		// Root Joint 0 ("BaseJoint")
		ozz::animation::offline::RawSkeleton::Joint joint0;
		joint0.name = "BaseJoint";
		joint0.transform.translation = ozz::math::Float3(0.0f, -params.height * 0.5f, 0.0f);
		joint0.transform.rotation = ozz::math::Quaternion::identity();
		joint0.transform.scale = ozz::math::Float3(1.0f, 1.0f, 1.0f);

		// Child Joint 1 ("TopJoint")
		ozz::animation::offline::RawSkeleton::Joint joint1;
		joint1.name = "TopJoint";
		joint1.transform.translation = ozz::math::Float3(0.0f, params.height, 0.0f);
		joint1.transform.rotation = ozz::math::Quaternion::identity();
		joint1.transform.scale = ozz::math::Float3(1.0f, 1.0f, 1.0f);

		joint0.children.push_back(joint1);
		rawSkeleton.roots.push_back(joint0);

		if (!rawSkeleton.Validate()) {
			throw std::runtime_error("RawSkeleton validation failed for SkinnedCylinder.");
		}

		ozz::animation::offline::SkeletonBuilder builder;
		m_skeleton = builder(rawSkeleton);
		if (!m_skeleton) {
			throw std::runtime_error("Failed to build Skeleton for SkinnedCylinder.");
		}

		m_localTransforms.resize(m_skeleton->num_soa_joints());
		for (int i = 0; i < m_skeleton->num_soa_joints(); ++i) {
			m_localTransforms[i] = m_skeleton->joint_rest_poses()[i];
		}
		m_modelMatrices.resize(m_skeleton->num_joints());
		m_skinningMatrices.resize(m_skeleton->num_joints());
		m_inverseBindPoses.resize(m_skeleton->num_joints());

		// Compute initial bind pose matrices in model space and their inverses
		ozz::animation::LocalToModelJob ltmBindJob;
		ltmBindJob.skeleton = m_skeleton.get();
		ltmBindJob.input = ozz::make_span(m_localTransforms);
		ltmBindJob.output = ozz::make_span(m_modelMatrices);
		ltmBindJob.Run();

		for (int i = 0; i < m_skeleton->num_joints(); ++i) {
			m_inverseBindPoses[i] = ozz::math::Invert(m_modelMatrices[i]);
		}

		// Influences: 2 joints per vertex
		m_jointIndices.clear();
		m_jointWeights.clear();

		const float minY = -params.height * 0.5f;
		const float h = params.height;

		for (const auto& pos : m_restPositions) {
			float t = (pos.y - minY) / h;
			t = std::clamp(t, 0.0f, 1.0f);

			m_jointIndices.push_back(0); // BaseJoint
			m_jointIndices.push_back(1); // TopJoint

			// For 2 influences, joint_weights stores 1 weight per vertex (for Joint 0)
			m_jointWeights.push_back(1.0f - t);
		}
	}

	void SkinnedCylinder::Update(float timeSeconds) {
		// Update local joint transforms for animation
		std::vector<ozz::math::Transform> transforms(m_skeleton->num_joints());
		transforms[0].translation = ozz::math::Float3(0.0f, -m_params.height * 0.5f, 0.0f);
		transforms[0].rotation = ozz::math::Quaternion::identity();
		transforms[0].scale = ozz::math::Float3(1.0f, 1.0f, 1.0f);

		float angle = std::sin(timeSeconds * 2.0f) * 0.5f;
		transforms[1].translation = ozz::math::Float3(0.0f, m_params.height, 0.0f);
		transforms[1].rotation = ozz::math::Quaternion::FromAxisAngle(
			ozz::math::Float3(0.0f, 0.0f, 1.0f),
			angle
		);
		transforms[1].scale = ozz::math::Float3(1.0f, 1.0f, 1.0f);

		// Pack into SoaTransforms
		const ozz::math::SimdFloat4 w_axis = ozz::math::simd_float4::w_axis();
		const ozz::math::SimdFloat4 zero = ozz::math::simd_float4::zero();
		const ozz::math::SimdFloat4 one = ozz::math::simd_float4::one();

		const int numJoints = m_skeleton->num_joints();
		for (int i = 0; i < m_skeleton->num_soa_joints(); ++i) {
			ozz::math::SimdFloat4 translations[4];
			ozz::math::SimdFloat4 scales[4];
			ozz::math::SimdFloat4 rotations[4];

			for (int j = 0; j < 4; ++j) {
				int jointIdx = i * 4 + j;
				if (jointIdx < numJoints) {
					translations[j] = ozz::math::simd_float4::Load3PtrU(&transforms[jointIdx].translation.x);
					rotations[j] = ozz::math::NormalizeSafe4(
						ozz::math::simd_float4::LoadPtrU(&transforms[jointIdx].rotation.x),
						w_axis
					);
					scales[j] = ozz::math::simd_float4::Load3PtrU(&transforms[jointIdx].scale.x);
				} else {
					translations[j] = zero;
					rotations[j] = w_axis;
					scales[j] = one;
				}
			}

			ozz::math::Transpose4x3(translations, &m_localTransforms[i].translation.x);
			ozz::math::Transpose4x4(rotations, &m_localTransforms[i].rotation.x);
			ozz::math::Transpose4x3(scales, &m_localTransforms[i].scale.x);
		}

		// Compute model matrices
		ozz::animation::LocalToModelJob ltmJob;
		ltmJob.skeleton = m_skeleton.get();
		ltmJob.input = ozz::make_span(m_localTransforms);
		ltmJob.output = ozz::make_span(m_modelMatrices);
		ltmJob.Run();

		// Compute skinning matrices = model_matrix * inverse_bind_pose
		for (int i = 0; i < m_skeleton->num_joints(); ++i) {
			m_skinningMatrices[i] = m_modelMatrices[i] * m_inverseBindPoses[i];
		}

		// Run SkinningJob
		ozz::geometry::SkinningJob skinningJob;
		skinningJob.vertex_count = static_cast<int>(m_restPositions.size());
		skinningJob.influences_count = 2;
		skinningJob.joint_matrices = ozz::make_span(m_skinningMatrices);
		skinningJob.joint_indices = ozz::make_span(m_jointIndices);
		skinningJob.joint_indices_stride = sizeof(std::uint16_t) * 2;
		skinningJob.joint_weights = ozz::make_span(m_jointWeights);
		skinningJob.joint_weights_stride = sizeof(float) * 1;

		skinningJob.in_positions = ozz::span<const float>(reinterpret_cast<const float*>(m_restPositions.data()), m_restPositions.size() * 3);
		skinningJob.in_positions_stride = sizeof(glm::vec3);
		skinningJob.out_positions = ozz::span<float>(reinterpret_cast<float*>(m_skinnedPositions.data()), m_skinnedPositions.size() * 3);
		skinningJob.out_positions_stride = sizeof(glm::vec3);

		skinningJob.in_normals = ozz::span<const float>(reinterpret_cast<const float*>(m_restNormals.data()), m_restNormals.size() * 3);
		skinningJob.in_normals_stride = sizeof(glm::vec3);
		skinningJob.out_normals = ozz::span<float>(reinterpret_cast<float*>(m_skinnedNormals.data()), m_skinnedNormals.size() * 3);
		skinningJob.out_normals_stride = sizeof(glm::vec3);

		skinningJob.Run();
	}

} // namespace brassica
