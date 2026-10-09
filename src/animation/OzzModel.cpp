#include "animation/OzzModel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/util.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include <meshoptimizer.h>

#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>

#include "spdlog/spdlog.h"

namespace brassica {

	OzzModel::OzzModel(const std::filesystem::path& glbPath) {
		if (!LoadFromGLB(glbPath)) {
			spdlog::error("Failed to load GLB model from path: {}", glbPath.string());
		}
	}

	OzzModel::~OzzModel() = default;

	bool OzzModel::LoadFromGLB(const std::filesystem::path& glbPath) {
		std::filesystem::path actualPath = glbPath;
		if (!std::filesystem::exists(actualPath)) {
			actualPath = std::filesystem::path("..") / glbPath;
		}
		if (!std::filesystem::exists(actualPath)) {
			spdlog::error("GLB file not found: {}", glbPath.string());
			return false;
		}

		auto data = fastgltf::GltfDataBuffer::FromPath(actualPath);
		if (data.error() != fastgltf::Error::None) {
			spdlog::error("Failed to read GLB data buffer: {}", glbPath.string());
			return false;
		}

		fastgltf::Parser parser;
		auto assetResult = parser.loadGltfBinary(
			data.get(),
			actualPath.parent_path(),
			fastgltf::Options::LoadExternalBuffers | fastgltf::Options::DecomposeNodeMatrices
		);

		if (assetResult.error() != fastgltf::Error::None) {
			spdlog::error("fastgltf failed to parse GLB: {}", glbPath.string());
			return false;
		}

		ProcessGLTF(assetResult.get());
		OptimizeMeshAndBuildMeshlets();
		return true;
	}

	void OzzModel::ProcessGLTF(const fastgltf::Asset& asset) {
		// 1. Process Skeletons / Joints
		ozz::animation::offline::RawSkeleton rawSkeleton;

		std::vector<int> gltfNodeToSkinJointIndex(asset.nodes.size(), -1);

		if (!asset.skins.empty()) {
			const auto& skin = asset.skins[0];

			// Map skin joints
			for (std::size_t i = 0; i < skin.joints.size(); ++i) {
				gltfNodeToSkinJointIndex[skin.joints[i]] = static_cast<int>(i);
			}

			// Build joint hierarchy for Ozz RawSkeleton
			std::vector<bool> isChild(skin.joints.size(), false);
			for (std::size_t i = 0; i < skin.joints.size(); ++i) {
				std::size_t nodeIdx = skin.joints[i];
				const auto& node = asset.nodes[nodeIdx];
				for (std::size_t childNodeIdx : node.children) {
					int childJointIdx = gltfNodeToSkinJointIndex[childNodeIdx];
					if (childJointIdx >= 0) {
						isChild[childJointIdx] = true;
					}
				}
			}

			auto buildRawJoint = [&](auto self, std::size_t nodeIdx) -> ozz::animation::offline::RawSkeleton::Joint {
				const auto& node = asset.nodes[nodeIdx];
				ozz::animation::offline::RawSkeleton::Joint rawJoint;
				rawJoint.name = node.name.c_str();

				// Get transform (always TRS because of DecomposeNodeMatrices)
				if (auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
					rawJoint.transform.translation = ozz::math::Float3(trs->translation[0], trs->translation[1], trs->translation[2]);
					rawJoint.transform.rotation = ozz::math::Quaternion(trs->rotation[0], trs->rotation[1], trs->rotation[2], trs->rotation[3]);
					rawJoint.transform.scale = ozz::math::Float3(trs->scale[0], trs->scale[1], trs->scale[2]);
				} else if (auto* mat = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
					glm::mat4 m(1.0f);
					for (int c = 0; c < 4; ++c) {
						for (int r = 0; r < 4; ++r) {
							m[c][r] = (*mat)[c][r];
						}
					}
					glm::vec3 scale(1.0f);
					glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
					glm::vec3 translation(0.0f);
					glm::vec3 skew;
					glm::vec4 perspective;
					glm::decompose(m, scale, rotation, translation, skew, perspective);

					rawJoint.transform.translation = ozz::math::Float3(translation.x, translation.y, translation.z);
					rawJoint.transform.rotation = ozz::math::Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
					rawJoint.transform.scale = ozz::math::Float3(scale.x, scale.y, scale.z);
				}

				for (std::size_t childNodeIdx : node.children) {
					int childJointIdx = gltfNodeToSkinJointIndex[childNodeIdx];
					if (childJointIdx >= 0) {
						rawJoint.children.push_back(self(self, childNodeIdx));
					}
				}
				return rawJoint;
			};

			for (std::size_t i = 0; i < skin.joints.size(); ++i) {
				if (!isChild[i]) {
					rawSkeleton.roots.push_back(buildRawJoint(buildRawJoint, skin.joints[i]));
				}
			}
		} else {
			// Single root joint if no skin
			ozz::animation::offline::RawSkeleton::Joint root;
			root.name = "Root";
			root.transform.translation = ozz::math::Float3(0, 0, 0);
			root.transform.rotation = ozz::math::Quaternion::identity();
			root.transform.scale = ozz::math::Float3(1, 1, 1);
			rawSkeleton.roots.push_back(root);
		}

		if (!rawSkeleton.Validate()) {
			spdlog::warn("Ozz RawSkeleton validation warning, attempting to build anyway.");
		}

		ozz::animation::offline::SkeletonBuilder skelBuilder;
		m_skeleton = skelBuilder(rawSkeleton);
		if (!m_skeleton) {
			spdlog::error("Failed to build Ozz Skeleton from GLB.");
			return;
		}

		std::size_t numJoints = m_skeleton->num_joints();
		m_inverseBindPoses.resize(numJoints);

		// Build GLTF Skin Joint Index -> Ozz Skeleton Joint Index mapping
		std::vector<int> gltfSkinJointToOzzJoint(numJoints, 0);
		for (std::size_t skinJointIdx = 0; skinJointIdx < numJoints; ++skinJointIdx) {
			std::size_t nodeIdx = asset.skins[0].joints[skinJointIdx];
			const std::string& name = asset.nodes[nodeIdx].name.c_str();
			for (int ozzIdx = 0; ozzIdx < static_cast<int>(numJoints); ++ozzIdx) {
				if (std::string(m_skeleton->joint_names()[ozzIdx]) == name) {
					gltfSkinJointToOzzJoint[skinJointIdx] = ozzIdx;
					break;
				}
			}
		}

		// Read Inverse Bind Matrices if available
		if (!asset.skins.empty() && asset.skins[0].inverseBindMatrices.has_value()) {
			const auto& accessor = asset.accessors[asset.skins[0].inverseBindMatrices.value()];
			std::size_t skinJointIndex = 0;
			fastgltf::iterateAccessor<glm::mat4>(asset, accessor, [&](glm::mat4 ibm) {
				if (skinJointIndex < numJoints) {
					int ozzIdx = gltfSkinJointToOzzJoint[skinJointIndex];
					for (int c = 0; c < 4; ++c) {
						m_inverseBindPoses[ozzIdx].cols[c] = ozz::math::simd_float4::Load(ibm[c][0], ibm[c][1], ibm[c][2], ibm[c][3]);
					}
					skinJointIndex++;
				}
			});
		} else {
			// Compute default inverse bind poses from rest pose
			std::vector<ozz::math::SoaTransform> restSoa(m_skeleton->num_soa_joints());
			std::vector<ozz::math::Float4x4> restModel(numJoints);
			for (std::size_t i = 0; i < m_skeleton->num_soa_joints(); ++i) {
				restSoa[i] = m_skeleton->joint_rest_poses()[i];
			}
			ozz::animation::LocalToModelJob ltmBindJob;
			ltmBindJob.skeleton = m_skeleton.get();
			ltmBindJob.input = ozz::make_span(restSoa);
			ltmBindJob.output = ozz::make_span(restModel);
			ltmBindJob.Run();

			for (std::size_t i = 0; i < numJoints; ++i) {
				m_inverseBindPoses[i] = ozz::math::Invert(restModel[i]);
			}
		}

		// 2. Process Animations
		m_animations.clear();
		m_animNameToIndex.clear();

		std::vector<int> gltfNodeToOzzJointIndex(asset.nodes.size(), -1);
		for (std::size_t nodeIdx = 0; nodeIdx < asset.nodes.size(); ++nodeIdx) {
			const std::string& name = asset.nodes[nodeIdx].name.c_str();
			for (int ozzIdx = 0; ozzIdx < static_cast<int>(numJoints); ++ozzIdx) {
				if (std::string(m_skeleton->joint_names()[ozzIdx]) == name) {
					gltfNodeToOzzJointIndex[nodeIdx] = ozzIdx;
					break;
				}
			}
		}

		for (const auto& gltfAnim : asset.animations) {
			ozz::animation::offline::RawAnimation rawAnim;
			rawAnim.name = gltfAnim.name.c_str();

			// Map joint tracks by skeleton joint names
			rawAnim.tracks.resize(numJoints);

			float maxDuration = 0.0f;

			for (const auto& channel : gltfAnim.channels) {
				if (!channel.nodeIndex.has_value()) continue;
				int ozzJointIdx = gltfNodeToOzzJointIndex[channel.nodeIndex.value()];
				if (ozzJointIdx < 0 || ozzJointIdx >= static_cast<int>(numJoints)) continue;

				const auto& sampler = gltfAnim.samplers[channel.samplerIndex];
				const auto& inputAcc = asset.accessors[sampler.inputAccessor];
				const auto& outputAcc = asset.accessors[sampler.outputAccessor];

				std::vector<float> times;
				times.reserve(inputAcc.count);
				fastgltf::iterateAccessor<float>(asset, inputAcc, [&](float timeVal) {
					times.push_back(timeVal);
					if (timeVal > maxDuration) maxDuration = timeVal;
				});

				auto& track = rawAnim.tracks[ozzJointIdx];

				if (channel.path == fastgltf::AnimationPath::Translation) {
					std::size_t idx = 0;
					fastgltf::iterateAccessor<glm::vec3>(asset, outputAcc, [&](glm::vec3 translation) {
						if (idx < times.size()) {
							ozz::animation::offline::RawAnimation::TranslationKey k;
							k.time = times[idx++];
							k.value = ozz::math::Float3(translation.x, translation.y, translation.z);
							track.translations.push_back(k);
						}
					});
				} else if (channel.path == fastgltf::AnimationPath::Rotation) {
					std::size_t idx = 0;
					fastgltf::iterateAccessor<glm::vec4>(asset, outputAcc, [&](glm::vec4 rotation) {
						if (idx < times.size()) {
							ozz::animation::offline::RawAnimation::RotationKey k;
							k.time = times[idx++];
							k.value = ozz::math::Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
							track.rotations.push_back(k);
						}
					});
				} else if (channel.path == fastgltf::AnimationPath::Scale) {
					std::size_t idx = 0;
					fastgltf::iterateAccessor<glm::vec3>(asset, outputAcc, [&](glm::vec3 scale) {
						if (idx < times.size()) {
							ozz::animation::offline::RawAnimation::ScaleKey k;
							k.time = times[idx++];
							k.value = ozz::math::Float3(scale.x, scale.y, scale.z);
							track.scales.push_back(k);
						}
					});
				}
			}

			rawAnim.duration = maxDuration > 0.0f ? maxDuration : 1.0f;

			if (rawAnim.Validate()) {
				ozz::animation::offline::AnimationBuilder animBuilder;
				auto ozzAnim = animBuilder(rawAnim);
				if (ozzAnim) {
					AnimationClip clip;
					clip.name = gltfAnim.name.empty() ? ("Anim_" + std::to_string(m_animations.size())) : std::string(gltfAnim.name);
					clip.duration = rawAnim.duration;
					clip.animation = std::move(ozzAnim);

					m_animNameToIndex[clip.name] = m_animations.size();
					m_animations.push_back(std::move(clip));
				}
			}
		}

		// 3. Process Mesh Data (Positions, Normals, Joints, Weights, Indices)
		m_restPositions.clear();
		m_restNormals.clear();
		m_jointIndices.clear();
		m_jointWeights.clear();
		m_indices.clear();

		std::uint32_t vertexOffset = 0;

		for (const auto& mesh : asset.meshes) {
			for (const auto& prim : mesh.primitives) {
				auto posIt = prim.findAttribute("POSITION");
				if (posIt == prim.attributes.end()) continue;

				const auto& posAcc = asset.accessors[posIt->accessorIndex];
				std::size_t primVertexCount = posAcc.count;

				// Positions
				fastgltf::iterateAccessor<glm::vec3>(asset, posAcc, [&](glm::vec3 pos) {
					m_restPositions.push_back(pos);
				});

				// Normals
				auto normIt = prim.findAttribute("NORMAL");
				if (normIt != prim.attributes.end()) {
					const auto& normAcc = asset.accessors[normIt->accessorIndex];
					fastgltf::iterateAccessor<glm::vec3>(asset, normAcc, [&](glm::vec3 norm) {
						m_restNormals.push_back(glm::normalize(norm));
					});
				} else {
					m_restNormals.resize(m_restPositions.size(), glm::vec3(0.0f, 1.0f, 0.0f));
				}

				// Joint Indices (4 per vertex, remapped from GLTF skin joint -> Ozz skeleton joint)
				auto jointIt = prim.findAttribute("JOINTS_0");
				if (jointIt != prim.attributes.end()) {
					const auto& jointAcc = asset.accessors[jointIt->accessorIndex];
					fastgltf::iterateAccessor<glm::uvec4>(asset, jointAcc, [&](glm::uvec4 j) {
						m_jointIndices.push_back(static_cast<std::uint16_t>(gltfSkinJointToOzzJoint[j.x]));
						m_jointIndices.push_back(static_cast<std::uint16_t>(gltfSkinJointToOzzJoint[j.y]));
						m_jointIndices.push_back(static_cast<std::uint16_t>(gltfSkinJointToOzzJoint[j.z]));
						m_jointIndices.push_back(static_cast<std::uint16_t>(gltfSkinJointToOzzJoint[j.w]));
					});
				} else {
					for (std::size_t i = 0; i < primVertexCount; ++i) {
						m_jointIndices.push_back(0);
						m_jointIndices.push_back(0);
						m_jointIndices.push_back(0);
						m_jointIndices.push_back(0);
					}
				}

				// Joint Weights (4 per vertex, for 4 influences, ozz skinning stores 3 weights)
				auto weightIt = prim.findAttribute("WEIGHTS_0");
				if (weightIt != prim.attributes.end()) {
					const auto& weightAcc = asset.accessors[weightIt->accessorIndex];
					fastgltf::iterateAccessor<glm::vec4>(asset, weightAcc, [&](glm::vec4 w) {
						float sum = w.x + w.y + w.z + w.w;
						if (sum > 0.0001f) w /= sum;
						m_jointWeights.push_back(w.x);
						m_jointWeights.push_back(w.y);
						m_jointWeights.push_back(w.z);
					});
				} else {
					for (std::size_t i = 0; i < primVertexCount; ++i) {
						m_jointWeights.push_back(1.0f);
						m_jointWeights.push_back(0.0f);
						m_jointWeights.push_back(0.0f);
					}
				}

				// Primitive Indices
				if (prim.indicesAccessor.has_value()) {
					const auto& idxAcc = asset.accessors[prim.indicesAccessor.value()];
					fastgltf::iterateAccessor<std::uint32_t>(asset, idxAcc, [&](std::uint32_t idx) {
						m_indices.push_back(vertexOffset + idx);
					});
				} else {
					for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(primVertexCount); ++i) {
						m_indices.push_back(vertexOffset + i);
					}
				}

				vertexOffset += static_cast<std::uint32_t>(primVertexCount);
			}
		}
	}

	void OzzModel::OptimizeMeshAndBuildMeshlets() {
		if (m_indices.empty() || m_restPositions.empty()) return;

		std::size_t indexCount = m_indices.size();
		std::size_t vertexCount = m_restPositions.size();

		// 1. Optimize vertex cache
		std::vector<std::uint32_t> cacheOptIndices(indexCount);
		meshopt_optimizeVertexCache(cacheOptIndices.data(), m_indices.data(), indexCount, vertexCount);

		// 2. Build temporary struct for overdraw / fetch optimization
		struct TempVert {
			glm::vec3 pos;
			glm::vec3 norm;
			std::uint16_t j[4];
			float w[3];
		};

		std::vector<TempVert> tempVerts(vertexCount);
		for (std::size_t i = 0; i < vertexCount; ++i) {
			tempVerts[i].pos = m_restPositions[i];
			tempVerts[i].norm = m_restNormals[i];
			tempVerts[i].j[0] = m_jointIndices[i * 4 + 0];
			tempVerts[i].j[1] = m_jointIndices[i * 4 + 1];
			tempVerts[i].j[2] = m_jointIndices[i * 4 + 2];
			tempVerts[i].j[3] = m_jointIndices[i * 4 + 3];
			tempVerts[i].w[0] = m_jointWeights[i * 3 + 0];
			tempVerts[i].w[1] = m_jointWeights[i * 3 + 1];
			tempVerts[i].w[2] = m_jointWeights[i * 3 + 2];
		}

		// Optimize overdraw
		std::vector<std::uint32_t> overdrawOptIndices(indexCount);
		meshopt_optimizeOverdraw(
			overdrawOptIndices.data(),
			cacheOptIndices.data(),
			indexCount,
			&tempVerts[0].pos.x,
			vertexCount,
			sizeof(TempVert),
			1.05f
		);

		// Optimize vertex fetch
		std::vector<TempVert> optVerts(vertexCount);
		meshopt_optimizeVertexFetch(
			optVerts.data(),
			overdrawOptIndices.data(),
			indexCount,
			tempVerts.data(),
			vertexCount,
			sizeof(TempVert)
		);

		m_indices = std::move(overdrawOptIndices);

		m_restPositions.resize(vertexCount);
		m_restNormals.resize(vertexCount);
		m_jointIndices.resize(vertexCount * 4);
		m_jointWeights.resize(vertexCount * 3);
		m_restVertices.resize(vertexCount);

		for (std::size_t i = 0; i < vertexCount; ++i) {
			m_restPositions[i] = optVerts[i].pos;
			m_restNormals[i] = optVerts[i].norm;
			m_jointIndices[i * 4 + 0] = optVerts[i].j[0];
			m_jointIndices[i * 4 + 1] = optVerts[i].j[1];
			m_jointIndices[i * 4 + 2] = optVerts[i].j[2];
			m_jointIndices[i * 4 + 3] = optVerts[i].j[3];
			m_jointWeights[i * 3 + 0] = optVerts[i].w[0];
			m_jointWeights[i * 3 + 1] = optVerts[i].w[1];
			m_jointWeights[i * 3 + 2] = optVerts[i].w[2];

			m_restVertices[i].position = glm::vec4(optVerts[i].pos, 1.0f);
			m_restVertices[i].normal = glm::vec4(optVerts[i].norm, 0.0f);
		}

		// 3. Build Meshlets using meshoptimizer
		constexpr std::size_t maxVertices = 64;
		constexpr std::size_t maxTriangles = 128;
		constexpr float coneWeight = 0.0f;

		std::size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, maxVertices, maxTriangles);

		std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
		m_meshletVertices.resize(maxMeshlets * maxVertices);
		m_meshletTriangles.resize(maxMeshlets * maxTriangles * 3);

		std::size_t meshletCount = meshopt_buildMeshlets(
			rawMeshlets.data(),
			m_meshletVertices.data(),
			m_meshletTriangles.data(),
			m_indices.data(),
			indexCount,
			&m_restVertices[0].position.x,
			vertexCount,
			sizeof(ModelVertex),
			maxVertices,
			maxTriangles,
			coneWeight
		);

		m_meshlets.resize(meshletCount);
		for (std::size_t i = 0; i < meshletCount; ++i) {
			m_meshlets[i].vertexOffset = rawMeshlets[i].vertex_offset;
			m_meshlets[i].triangleOffset = rawMeshlets[i].triangle_offset;
			m_meshlets[i].vertexCount = rawMeshlets[i].vertex_count;
			m_meshlets[i].triangleCount = rawMeshlets[i].triangle_count;
		}

		const auto& lastMeshlet = rawMeshlets[meshletCount - 1];
		m_meshletVertices.resize(lastMeshlet.vertex_offset + lastMeshlet.vertex_count);
		m_meshletTriangles.resize(lastMeshlet.triangle_offset + lastMeshlet.triangle_count * 3);
	}

	int OzzModel::FindAnimationIndex(const std::string& name) const {
		auto it = m_animNameToIndex.find(name);
		if (it != m_animNameToIndex.end()) {
			return static_cast<int>(it->second);
		}
		for (std::size_t i = 0; i < m_animations.size(); ++i) {
			if (m_animations[i].name.find(name) != std::string::npos) {
				return static_cast<int>(i);
			}
		}
		return -1;
	}

	OzzModelInstance OzzModel::CreateInstance() const {
		OzzModelInstance inst;
		if (!m_skeleton) return inst;

		std::size_t numJoints = m_skeleton->num_joints();
		inst.localTransforms.resize(m_skeleton->num_soa_joints());
		inst.modelMatrices.resize(numJoints);
		inst.skinningMatrices.resize(numJoints);
		inst.skinnedPositions = m_restPositions;
		inst.skinnedNormals = m_restNormals;
		inst.skinnedVertices = m_restVertices;

		return inst;
	}

	void OzzModel::UpdateInstance(OzzModelInstance& instance, float dt, const std::string& animName) const {
		int idx = FindAnimationIndex(animName);
		UpdateInstance(instance, dt, idx >= 0 ? static_cast<std::size_t>(idx) : 0);
	}

	void OzzModel::UpdateInstance(OzzModelInstance& instance, float dt, std::size_t animIndex) const {
		if (!m_skeleton || m_animations.empty()) return;

		if (animIndex != instance.currentAnimIndex) {
			instance.currentAnimIndex = animIndex % m_animations.size();
			instance.playbackTime = 0.0f;
		} else {
			instance.playbackTime += dt;
		}

		const auto& clip = m_animations[instance.currentAnimIndex];
		if (clip.duration > 0.0f) {
			instance.playbackTime = std::fmod(instance.playbackTime, clip.duration);
		}

		// 1. Sampling Job
		ozz::animation::SamplingJob samplingJob;
		samplingJob.animation = clip.animation.get();
		samplingJob.ratio = clip.duration > 0.0f ? (instance.playbackTime / clip.duration) : 0.0f;
		samplingJob.output = ozz::make_span(instance.localTransforms);
		samplingJob.Run();

		// 2. Local-To-Model Job
		ozz::animation::LocalToModelJob ltmJob;
		ltmJob.skeleton = m_skeleton.get();
		ltmJob.input = ozz::make_span(instance.localTransforms);
		ltmJob.output = ozz::make_span(instance.modelMatrices);
		ltmJob.Run();

		// 3. Compute skinning matrices = model_matrix * inverse_bind_pose
		std::size_t numJoints = m_skeleton->num_joints();
		for (std::size_t i = 0; i < numJoints; ++i) {
			instance.skinningMatrices[i] = instance.modelMatrices[i] * m_inverseBindPoses[i];
		}

		// 4. Skinning Job
		if (!m_restPositions.empty()) {
			ozz::geometry::SkinningJob skinningJob;
			skinningJob.vertex_count = static_cast<int>(m_restPositions.size());
			skinningJob.influences_count = 4;

			skinningJob.joint_matrices = ozz::make_span(instance.skinningMatrices);

			skinningJob.joint_indices = ozz::make_span(m_jointIndices);
			skinningJob.joint_indices_stride = sizeof(std::uint16_t) * 4;

			skinningJob.joint_weights = ozz::make_span(m_jointWeights);
			skinningJob.joint_weights_stride = sizeof(float) * 3;

			skinningJob.in_positions = ozz::span<const float>(reinterpret_cast<const float*>(m_restPositions.data()), m_restPositions.size() * 3);
			skinningJob.in_positions_stride = sizeof(glm::vec3);
			skinningJob.out_positions = ozz::span<float>(reinterpret_cast<float*>(instance.skinnedPositions.data()), instance.skinnedPositions.size() * 3);
			skinningJob.out_positions_stride = sizeof(glm::vec3);

			skinningJob.in_normals = ozz::span<const float>(reinterpret_cast<const float*>(m_restNormals.data()), m_restNormals.size() * 3);
			skinningJob.in_normals_stride = sizeof(glm::vec3);
			skinningJob.out_normals = ozz::span<float>(reinterpret_cast<float*>(instance.skinnedNormals.data()), instance.skinnedNormals.size() * 3);
			skinningJob.out_normals_stride = sizeof(glm::vec3);

			skinningJob.Run();

			// Sync skinned positions and normals into ModelVertex buffer
			std::size_t vertexCount = m_restPositions.size();
			if (instance.skinnedVertices.size() != vertexCount) {
				instance.skinnedVertices.resize(vertexCount);
			}
			for (std::size_t i = 0; i < vertexCount; ++i) {
				instance.skinnedVertices[i].position = glm::vec4(instance.skinnedPositions[i], 1.0f);
				instance.skinnedVertices[i].normal = glm::vec4(instance.skinnedNormals[i], 0.0f);
			}
		}
	}

} // namespace brassica
