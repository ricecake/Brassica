#include "gltf/GltfModel.hpp"
#include "gltf/GltfTextureManager.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <meshoptimizer.h>
#include "spdlog/spdlog.h"

namespace brassica {

	GltfModel::~GltfModel() {
		Destroy();
	}

	GltfModel::GltfModel(GltfModel&& other) noexcept {
		*this = std::move(other);
	}

	GltfModel& GltfModel::operator=(GltfModel&& other) noexcept {
		if (this != &other) {
			Destroy();

			device = other.device;
			allocator = other.allocator;

			vertices = std::move(other.vertices);
			meshlets = std::move(other.meshlets);
			meshletVertices = std::move(other.meshletVertices);
			meshletTriangles = std::move(other.meshletTriangles);
			primitives = std::move(other.primitives);
			materials = std::move(other.materials);
			nodes = std::move(other.nodes);
			skins = std::move(other.skins);
			animations = std::move(other.animations);

			vertexBuffer = other.vertexBuffer;
			vertexAllocation = other.vertexAllocation;
			other.vertexBuffer = nullptr;
			other.vertexAllocation = VK_NULL_HANDLE;

			meshletBuffer = other.meshletBuffer;
			meshletAllocation = other.meshletAllocation;
			other.meshletBuffer = nullptr;
			other.meshletAllocation = VK_NULL_HANDLE;

			meshletVerticesBuffer = other.meshletVerticesBuffer;
			meshletVerticesAllocation = other.meshletVerticesAllocation;
			other.meshletVerticesBuffer = nullptr;
			other.meshletVerticesAllocation = VK_NULL_HANDLE;

			meshletTrianglesBuffer = other.meshletTrianglesBuffer;
			meshletTrianglesAllocation = other.meshletTrianglesAllocation;
			other.meshletTrianglesBuffer = nullptr;
			other.meshletTrianglesAllocation = VK_NULL_HANDLE;

			primitiveBuffer = other.primitiveBuffer;
			primitiveAllocation = other.primitiveAllocation;
			other.primitiveBuffer = nullptr;
			other.primitiveAllocation = VK_NULL_HANDLE;

			materialBuffer = other.materialBuffer;
			materialAllocation = other.materialAllocation;
			other.materialBuffer = nullptr;
			other.materialAllocation = VK_NULL_HANDLE;
		}
		return *this;
	}

	void GltfModel::Destroy() {
		if (allocator != VK_NULL_HANDLE) {
			if (vertexBuffer) vmaDestroyBuffer(allocator, vertexBuffer, vertexAllocation);
			if (meshletBuffer) vmaDestroyBuffer(allocator, meshletBuffer, meshletAllocation);
			if (meshletVerticesBuffer) vmaDestroyBuffer(allocator, meshletVerticesBuffer, meshletVerticesAllocation);
			if (meshletTrianglesBuffer) vmaDestroyBuffer(allocator, meshletTrianglesBuffer, meshletTrianglesAllocation);
			if (primitiveBuffer) vmaDestroyBuffer(allocator, primitiveBuffer, primitiveAllocation);
			if (materialBuffer) vmaDestroyBuffer(allocator, materialBuffer, materialAllocation);
		}

		vertexBuffer = nullptr;
		vertexAllocation = VK_NULL_HANDLE;
		meshletBuffer = nullptr;
		meshletAllocation = VK_NULL_HANDLE;
		meshletVerticesBuffer = nullptr;
		meshletVerticesAllocation = VK_NULL_HANDLE;
		meshletTrianglesBuffer = nullptr;
		meshletTrianglesAllocation = VK_NULL_HANDLE;
		primitiveBuffer = nullptr;
		primitiveAllocation = VK_NULL_HANDLE;
		materialBuffer = nullptr;
		materialAllocation = VK_NULL_HANDLE;

		vertices.clear();
		meshlets.clear();
		meshletVertices.clear();
		meshletTriangles.clear();
		primitives.clear();
		materials.clear();
		nodes.clear();
		skins.clear();
		animations.clear();
	}

	bool GltfModel::LoadFromFile(
		vk::Device dev,
		VmaAllocator alloc,
		GltfTextureManager* textureManager,
		const std::string& filepath
	) {
		Destroy();
		device = dev;
		allocator = alloc;

		std::filesystem::path path(filepath);
		if (!std::filesystem::exists(path)) {
			spdlog::error("GltfModel file not found: {}", filepath);
			return false;
		}

		fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
		auto data = fastgltf::GltfDataBuffer::FromPath(path);
		if (data.error() != fastgltf::Error::None) {
			spdlog::error("Failed to load glTF file buffer: {}", filepath);
			return false;
		}

		auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages;
		auto assetExpected = parser.loadGltf(data.get(), path.parent_path(), options);
		if (assetExpected.error() != fastgltf::Error::None) {
			spdlog::error("Failed to parse glTF file: {} error: {}", filepath, static_cast<uint64_t>(assetExpected.error()));
			return false;
		}

		auto asset = std::move(assetExpected.get());

		// 1. Process Materials & Textures
		std::vector<int32_t> textureMap;
		if (textureManager) {
			textureMap.reserve(asset.textures.size());
			for (const auto& tex : asset.textures) {
				if (tex.imageIndex.has_value()) {
					size_t imgIdx = tex.imageIndex.value();
					const auto& img = asset.images[imgIdx];
					int32_t globalIdx = textureManager->LoadGltfImage(asset, img, path.parent_path().string());
					textureMap.push_back(globalIdx);
				} else {
					textureMap.push_back(-1);
				}
			}
		}

		materials.reserve(asset.materials.size());
		for (const auto& mat : asset.materials) {
			GltfMaterialGPU matGPU{};
			matGPU.baseColorFactor = glm::vec4(
				mat.pbrData.baseColorFactor[0],
				mat.pbrData.baseColorFactor[1],
				mat.pbrData.baseColorFactor[2],
				mat.pbrData.baseColorFactor[3]
			);
			matGPU.metallicFactor = mat.pbrData.metallicFactor;
			matGPU.roughnessFactor = mat.pbrData.roughnessFactor;

			if (mat.pbrData.baseColorTexture.has_value()) {
				size_t texIdx = mat.pbrData.baseColorTexture.value().textureIndex;
				if (texIdx < textureMap.size()) matGPU.baseColorTextureIndex = textureMap[texIdx];
			}
			if (mat.normalTexture.has_value()) {
				size_t texIdx = mat.normalTexture.value().textureIndex;
				if (texIdx < textureMap.size()) matGPU.normalTextureIndex = textureMap[texIdx];
			}
			if (mat.pbrData.metallicRoughnessTexture.has_value()) {
				size_t texIdx = mat.pbrData.metallicRoughnessTexture.value().textureIndex;
				if (texIdx < textureMap.size()) matGPU.metallicRoughnessTextureIndex = textureMap[texIdx];
			}

			materials.push_back(matGPU);
		}

		if (materials.empty()) {
			GltfMaterialGPU defaultMat{};
			materials.push_back(defaultMat);
		}

		// 2. Process Nodes
		nodes.reserve(asset.nodes.size());
		for (const auto& node : asset.nodes) {
			GltfNode gnode{};
			gnode.name = std::string(node.name);
			gnode.meshIndex = node.meshIndex.has_value() ? static_cast<int32_t>(node.meshIndex.value()) : -1;
			gnode.skinIndex = node.skinIndex.has_value() ? static_cast<int32_t>(node.skinIndex.value()) : -1;

			for (auto c : node.children) {
				gnode.children.push_back(static_cast<int32_t>(c));
			}

			if (auto* transform = std::get_if<fastgltf::TRS>(&node.transform)) {
				gnode.translation = glm::vec3(transform->translation[0], transform->translation[1], transform->translation[2]);
				gnode.rotation = glm::quat(transform->rotation[3], transform->rotation[0], transform->rotation[1], transform->rotation[2]);
				gnode.scale = glm::vec3(transform->scale[0], transform->scale[1], transform->scale[2]);

				glm::mat4 T = glm::translate(glm::mat4(1.0f), gnode.translation);
				glm::mat4 R = glm::mat4_cast(gnode.rotation);
				glm::mat4 S = glm::scale(glm::mat4(1.0f), gnode.scale);
				gnode.localMatrix = T * R * S;
			} else if (auto* mat = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
				for (int r = 0; r < 4; ++r) {
					for (int c = 0; c < 4; ++c) {
						gnode.localMatrix[c][r] = (*mat)[r][c];
					}
				}
			}

			nodes.push_back(gnode);
		}

		// Compute parent pointers
		for (size_t i = 0; i < nodes.size(); ++i) {
			for (int32_t child : nodes[i].children) {
				if (child >= 0 && child < static_cast<int32_t>(nodes.size())) {
					nodes[child].parent = static_cast<int32_t>(i);
				}
			}
		}

		// 3. Process Skins
		skins.reserve(asset.skins.size());
		for (const auto& skin : asset.skins) {
			GltfSkin gskin{};
			gskin.name = std::string(skin.name);
			gskin.skeletonRoot = skin.skeleton.has_value() ? static_cast<int32_t>(skin.skeleton.value()) : -1;

			for (auto j : skin.joints) {
				gskin.joints.push_back(static_cast<int32_t>(j));
			}

			if (skin.inverseBindMatrices.has_value()) {
				const auto& acc = asset.accessors[skin.inverseBindMatrices.value()];
				gskin.inverseBindMatrices.resize(acc.count);
				fastgltf::iterateAccessorWithIndex<glm::mat4>(asset, acc, [&](glm::mat4 ibm, size_t idx) {
					gskin.inverseBindMatrices[idx] = ibm;
				});
			} else {
				gskin.inverseBindMatrices.assign(gskin.joints.size(), glm::mat4(1.0f));
			}

			skins.push_back(gskin);
		}

		// 4. Process Animations
		animations.reserve(asset.animations.size());
		for (const auto& anim : asset.animations) {
			GltfAnimation ganim{};
			ganim.name = std::string(anim.name);

			for (const auto& channel : anim.channels) {
				if (!channel.nodeIndex.has_value()) continue;

				GltfAnimationChannel gchan{};
				gchan.targetNode = static_cast<int32_t>(channel.nodeIndex.value());

				switch (channel.path) {
					case fastgltf::AnimationPath::Translation: gchan.path = GltfTargetPath::Translation; break;
					case fastgltf::AnimationPath::Rotation:    gchan.path = GltfTargetPath::Rotation; break;
					case fastgltf::AnimationPath::Scale:       gchan.path = GltfTargetPath::Scale; break;
					default: continue;
				}

				const auto& sampler = anim.samplers[channel.samplerIndex];
				switch (sampler.interpolation) {
					case fastgltf::AnimationInterpolation::Step:        gchan.interpolation = GltfInterpolationType::Step; break;
					case fastgltf::AnimationInterpolation::CubicSpline: gchan.interpolation = GltfInterpolationType::Cubicspline; break;
					case fastgltf::AnimationInterpolation::Linear:
					default:                                            gchan.interpolation = GltfInterpolationType::Linear; break;
				}

				// Keyframe Timestamps
				const auto& inputAcc = asset.accessors[sampler.inputAccessor];
				gchan.samplersInput.resize(inputAcc.count);
				fastgltf::iterateAccessorWithIndex<float>(asset, inputAcc, [&](float timeVal, size_t idx) {
					gchan.samplersInput[idx] = timeVal;
					ganim.duration = std::max(ganim.duration, timeVal);
				});

				// Keyframe Values
				const auto& outputAcc = asset.accessors[sampler.outputAccessor];
				gchan.samplersOutput.resize(outputAcc.count);

				if (gchan.path == GltfTargetPath::Rotation) {
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, outputAcc, [&](glm::vec4 rotVal, size_t idx) {
						gchan.samplersOutput[idx] = rotVal;
					});
				} else {
					fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, outputAcc, [&](glm::vec3 posScaleVal, size_t idx) {
						gchan.samplersOutput[idx] = glm::vec4(posScaleVal, 0.0f);
					});
				}

				ganim.channels.push_back(gchan);
			}

			animations.push_back(ganim);
		}

		// 5. Process Meshes & Primitives -> Generate Meshlets via meshoptimizer
		constexpr size_t maxMeshletVertices = 64;
		constexpr size_t maxMeshletTriangles = 124;

		for (const auto& mesh : asset.meshes) {
			for (const auto& prim : mesh.primitives) {
				auto posIt = prim.findAttribute("POSITION");
				if (posIt == prim.attributes.end()) continue;

				size_t firstVertex = vertices.size();
				size_t vertexCount = asset.accessors[posIt->accessorIndex].count;

				std::vector<GltfVertex> primVertices(vertexCount);

				// Extract Positions
				const auto& posAcc = asset.accessors[posIt->accessorIndex];
				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAcc, [&](glm::vec3 pos, size_t idx) {
					primVertices[idx].position = pos;
				});

				// Extract Normals
				auto normIt = prim.findAttribute("NORMAL");
				if (normIt != prim.attributes.end()) {
					const auto& normAcc = asset.accessors[normIt->accessorIndex];
					fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, normAcc, [&](glm::vec3 norm, size_t idx) {
						primVertices[idx].normal = norm;
					});
				}

				// Extract Tangents
				auto tanIt = prim.findAttribute("TANGENT");
				if (tanIt != prim.attributes.end()) {
					const auto& tanAcc = asset.accessors[tanIt->accessorIndex];
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, tanAcc, [&](glm::vec4 tan, size_t idx) {
						primVertices[idx].tangent = tan;
					});
				}

				// Extract Texcoords
				auto texIt = prim.findAttribute("TEXCOORD_0");
				if (texIt != prim.attributes.end()) {
					const auto& texAcc = asset.accessors[texIt->accessorIndex];
					fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, texAcc, [&](glm::vec2 uv, size_t idx) {
						primVertices[idx].texcoord = uv;
					});
				}

				// Extract Joint Indices
				auto jointIt = prim.findAttribute("JOINTS_0");
				if (jointIt != prim.attributes.end()) {
					const auto& jointAcc = asset.accessors[jointIt->accessorIndex];
					fastgltf::iterateAccessorWithIndex<glm::uvec4>(asset, jointAcc, [&](glm::uvec4 j, size_t idx) {
						primVertices[idx].joints = j;
					});
				}

				// Extract Joint Weights
				auto weightIt = prim.findAttribute("WEIGHTS_0");
				if (weightIt != prim.attributes.end()) {
					const auto& weightAcc = asset.accessors[weightIt->accessorIndex];
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, weightAcc, [&](glm::vec4 w, size_t idx) {
						primVertices[idx].weights = w;
					});
				}

				// Extract Indices
				std::vector<uint32_t> primIndices;
				if (prim.indicesAccessor.has_value()) {
					const auto& idxAcc = asset.accessors[prim.indicesAccessor.value()];
					primIndices.resize(idxAcc.count);
					fastgltf::iterateAccessorWithIndex<uint32_t>(asset, idxAcc, [&](uint32_t i, size_t idx) {
						primIndices[idx] = i;
					});
				} else {
					primIndices.resize(vertexCount);
					for (uint32_t i = 0; i < vertexCount; ++i) primIndices[i] = i;
				}

				vertices.insert(vertices.end(), primVertices.begin(), primVertices.end());

				// Build meshlets using meshoptimizer
				size_t maxMeshlets = meshopt_buildMeshletsBound(primIndices.size(), maxMeshletVertices, maxMeshletTriangles);
				std::vector<meshopt_Meshlet> localMeshlets(maxMeshlets);
				std::vector<unsigned int> localMeshletVertices(maxMeshlets * maxMeshletVertices);
				std::vector<unsigned char> localMeshletTriangles(maxMeshlets * maxMeshletTriangles * 3);

				size_t meshletCount = meshopt_buildMeshlets(
					localMeshlets.data(),
					localMeshletVertices.data(),
					localMeshletTriangles.data(),
					primIndices.data(),
					primIndices.size(),
					&primVertices[0].position.x,
					primVertices.size(),
					sizeof(GltfVertex),
					maxMeshletVertices,
					maxMeshletTriangles,
					0.0f
				);

				localMeshlets.resize(meshletCount);

				GltfPrimitiveGPU gpuPrim{};
				gpuPrim.firstMeshlet = static_cast<uint32_t>(meshlets.size());
				gpuPrim.meshletCount = static_cast<uint32_t>(meshletCount);
				gpuPrim.materialIndex = prim.materialIndex.has_value() ? static_cast<uint32_t>(prim.materialIndex.value()) : 0;
				gpuPrim.firstVertex = static_cast<uint32_t>(firstVertex);
				gpuPrim.vertexCount = static_cast<uint32_t>(vertexCount);

				for (size_t m = 0; m < meshletCount; ++m) {
					const auto& lm = localMeshlets[m];

					uint32_t meshletVertOffset = static_cast<uint32_t>(meshletVertices.size());
					uint32_t meshletTriOffset = static_cast<uint32_t>(meshletTriangles.size());

					for (unsigned int v = 0; v < lm.vertex_count; ++v) {
						meshletVertices.push_back(static_cast<uint32_t>(firstVertex + localMeshletVertices[lm.vertex_offset + v]));
					}

					for (unsigned int t = 0; t < lm.triangle_count * 3; ++t) {
						meshletTriangles.push_back(localMeshletTriangles[lm.triangle_offset + t]);
					}

					// Compute meshlet bounding sphere
					meshopt_Bounds bounds = meshopt_computeMeshletBounds(
						&localMeshletVertices[lm.vertex_offset],
						&localMeshletTriangles[lm.triangle_offset],
						lm.triangle_count,
						&primVertices[0].position.x,
						primVertices.size(),
						sizeof(GltfVertex)
					);

					GltfMeshlet gmeshlet{};
					gmeshlet.vertexOffset = meshletVertOffset;
					gmeshlet.triangleOffset = meshletTriOffset;
					gmeshlet.vertexCount = lm.vertex_count;
					gmeshlet.triangleCount = lm.triangle_count;
					gmeshlet.center = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
					gmeshlet.radius = bounds.radius;

					meshlets.push_back(gmeshlet);
				}

				primitives.push_back(gpuPrim);
			}
		}

		CreateGpuBuffers();

		spdlog::info("Loaded glTF model '{}': {} vertices, {} meshlets, {} primitives, {} nodes, {} animations.",
			filepath, vertices.size(), meshlets.size(), primitives.size(), nodes.size(), animations.size());

		return true;
	}

	void GltfModel::CreateGpuBuffers() {
		auto createBuffer = [this](size_t bufferSize, vk::BufferUsageFlags usage, const void* data, vk::Buffer& outBuffer, VmaAllocation& outAlloc) {
			if (bufferSize == 0) return;

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = bufferSize;
			bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer rawBuffer = VK_NULL_HANDLE;
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &rawBuffer, &outAlloc, &allocResultInfo) == VK_SUCCESS) {
				outBuffer = rawBuffer;
				if (data && allocResultInfo.pMappedData) {
					std::memcpy(allocResultInfo.pMappedData, data, bufferSize);
				}
			} else {
				spdlog::error("Failed to allocate glTF GPU buffer of size {}", bufferSize);
			}
		};

		vk::BufferUsageFlags ssboUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

		createBuffer(vertices.size() * sizeof(GltfVertex), ssboUsage, vertices.data(), vertexBuffer, vertexAllocation);
		createBuffer(meshlets.size() * sizeof(GltfMeshlet), ssboUsage, meshlets.data(), meshletBuffer, meshletAllocation);
		createBuffer(meshletVertices.size() * sizeof(uint32_t), ssboUsage, meshletVertices.data(), meshletVerticesBuffer, meshletVerticesAllocation);
		createBuffer(meshletTriangles.size() * sizeof(uint8_t), ssboUsage, meshletTriangles.data(), meshletTrianglesBuffer, meshletTrianglesAllocation);
		createBuffer(primitives.size() * sizeof(GltfPrimitiveGPU), ssboUsage, primitives.data(), primitiveBuffer, primitiveAllocation);
		createBuffer(materials.size() * sizeof(GltfMaterialGPU), ssboUsage, materials.data(), materialBuffer, materialAllocation);
	}

} // namespace brassica
