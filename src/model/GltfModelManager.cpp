#include "model/GltfModelManager.hpp"

#include <cmath>

#include "graph/Graph.hpp"
#include "vulkan/vulkan.hpp"

namespace brassica {

	namespace {

		graph::ResourceDesc DeviceAddressBufferDesc(std::uint64_t byteSize) {
			graph::ResourceDesc desc = graph::StagedStorageBufferDesc(byteSize);
			desc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			return desc;
		}

		std::uint32_t ResolveTextureId(std::int32_t imageIndex, const std::vector<std::uint32_t>& bindlessIndices) {
			if (imageIndex < 0 || static_cast<std::size_t>(imageIndex) >= bindlessIndices.size()) {
				return UNBOUND_TEXTURE_ID;
			}
			return bindlessIndices[static_cast<std::size_t>(imageIndex)];
		}

	} // namespace

	bool GltfModelManager::LoadFromFile(
		const std::filesystem::path&     path,
		graph::PhysicalResourceRegistry& registry,
		MaterialManager&                 materialManager
	) {
		if (!m_model.LoadFromFile(path)) {
			return false;
		}

		std::vector<std::uint32_t> imageBindlessIndices = m_textureManager.UploadAll(registry, m_model.Images());

		// Registers into the *shared* MaterialManager (not a parallel GLTF-only buffer) -- see
		// this class's own header comment. localToGlobal[local material index] = the real
		// MaterialBuffer index GltfMeshlet.materialIndex needs; RemapMaterialIndices rewrites
		// every meshlet from the local index GltfModel assigned at parse time to this one.
		std::vector<std::uint32_t> localToGlobal;
		localToGlobal.reserve(m_model.Materials().size());
		for (const GltfMaterial& gm : m_model.Materials()) {
			MaterialData data{};
			data.textureIds = glm::uvec4(
				ResolveTextureId(gm.baseColorImageIndex, imageBindlessIndices),
				ResolveTextureId(gm.normalImageIndex, imageBindlessIndices),
				ResolveTextureId(gm.metallicRoughnessImageIndex, imageBindlessIndices),
				ResolveTextureId(gm.emissiveImageIndex, imageBindlessIndices)
			);
			// x: tint offset -- neutral (0), this is the palette-system's own knob, not a glTF
			// concept. y/z: roughness/metallic factors. w: emission strength -- neutral (1),
			// multiplies emissiveFactor in evaluateGltfEmissive.
			data.colorTintOffset = glm::vec4(0.0f, gm.roughnessFactor, gm.metallicFactor, 1.0f);
			data.baseColorFactor = gm.baseColorFactor;
			data.emissiveFactor = glm::vec4(gm.emissiveFactor, 0.0f);
			localToGlobal.push_back(materialManager.RegisterMaterial(data));
		}
		m_model.RemapMaterialIndices(localToGlobal);

		m_vertexBufNode.SetData(m_model.Vertices(), DeviceAddressBufferDesc(m_model.Vertices().size() * sizeof(GltfVertex)));
		m_meshletVertexBufNode.SetData(
			m_model.MeshletVertices(),
			DeviceAddressBufferDesc(m_model.MeshletVertices().size() * sizeof(std::uint32_t))
		);
		m_meshletTriangleBufNode.SetData(
			m_model.MeshletTrianglesPacked(),
			DeviceAddressBufferDesc(m_model.MeshletTrianglesPacked().size() * sizeof(std::uint32_t))
		);
		m_meshletBufNode.SetData(
			m_model.Meshlets(),
			DeviceAddressBufferDesc(m_model.Meshlets().size() * sizeof(GltfMeshlet))
		);

		// Seeds the joint matrix buffer with a real pose immediately (t=0 of whichever animation
		// is selected, or a single identity matrix if the model has no skin) -- UpdatePose is also
		// called every frame from here on (Engine.cpp), but without this the very first frame's
		// buffer would be empty/unset before that first call ever runs.
		UpdatePose(0.0f);
		return true;
	}

	void GltfModelManager::UpdatePose(float time) {
		std::vector<glm::mat4> jointMatrices;
		if (m_model.JointCount() == 0) {
			jointMatrices.assign(1, glm::mat4(1.0f));
		} else {
			float duration =
				m_currentAnimationIndex < m_model.Animations().size()
					? m_model.Animations()[m_currentAnimationIndex].duration
					: 0.0f;
			float wrappedTime = duration > 0.0f ? std::fmod(time, duration) : 0.0f;
			m_model.EvaluatePose(m_currentAnimationIndex, wrappedTime, jointMatrices);
		}
		m_jointMatrixBufNode.SetData(jointMatrices, DeviceAddressBufferDesc(jointMatrices.size() * sizeof(glm::mat4)));
	}

	void GltfModelManager::RegisterBufferNodes(graph::Graph& frameGraph) {
		frameGraph.RegisterRef(m_vertexBufNode);
		frameGraph.RegisterRef(m_meshletVertexBufNode);
		frameGraph.RegisterRef(m_meshletTriangleBufNode);
		frameGraph.RegisterRef(m_meshletBufNode);
		frameGraph.RegisterRef(m_jointMatrixBufNode);
	}

} // namespace brassica
