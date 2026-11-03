#include "model/GltfModelManager.hpp"

#include "model/ModelAsset.hpp"

namespace brassica {

	namespace {

		std::uint32_t ResolveTextureId(std::int32_t imageIndex, const std::vector<std::uint32_t>& bindlessIndices) {
			if (imageIndex < 0 || static_cast<std::size_t>(imageIndex) >= bindlessIndices.size()) {
				return UNBOUND_TEXTURE_ID;
			}
			return bindlessIndices[static_cast<std::size_t>(imageIndex)];
		}

		// Model-local (pre instance-transform) bounds, the union of every rest-pose skinned
		// vertex -- same skin-and-bound pattern GltfModel::BakeClips uses per baked frame,
		// just for the rest pose (palette[0..jointCount)) specifically.
		void ComputeRestBounds(const std::vector<GltfVertex>& vertices, const std::vector<glm::mat4>& restPose, ModelAsset& asset) {
			glm::vec3 lo(1e30f);
			glm::vec3 hi(-1e30f);
			for (const GltfVertex& v : vertices) {
				float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
				glm::vec3 pos = v.position;
				if (weightSum > 0.0001f && !restPose.empty()) {
					glm::mat4 skin = v.weights.x * restPose[v.joints.x] + v.weights.y * restPose[v.joints.y] +
						v.weights.z * restPose[v.joints.z] + v.weights.w * restPose[v.joints.w];
					pos = glm::vec3(skin * glm::vec4(v.position, 1.0f));
				}
				lo = glm::min(lo, pos);
				hi = glm::max(hi, pos);
			}
			if (lo.x <= hi.x) {
				asset.restBoundsCenter = (lo + hi) * 0.5f;
				asset.restBoundsRadius = glm::length(hi - asset.restBoundsCenter);
			}
		}

	} // namespace

	bool GltfModelManager::LoadFromFile(
		const std::filesystem::path&     path,
		graph::PhysicalResourceRegistry& registry,
		MaterialManager&                 materialManager,
		ModelAsset&                      outAsset
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

		outAsset.vertices = m_model.Vertices();
		outAsset.meshletVertices = m_model.MeshletVertices();
		outAsset.meshletTrianglesPacked = m_model.MeshletTrianglesPacked();
		outAsset.meshlets = m_model.Meshlets();
		outAsset.jointCount = static_cast<std::uint32_t>(m_model.JointCount());

		// Baked once at load (Stage 0's measured 120 Hz floor -- see GltfModel::BakeClips's own
		// comment).
		m_model.BakeClips(120.0f, outAsset.palette, outAsset.clipTable);

		std::vector<glm::mat4> restPose(
			outAsset.palette.begin(),
			outAsset.palette.begin() + static_cast<std::ptrdiff_t>(outAsset.jointCount)
		);
		ComputeRestBounds(outAsset.vertices, restPose, outAsset);

		return true;
	}

} // namespace brassica
