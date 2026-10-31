#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "model/GltfTypes.hpp"

namespace brassica {

	// One decoded (not yet GPU-uploaded) RGBA8 image -- stb_image decode happens here, inside
	// GltfModel, specifically so this class stays Vulkan-free and headlessly testable (same
	// reasoning as meshopt_buildMeshlets already running here in Stage 1): the fastgltf::Asset
	// this is extracted from only lives for the duration of LoadFromFile, so decoding has to
	// happen now, not deferred to whatever later owns a real GPU device (GltfTextureManager).
	struct GltfImage {
		std::vector<std::uint8_t> pixelsRGBA8;
		std::uint32_t             width{0};
		std::uint32_t             height{0};
		// true for a color-role texture (baseColor/emissive, glTF's own sRGB-encoded roles) vs.
		// false for a data-role one (normal/metallicRoughness/occlusion, always linear) -- set
		// from whichever material role first referenced this image. A real per-usage distinction
		// (not a guess): glTF defines color vs. data textures by role, not by image file.
		bool isSrgb{false};
	};

	// CPU-side mirror of the glTF material fields Stage 2 actually wires up. Image indices (not
	// glTF texture indices) point into GltfModel::Images() -- the loader already resolves the
	// textureIndex -> imageIndex indirection and the sampler choice (every glTF texture uses this
	// engine's one BRASSICA_SAMPLER_LINEAR_REPEAT_MIP bindless sampler, matching glTF's own
	// repeat-wrap default; per-texture sampler settings like a non-repeat wrap mode are not
	// respected -- a real simplification, not a silent one).
	//
	// normalImageIndex is parsed and stored but never sampled (GltfVertex carries no tangent
	// attribute yet -- see material_system.glsl's own comment on evaluateGltfBaseColor and
	// friends). occlusionImageIndex only ever gets filled in when it's the *same* glTF texture as
	// metallicRoughnessImageIndex -- MaterialData has exactly one combined slot for both (see
	// MaterialData.hpp), so an asset using two different images for occlusion vs.
	// metallic/roughness loses the occlusion texture (logged, not silently dropped).
	struct GltfMaterial {
		glm::vec4    baseColorFactor{1.0f};
		glm::vec3    emissiveFactor{0.0f};
		float        metallicFactor{1.0f};
		float        roughnessFactor{1.0f};
		std::int32_t baseColorImageIndex{-1};
		std::int32_t metallicRoughnessImageIndex{-1};
		std::int32_t occlusionImageIndex{-1};
		std::int32_t emissiveImageIndex{-1};
		std::int32_t normalImageIndex{-1};
	};

	// One glTF node's rest-pose local transform plus its children, flattened to the same indices
	// fastgltf's own asset.nodes uses (so a channel/skin's nodeIndex is usable directly, no
	// remapping). Scope matches the rest of this pipeline: Options::DecomposeNodeMatrices (set by
	// LoadFromFile) guarantees every node.transform is TRS, not a raw matrix, so there's no
	// matrix-decompose branch to carry here at all.
	struct GltfSkeletonNode {
		glm::vec3                 restTranslation{0.0f};
		glm::quat                 restRotation{1.0f, 0.0f, 0.0f, 0.0f}; // identity (w, x, y, z)
		glm::vec3                 restScale{1.0f};
		std::vector<std::uint32_t> children;
	};

	// One animated property of one node, pre-split by path so sampling never has to branch on it
	// at evaluation time. Rotation values are vec4(x, y, z, w) (glTF's own quaternion component
	// order); translation/scale values use only .xyz. CUBICSPLINE channels are dropped entirely
	// at load time (logged, not silently misread) -- that interpolation mode packs 3x the output
	// values per keyframe (in-tangent, value, out-tangent), and none of this project's test assets
	// use it, so there's nothing to verify a real implementation against yet.
	struct GltfAnimationChannel {
		enum class Path : std::uint8_t { Translation, Rotation, Scale };

		std::uint32_t          nodeIndex{0};
		Path                   path{Path::Translation};
		bool                   isStep{false}; // true: STEP interpolation, false: LINEAR
		std::vector<float>     times;
		std::vector<glm::vec4> values;
	};

	struct GltfAnimation {
		std::string                       name;
		float                              duration{0.0f}; // max keyframe time across all channels
		std::vector<GltfAnimationChannel> channels;
	};

	// Stage 1 of the GLTF/GLB pipeline (static mesh loading + GPU-driven rendering): parses a
	// single glTF/GLB asset via fastgltf, decodes every triangle-list primitive's accessors into
	// a flat GltfVertex array, and builds EXT_mesh_shader-ready meshlets via
	// meshopt_buildMeshlets. Stage 2 (materials + textures) added image decoding and material
	// parsing, still CPU-only. Stage 3 (skeleton + animation) added the node hierarchy, one skin's
	// joints/inverse-bind matrices, and every animation clip -- still CPU-only, no ozz (dropped
	// per the agreed plan); EvaluatePose() is this stage's own minimal keyframe sampler (LERP
	// translation/scale, NLERP rotation, no blending between clips). Deliberately narrow scope:
	// one asset, at most one skin, no multi-instance support.
	class GltfModel {
	public:
		// Clears any previously loaded data first, so a failed reload can't leave a stale mix of
		// old and new primitives.
		bool LoadFromFile(const std::filesystem::path& path);

		[[nodiscard]] const std::vector<GltfVertex>& Vertices() const { return m_vertices; }

		// Indices into Vertices(), one entry per meshlet-local vertex slot -- see
		// shaders/gltf_mesh.glsl's GltfMeshletVertexRef comment.
		[[nodiscard]] const std::vector<std::uint32_t>& MeshletVertices() const { return m_meshletVertices; }

		// 4 micro-indices packed per uint (little-endian byte order) -- see
		// shaders/gltf_mesh.glsl's GltfMeshletTriangleRef comment for the unpacking formula.
		[[nodiscard]] const std::vector<std::uint32_t>& MeshletTrianglesPacked() const {
			return m_meshletTrianglesPacked;
		}

		[[nodiscard]] const std::vector<GltfMeshlet>& Meshlets() const { return m_meshlets; }

		[[nodiscard]] const std::vector<GltfImage>& Images() const { return m_images; }

		[[nodiscard]] const std::vector<GltfMaterial>& Materials() const { return m_materials; }

		[[nodiscard]] bool IsLoaded() const { return !m_meshlets.empty(); }

		// Rewrites every meshlet's materialIndex from a local (this model's own Materials()
		// index, or 0xFFFFFFFFu for "no material") to whatever GltfModelManager resolved it to
		// after registering each GltfMaterial into the shared MaterialManager -- see
		// GltfModelManager::LoadFromFile. localToGlobal[i] is the global MaterialBuffer index for
		// local material i; indices out of range (including the sentinel) map to 0, the global
		// MaterialManager's own always-present default material.
		void RemapMaterialIndices(std::span<const std::uint32_t> localToGlobal);

		[[nodiscard]] const std::vector<GltfAnimation>& Animations() const { return m_animations; }

		// Number of joints this model's skin has -- 0 if the asset has no skin at all, in which
		// case EvaluatePose()/GPU skinning have nothing to do (every vertex's zero-filled
		// JOINTS_0/WEIGHTS_0 already makes skinning a no-op on the GPU side regardless, see
		// gltf_render.mesh's own comment, but this lets a caller skip the CPU work too).
		[[nodiscard]] std::size_t JointCount() const { return m_jointNodeIndices.size(); }

		// Computes this animation's pose at `time` (seconds, wrapped to [0, duration) by the
		// caller -- see GltfModelManager::UpdatePose) and writes one mat4 per joint, in skin-joint
		// order, into outJointMatrices (resized as needed). animationIndex out of range (including
		// when the asset has no animations at all) falls back to the skeleton's rest pose --
		// never an error, since "no animation selected yet" is a normal state, not a failure.
		void EvaluatePose(std::uint32_t animationIndex, float time, std::vector<glm::mat4>& outJointMatrices) const;

	private:
		std::vector<GltfVertex>    m_vertices;
		std::vector<std::uint32_t> m_meshletVertices;
		std::vector<std::uint32_t> m_meshletTrianglesPacked;
		std::vector<GltfMeshlet>   m_meshlets;
		std::vector<GltfImage>     m_images;
		std::vector<GltfMaterial>  m_materials;

		std::vector<GltfSkeletonNode> m_skeletonNodes; // same indices as the source asset.nodes
		std::vector<std::uint32_t>    m_rootNodeIndices; // default scene's own root node list
		std::vector<std::uint32_t>    m_jointNodeIndices; // skin.joints -- node index per joint slot
		std::vector<glm::mat4>        m_inverseBindMatrices; // same order as m_jointNodeIndices
		std::vector<GltfAnimation>    m_animations;
	};

} // namespace brassica
