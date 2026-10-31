#include "model/GltfModel.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "fastgltf/core.hpp"
#include "fastgltf/glm_element_traits.hpp"
#include "fastgltf/tools.hpp"
#include "meshoptimizer.h"
#include "spdlog/spdlog.h"
#include "stb_image.h"

namespace brassica {

	namespace {

		constexpr std::size_t kMaxMeshletVerts = 128;
		constexpr std::size_t kMaxMeshletTris = 128;

		// Decodes one triangle-list primitive's accessors and its meshlets, then appends them
		// into the model-wide arrays, rebasing every index/offset so they address the model-wide
		// buffers rather than this primitive's own local-to-itself ones. triangleBytes is a
		// staging area shared across every primitive in the model: micro-indices are only packed
		// 4-per-uint (GltfModel::MeshletTrianglesPacked's layout) once, after every primitive has
        // contributed, so triangleOffset rebasing here must use a running byte count, not a
        // running word count.
		void LoadPrimitive(
			const fastgltf::Asset&    asset,
			const fastgltf::Primitive& prim,
			std::vector<GltfVertex>&   outVertices,
			std::vector<std::uint32_t>& outMeshletVertices,
			std::vector<std::uint8_t>&  triangleBytes,
			std::vector<GltfMeshlet>&   outMeshlets
		) {
			auto posIt = prim.findAttribute("POSITION");
			if (posIt == prim.attributes.end()) {
				spdlog::warn("GltfModel: primitive has no POSITION attribute, skipping");
				return;
			}
			const fastgltf::Accessor& posAccessor = asset.accessors[posIt->accessorIndex];
			std::vector<GltfVertex>   primVertices(posAccessor.count);

			fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor, [&](glm::vec3 v, std::size_t idx) {
				primVertices[idx].position = v;
			});

			// NORMAL/TEXCOORD_0 are technically optional in glTF; a missing NORMAL falls back to a
			// flat up-vector rather than computing real flat normals, since Stage 1's goal is a
			// working render path, not correct shading for the (so far unseen in this project's own
			// test assets) no-normals case -- worth revisiting if an asset without NORMAL actually
			// shows up looking wrong.
			if (auto normIt = prim.findAttribute("NORMAL"); normIt != prim.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::vec3>(
					asset,
					asset.accessors[normIt->accessorIndex],
					[&](glm::vec3 v, std::size_t idx) { primVertices[idx].normal = v; }
				);
			} else {
				for (auto& v : primVertices) {
					v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
				}
			}

			if (auto uvIt = prim.findAttribute("TEXCOORD_0"); uvIt != prim.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::vec2>(
					asset,
					asset.accessors[uvIt->accessorIndex],
					[&](glm::vec2 v, std::size_t idx) { primVertices[idx].uv = v; }
				);
			}

			// JOINTS_0/WEIGHTS_0 stay zero-filled (GltfVertex's own in-class defaults) for any
			// primitive that lacks them -- correct for Stage 1's non-skinned test assets, and
			// already the right fallback for Stage 3/4 skinning too (a zero-weight joint
			// contributes nothing to the skin matrix blend).
			if (auto jointsIt = prim.findAttribute("JOINTS_0"); jointsIt != prim.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::uvec4>(
					asset,
					asset.accessors[jointsIt->accessorIndex],
					[&](glm::uvec4 v, std::size_t idx) { primVertices[idx].joints = v; }
				);
			}
			if (auto weightsIt = prim.findAttribute("WEIGHTS_0"); weightsIt != prim.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::vec4>(
					asset,
					asset.accessors[weightsIt->accessorIndex],
					[&](glm::vec4 v, std::size_t idx) { primVertices[idx].weights = v; }
				);
			}

			if (!prim.indicesAccessor.has_value()) {
				// Shouldn't happen -- Options::GenerateMeshIndices (set by GltfModel::LoadFromFile)
				// makes fastgltf synthesize indices for any primitive that lacks them.
				spdlog::warn("GltfModel: primitive has no indices even after GenerateMeshIndices, skipping");
				return;
			}
			const fastgltf::Accessor&  idxAccessor = asset.accessors[prim.indicesAccessor.value()];
			std::vector<std::uint32_t> primIndices(idxAccessor.count);
			fastgltf::iterateAccessorWithIndex<std::uint32_t>(asset, idxAccessor, [&](std::uint32_t v, std::size_t idx) {
				primIndices[idx] = v;
			});

			if (primIndices.empty()) {
				return;
			}

			std::size_t                     maxMeshlets = meshopt_buildMeshletsBound(
                primIndices.size(),
                kMaxMeshletVerts,
                kMaxMeshletTris
            );
			std::vector<meshopt_Meshlet>    rawMeshlets(maxMeshlets);
			std::vector<unsigned int>       rawMeshletVertices(maxMeshlets * kMaxMeshletVerts);
			std::vector<unsigned char>      rawMeshletTriangles(maxMeshlets * kMaxMeshletTris * 3);

			std::size_t meshletCount = meshopt_buildMeshlets(
				rawMeshlets.data(),
				rawMeshletVertices.data(),
				rawMeshletTriangles.data(),
				primIndices.data(),
				primIndices.size(),
				&primVertices[0].position.x,
				primVertices.size(),
				sizeof(GltfVertex),
				kMaxMeshletVerts,
				kMaxMeshletTris,
				0.0f
			);
			rawMeshlets.resize(meshletCount);
			if (meshletCount > 0) {
				const meshopt_Meshlet& last = rawMeshlets.back();
				rawMeshletVertices.resize(last.vertex_offset + last.vertex_count);
				rawMeshletTriangles.resize(last.triangle_offset + last.triangle_count * 3);
			} else {
				rawMeshletVertices.clear();
				rawMeshletTriangles.clear();
			}

			// Recommended by meshoptimizer's own meshopt_buildMeshlets doc comment: reorders each
			// meshlet's own vertex/triangle ranges in place for better locality, same arrays /
			// offsets, no rebasing implications for the code below.
			for (const meshopt_Meshlet& m : rawMeshlets) {
				meshopt_optimizeMeshlet(
					&rawMeshletVertices[m.vertex_offset],
					&rawMeshletTriangles[m.triangle_offset],
					m.triangle_count,
					m.vertex_count
				);
			}

			const auto vertexBase = static_cast<std::uint32_t>(outVertices.size());
			const auto meshletVertexBase = static_cast<std::uint32_t>(outMeshletVertices.size());
			const auto triangleByteBase = static_cast<std::uint32_t>(triangleBytes.size());

			outVertices.insert(outVertices.end(), primVertices.begin(), primVertices.end());

			outMeshletVertices.reserve(outMeshletVertices.size() + rawMeshletVertices.size());
			for (unsigned int localVertexIndex : rawMeshletVertices) {
				outMeshletVertices.push_back(vertexBase + localVertexIndex);
			}

			triangleBytes.insert(triangleBytes.end(), rawMeshletTriangles.begin(), rawMeshletTriangles.end());

			outMeshlets.reserve(outMeshlets.size() + rawMeshlets.size());
			for (const meshopt_Meshlet& m : rawMeshlets) {
				meshopt_Bounds bounds = meshopt_computeMeshletBounds(
					&rawMeshletVertices[m.vertex_offset],
					&rawMeshletTriangles[m.triangle_offset],
					m.triangle_count,
					&primVertices[0].position.x,
					primVertices.size(),
					sizeof(GltfVertex)
				);

				GltfMeshlet gpuMeshlet{};
				gpuMeshlet.vertexOffset = meshletVertexBase + m.vertex_offset;
				// meshopt's own triangle_offset addresses its flat uint8 micro-index buffer
				// directly (a byte/corner offset); GltfMeshlet stores triangleOffset in *triangle*
				// units instead (see shaders/gltf_mesh.glsl's GltfMeshletTriangleRef comment:
				// byteIndex = meshlet.triangleOffset*3 + ...), so divide by 3 here. Always exact:
				// meshopt lays out each meshlet's byte range as count*3, a whole multiple of 3.
				gpuMeshlet.triangleOffset = (triangleByteBase + m.triangle_offset) / 3u;
				gpuMeshlet.vertexCount = m.vertex_count;
				gpuMeshlet.triangleCount = m.triangle_count;
				gpuMeshlet.boundsCenter = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
				gpuMeshlet.boundsRadius = bounds.radius;
				gpuMeshlet.coneAxis = glm::vec3(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]);
				gpuMeshlet.coneCutoff = bounds.cone_cutoff;
				// Local (this model's own Materials() index) for now -- GltfModelManager rewrites
				// this to the real global MaterialBuffer index via RemapMaterialIndices once every
				// GltfMaterial has been registered into the shared MaterialManager. 0xFFFFFFFFu
				// ("no material") remaps to MaterialManager's own always-present default material
				// at global index 0 -- see RemapMaterialIndices.
				gpuMeshlet.materialIndex =
					prim.materialIndex.has_value() ? static_cast<std::uint32_t>(prim.materialIndex.value()) : 0xFFFFFFFFu;
				outMeshlets.push_back(gpuMeshlet);
			}
		}

		// Raw encoded (PNG/JPEG) bytes for one glTF image. GLB-embedded images are
		// sources::BufferView (resolved through the same DefaultBufferDataAdapter the accessor
		// decoding above already relies on for buffer data); an external .gltf+.png/.jpg pair
		// loaded via Options::LoadExternalImages is sources::Array. sources::URI (an unresolved
		// external file path) isn't handled -- LoadExternalImages means fastgltf should never
		// hand one back for an image fastgltf itself already decided to load.
		std::vector<std::byte> ExtractEncodedImageBytes(const fastgltf::Asset& asset, const fastgltf::Image& image) {
			if (const auto* bufferView = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
				fastgltf::DefaultBufferDataAdapter adapter;
				auto                                bytes = adapter(asset, bufferView->bufferViewIndex);
				return std::vector<std::byte>(bytes.begin(), bytes.end());
			}
			if (const auto* array = std::get_if<fastgltf::sources::Array>(&image.data)) {
				return std::vector<std::byte>(array->bytes.begin(), array->bytes.end());
			}
			spdlog::warn("GltfModel: image '{}' has an unsupported data source, skipping", image.name);
			return {};
		}

		// Decodes (if not already cached) the glTF image referenced by `textureIndex` and returns
		// its index into outImages, or -1 if textureIndex is absent/invalid/fails to decode.
		// imageCache dedups by glTF image index (not by role) -- an image referenced in two
		// different material roles is decoded once; isSrgb is only honored on the first
		// reference (see GltfImage's own comment).
		std::int32_t ResolveImage(
			const fastgltf::Asset&                asset,
			const fastgltf::Optional<std::size_t>& textureIndex,
			bool                                   isSrgb,
			std::unordered_map<std::size_t, std::int32_t>& imageCache,
			std::vector<GltfImage>&                         outImages
		) {
			if (!textureIndex.has_value() || textureIndex.value() >= asset.textures.size()) {
				return -1;
			}
			const fastgltf::Texture& texture = asset.textures[textureIndex.value()];
			if (!texture.imageIndex.has_value() || texture.imageIndex.value() >= asset.images.size()) {
				return -1;
			}
			std::size_t imageIndex = texture.imageIndex.value();

			if (auto it = imageCache.find(imageIndex); it != imageCache.end()) {
				return it->second;
			}

			std::vector<std::byte> encoded = ExtractEncodedImageBytes(asset, asset.images[imageIndex]);
			if (encoded.empty()) {
				imageCache[imageIndex] = -1;
				return -1;
			}

			int width = 0, height = 0, channels = 0;
			stbi_uc* pixels = stbi_load_from_memory(
				reinterpret_cast<const stbi_uc*>(encoded.data()),
				static_cast<int>(encoded.size()),
				&width,
				&height,
				&channels,
				4 // force RGBA8 regardless of the source file's real channel count
			);
			if (!pixels) {
				spdlog::warn("GltfModel: failed to decode image index {}: {}", imageIndex, stbi_failure_reason());
				imageCache[imageIndex] = -1;
				return -1;
			}

			GltfImage decoded;
			decoded.width = static_cast<std::uint32_t>(width);
			decoded.height = static_cast<std::uint32_t>(height);
			decoded.isSrgb = isSrgb;
			decoded.pixelsRGBA8.assign(pixels, pixels + static_cast<std::size_t>(width) * height * 4);
			stbi_image_free(pixels);

			auto newIndex = static_cast<std::int32_t>(outImages.size());
			outImages.push_back(std::move(decoded));
			imageCache[imageIndex] = newIndex;
			return newIndex;
		}

		void LoadMaterials(
			const fastgltf::Asset&   asset,
			std::vector<GltfImage>&  outImages,
			std::vector<GltfMaterial>& outMaterials
		) {
			std::unordered_map<std::size_t, std::int32_t> imageCache;

			for (const fastgltf::Material& material : asset.materials) {
				GltfMaterial gm;
				const auto&  pbr = material.pbrData;
				gm.baseColorFactor =
					glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2], pbr.baseColorFactor[3]);
				gm.metallicFactor = pbr.metallicFactor;
				gm.roughnessFactor = pbr.roughnessFactor;
				gm.emissiveFactor =
					glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]);

				gm.baseColorImageIndex = ResolveImage(
					asset,
					pbr.baseColorTexture ? fastgltf::Optional<std::size_t>(pbr.baseColorTexture->textureIndex)
										  : fastgltf::Optional<std::size_t>(),
					/*isSrgb=*/true,
					imageCache,
					outImages
				);

				fastgltf::Optional<std::size_t> mrTextureIndex = pbr.metallicRoughnessTexture
					? fastgltf::Optional<std::size_t>(pbr.metallicRoughnessTexture->textureIndex)
					: fastgltf::Optional<std::size_t>();
				gm.metallicRoughnessImageIndex = ResolveImage(asset, mrTextureIndex, /*isSrgb=*/false, imageCache, outImages);

				// Occlusion only fits MaterialData's single packed slot when it's literally the same
				// glTF texture as metallicRoughness -- see GltfMaterial's own comment. Different
				// textures: log it and drop occlusion rather than silently ignoring the mismatch.
				if (material.occlusionTexture) {
					if (mrTextureIndex.has_value() && material.occlusionTexture->textureIndex == mrTextureIndex.value()) {
						gm.occlusionImageIndex = gm.metallicRoughnessImageIndex;
					} else {
						spdlog::warn(
							"GltfModel: material '{}' has a separate occlusion texture from its "
							"metallicRoughness texture -- MaterialData has one combined slot for both, "
							"so occlusion is dropped for this material",
							material.name
						);
					}
				}

				gm.emissiveImageIndex = ResolveImage(
					asset,
					material.emissiveTexture ? fastgltf::Optional<std::size_t>(material.emissiveTexture->textureIndex)
											   : fastgltf::Optional<std::size_t>(),
					/*isSrgb=*/true,
					imageCache,
					outImages
				);
				gm.normalImageIndex = ResolveImage(
					asset,
					material.normalTexture ? fastgltf::Optional<std::size_t>(material.normalTexture->textureIndex)
											 : fastgltf::Optional<std::size_t>(),
					/*isSrgb=*/false,
					imageCache,
					outImages
				);

				outMaterials.push_back(gm);
			}
		}

		// Options::DecomposeNodeMatrices (set by LoadFromFile) guarantees every node.transform
		// variant holds a TRS, never a raw matrix -- there is deliberately no matrix branch here.
		void LoadSkeleton(
			const fastgltf::Asset&          asset,
			std::vector<GltfSkeletonNode>&  outNodes,
			std::vector<std::uint32_t>&     outRoots,
			std::vector<std::uint32_t>&     outJointIndices,
			std::vector<glm::mat4>&         outInverseBindMatrices
		) {
			outNodes.resize(asset.nodes.size());
			for (std::size_t i = 0; i < asset.nodes.size(); ++i) {
				const fastgltf::Node& node = asset.nodes[i];
				GltfSkeletonNode&     sn = outNodes[i];
				if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
					sn.restTranslation = glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2]);
					sn.restRotation =
						glm::quat(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]);
					sn.restScale = glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]);
				}
				sn.children.reserve(node.children.size());
				for (std::size_t childIdx : node.children) {
					sn.children.push_back(static_cast<std::uint32_t>(childIdx));
				}
			}

			std::size_t sceneIndex = asset.defaultScene.has_value() ? asset.defaultScene.value() : 0;
			if (sceneIndex < asset.scenes.size()) {
				outRoots.reserve(asset.scenes[sceneIndex].nodeIndices.size());
				for (std::size_t rootIdx : asset.scenes[sceneIndex].nodeIndices) {
					outRoots.push_back(static_cast<std::uint32_t>(rootIdx));
				}
			}

			// Stage 3 scope: at most one skin, matching every real test asset (cow/rabbit/
			// base_character.glb each have exactly one). A multi-skin asset's later skins are
			// simply ignored, not an error -- logged for visibility.
			if (asset.skins.empty()) {
				return;
			}
			if (asset.skins.size() > 1) {
				spdlog::warn("GltfModel: asset has {} skins, only the first is used", asset.skins.size());
			}
			const fastgltf::Skin& skin = asset.skins[0];
			outJointIndices.reserve(skin.joints.size());
			for (std::size_t jointNodeIdx : skin.joints) {
				outJointIndices.push_back(static_cast<std::uint32_t>(jointNodeIdx));
			}

			outInverseBindMatrices.assign(outJointIndices.size(), glm::mat4(1.0f));
			if (skin.inverseBindMatrices.has_value()) {
				const fastgltf::Accessor& ibmAccessor = asset.accessors[skin.inverseBindMatrices.value()];
				fastgltf::iterateAccessorWithIndex<glm::mat4>(asset, ibmAccessor, [&](glm::mat4 m, std::size_t idx) {
					if (idx < outInverseBindMatrices.size()) {
						outInverseBindMatrices[idx] = m;
					}
				});
			}
		}

		void LoadAnimations(const fastgltf::Asset& asset, std::vector<GltfAnimation>& outAnimations) {
			outAnimations.reserve(asset.animations.size());
			for (const fastgltf::Animation& anim : asset.animations) {
				GltfAnimation ga;
				ga.name = std::string(anim.name);

				for (const fastgltf::AnimationChannel& channel : anim.channels) {
					if (!channel.nodeIndex.has_value()) {
						continue; // e.g. a KHR_animation_pointer channel -- not supported
					}
					if (channel.path == fastgltf::AnimationPath::Weights) {
						continue; // morph-target animation is out of scope
					}

					const fastgltf::AnimationSampler& sampler = anim.samplers[channel.samplerIndex];
					if (sampler.interpolation == fastgltf::AnimationInterpolation::CubicSpline) {
						// Packs 3x the output values per keyframe (in-tangent, value, out-tangent)
						// -- decoding it as if it were LINEAR/STEP would read tangent data as pose
						// data. None of this project's test assets use it, so there's nothing to
						// verify a real implementation against yet; dropped, not misread.
						spdlog::warn(
							"GltfModel: animation '{}' channel on node {} uses CUBICSPLINE "
							"interpolation, dropping (unsupported)",
							ga.name,
							channel.nodeIndex.value()
						);
						continue;
					}

					GltfAnimationChannel gc;
					gc.nodeIndex = static_cast<std::uint32_t>(channel.nodeIndex.value());
					gc.isStep = sampler.interpolation == fastgltf::AnimationInterpolation::Step;
					switch (channel.path) {
					case fastgltf::AnimationPath::Translation:
						gc.path = GltfAnimationChannel::Path::Translation;
						break;
					case fastgltf::AnimationPath::Rotation:
						gc.path = GltfAnimationChannel::Path::Rotation;
						break;
					case fastgltf::AnimationPath::Scale:
						gc.path = GltfAnimationChannel::Path::Scale;
						break;
					default:
						continue; // shouldn't happen -- Weights is filtered above, no other path exists
					}

					const fastgltf::Accessor& inputAccessor = asset.accessors[sampler.inputAccessor];
					gc.times.resize(inputAccessor.count);
					fastgltf::iterateAccessorWithIndex<float>(asset, inputAccessor, [&](float t, std::size_t idx) {
						gc.times[idx] = t;
					});

					const fastgltf::Accessor& outputAccessor = asset.accessors[sampler.outputAccessor];
					gc.values.resize(outputAccessor.count);
					if (gc.path == GltfAnimationChannel::Path::Rotation) {
						fastgltf::iterateAccessorWithIndex<glm::vec4>(
							asset,
							outputAccessor,
							[&](glm::vec4 v, std::size_t idx) { gc.values[idx] = v; }
						);
					} else {
						fastgltf::iterateAccessorWithIndex<glm::vec3>(
							asset,
							outputAccessor,
							[&](glm::vec3 v, std::size_t idx) { gc.values[idx] = glm::vec4(v, 0.0f); }
						);
					}

					if (!gc.times.empty()) {
						ga.duration = std::max(ga.duration, gc.times.back());
					}
					ga.channels.push_back(std::move(gc));
				}

				outAnimations.push_back(std::move(ga));
			}
		}

		// Returns the bracketed (or stepped) value at `time`, NLERP-normalized with a shortest-
		// path correction for rotation channels specifically -- a raw component-wise lerp between
		// two quaternions isn't itself a unit quaternion, and lerping the "long way around" between
		// two keyframes that are actually close in rotation space (one negated relative to the
		// other, the same rotation either way) would visibly take the wrong path. Translation/
		// scale channels skip both: they're genuinely just vec3s riding in a vec4's .xyz.
		glm::vec4 SampleChannel(const GltfAnimationChannel& channel, float time, bool isRotation) {
			if (channel.times.empty()) {
				return glm::vec4(0.0f);
			}
			if (time <= channel.times.front()) {
				return channel.values.front();
			}
			if (time >= channel.times.back()) {
				return channel.values.back();
			}
			// Linear scan: real channel keyframe counts here are small (tens, confirmed against
			// this project's own test assets), so a binary search isn't worth the complexity.
			for (std::size_t i = 0; i + 1 < channel.times.size(); ++i) {
				if (time < channel.times[i] || time > channel.times[i + 1]) {
					continue;
				}
				if (channel.isStep) {
					return channel.values[i];
				}
				glm::vec4 a = channel.values[i];
				glm::vec4 b = channel.values[i + 1];
				if (isRotation && glm::dot(a, b) < 0.0f) {
					b = -b;
				}
				float     span = channel.times[i + 1] - channel.times[i];
				float     t = span > 0.0f ? (time - channel.times[i]) / span : 0.0f;
				glm::vec4 result = glm::mix(a, b, t);
				if (isRotation) {
					float len = glm::length(result);
					if (len > 0.0001f) {
						result /= len;
					}
				}
				return result;
			}
			return channel.values.back();
		}

	} // namespace

	bool GltfModel::LoadFromFile(const std::filesystem::path& path) {
		m_vertices.clear();
		m_meshletVertices.clear();
		m_meshletTrianglesPacked.clear();
		m_meshlets.clear();
		m_images.clear();
		m_materials.clear();
		m_skeletonNodes.clear();
		m_rootNodeIndices.clear();
		m_jointNodeIndices.clear();
		m_inverseBindMatrices.clear();
		m_animations.clear();

		auto dataResult = fastgltf::GltfDataBuffer::FromPath(path);
		if (dataResult.error() != fastgltf::Error::None) {
			spdlog::error(
				"GltfModel: failed to open '{}': {}",
				path.string(),
				fastgltf::getErrorMessage(dataResult.error())
			);
			return false;
		}

		fastgltf::Parser parser;
		// GenerateMeshIndices: this loader only ever walks indexed primitives (see LoadPrimitive's
		// early-return otherwise) -- synthesizing indices for any non-indexed primitive means that
		// early-return is a real "shouldn't happen" guard, not a silent mesh-drop for ordinary
		// assets. LoadExternalBuffers: test assets are all single-file .glb today, but a .gltf +
		// .bin pair should load correctly too without this being revisited. LoadExternalImages
		// (Stage 2): without it, an external (non-GLB-embedded) image's DataSource stays an
		// unresolved sources::URI, which ExtractEncodedImageBytes below doesn't handle.
		// DecomposeNodeMatrices (Stage 3): guarantees every node.transform is a TRS, never a raw
		// matrix, so LoadSkeleton never needs a matrix-decompose branch -- fastgltf's own doc
		// comment recommends this specifically for working with animated nodes.
		constexpr auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
			fastgltf::Options::DecomposeNodeMatrices | fastgltf::Options::GenerateMeshIndices;
		auto           assetResult = parser.loadGltf(dataResult.get(), path.parent_path(), options);
		if (assetResult.error() != fastgltf::Error::None) {
			spdlog::error(
				"GltfModel: failed to parse '{}': {}",
				path.string(),
				fastgltf::getErrorMessage(assetResult.error())
			);
			return false;
		}
		fastgltf::Asset& asset = assetResult.get();

		LoadMaterials(asset, m_images, m_materials);
		LoadSkeleton(asset, m_skeletonNodes, m_rootNodeIndices, m_jointNodeIndices, m_inverseBindMatrices);
		LoadAnimations(asset, m_animations);

		std::vector<std::uint8_t> triangleBytes;

		for (const fastgltf::Mesh& mesh : asset.meshes) {
			for (const fastgltf::Primitive& prim : mesh.primitives) {
				if (prim.type != fastgltf::PrimitiveType::Triangles) {
					continue; // Stage 1 scope: triangle lists only (points/lines/strips/fans unsupported)
				}
				LoadPrimitive(asset, prim, m_vertices, m_meshletVertices, triangleBytes, m_meshlets);
			}
		}

		// Pack 4 micro-index bytes per uint32, little-endian -- matches shaders/gltf_mesh.glsl's
		// GltfMeshletTriangleRef unpacking formula exactly (word = indices[byteIndex >> 2]; value =
		// (word >> ((byteIndex & 3) * 8)) & 0xFFu).
		std::size_t wordCount = (triangleBytes.size() + 3) / 4;
		m_meshletTrianglesPacked.assign(wordCount, 0u);
		for (std::size_t i = 0; i < triangleBytes.size(); ++i) {
			m_meshletTrianglesPacked[i >> 2] |= static_cast<std::uint32_t>(triangleBytes[i]) << ((i & 3u) * 8u);
		}

		if (m_meshlets.empty()) {
			spdlog::error("GltfModel: '{}' produced no renderable meshlets", path.string());
			return false;
		}

		spdlog::info(
			"GltfModel: loaded '{}' -- {} vertices, {} meshlets, {} materials, {} images, {} joints, "
			"{} animations",
			path.string(),
			m_vertices.size(),
			m_meshlets.size(),
			m_materials.size(),
			m_images.size(),
			m_jointNodeIndices.size(),
			m_animations.size()
		);
		return true;
	}

	void GltfModel::RemapMaterialIndices(std::span<const std::uint32_t> localToGlobal) {
		for (GltfMeshlet& meshlet : m_meshlets) {
			meshlet.materialIndex =
				meshlet.materialIndex < localToGlobal.size() ? localToGlobal[meshlet.materialIndex] : 0u;
		}
	}

	void GltfModel::EvaluatePose(
		std::uint32_t           animationIndex,
		float                   time,
		std::vector<glm::mat4>& outJointMatrices
	) const {
		std::vector<glm::mat4> localTransforms(m_skeletonNodes.size());
		for (std::size_t i = 0; i < m_skeletonNodes.size(); ++i) {
			const GltfSkeletonNode& node = m_skeletonNodes[i];
			localTransforms[i] = glm::translate(glm::mat4(1.0f), node.restTranslation) *
				glm::mat4_cast(node.restRotation) * glm::scale(glm::mat4(1.0f), node.restScale);
		}

		if (animationIndex < m_animations.size()) {
			const GltfAnimation& anim = m_animations[animationIndex];

			// Per-node overrides: a channel only replaces the ONE property (translation/rotation/
			// scale) it targets -- the other two keep their rest-pose value, same independence the
			// glTF spec itself gives each path.
			std::vector<glm::vec3> translations(m_skeletonNodes.size());
			std::vector<glm::quat> rotations(m_skeletonNodes.size());
			std::vector<glm::vec3> scales(m_skeletonNodes.size());
			std::vector<bool>      hasTranslation(m_skeletonNodes.size(), false);
			std::vector<bool>      hasRotation(m_skeletonNodes.size(), false);
			std::vector<bool>      hasScale(m_skeletonNodes.size(), false);

			for (const GltfAnimationChannel& channel : anim.channels) {
				if (channel.nodeIndex >= m_skeletonNodes.size()) {
					continue;
				}
				bool      isRotation = channel.path == GltfAnimationChannel::Path::Rotation;
				glm::vec4 sampled = SampleChannel(channel, time, isRotation);
				switch (channel.path) {
				case GltfAnimationChannel::Path::Translation:
					translations[channel.nodeIndex] = glm::vec3(sampled);
					hasTranslation[channel.nodeIndex] = true;
					break;
				case GltfAnimationChannel::Path::Rotation:
					rotations[channel.nodeIndex] = glm::quat(sampled.w, sampled.x, sampled.y, sampled.z);
					hasRotation[channel.nodeIndex] = true;
					break;
				case GltfAnimationChannel::Path::Scale:
					scales[channel.nodeIndex] = glm::vec3(sampled);
					hasScale[channel.nodeIndex] = true;
					break;
				}
			}

			for (std::size_t i = 0; i < m_skeletonNodes.size(); ++i) {
				if (!hasTranslation[i] && !hasRotation[i] && !hasScale[i]) {
					continue; // unanimated node keeps its rest-pose local transform
				}
				glm::vec3 t = hasTranslation[i] ? translations[i] : m_skeletonNodes[i].restTranslation;
				glm::quat r = hasRotation[i] ? rotations[i] : m_skeletonNodes[i].restRotation;
				glm::vec3 s = hasScale[i] ? scales[i] : m_skeletonNodes[i].restScale;
				localTransforms[i] =
					glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
			}
		}

		std::vector<glm::mat4> globalTransforms(m_skeletonNodes.size(), glm::mat4(1.0f));
		std::vector<bool>      computed(m_skeletonNodes.size(), false);

		// Iterative stack-based walk, not recursive -- avoids ever worrying about stack depth,
		// even though real joint hierarchies here are shallow (tens of nodes).
		std::vector<std::pair<std::uint32_t, glm::mat4>> stack;
		for (std::uint32_t root : m_rootNodeIndices) {
			stack.push_back({root, glm::mat4(1.0f)});
		}
		while (!stack.empty()) {
			auto [nodeIndex, parentGlobal] = stack.back();
			stack.pop_back();
			if (nodeIndex >= m_skeletonNodes.size() || computed[nodeIndex]) {
				continue;
			}
			glm::mat4 global = parentGlobal * localTransforms[nodeIndex];
			globalTransforms[nodeIndex] = global;
			computed[nodeIndex] = true;
			for (std::uint32_t child : m_skeletonNodes[nodeIndex].children) {
				stack.push_back({child, global});
			}
		}

		outJointMatrices.resize(m_jointNodeIndices.size());
		for (std::size_t i = 0; i < m_jointNodeIndices.size(); ++i) {
			std::uint32_t jointNode = m_jointNodeIndices[i];
			glm::mat4     global = jointNode < globalTransforms.size() ? globalTransforms[jointNode] : glm::mat4(1.0f);
			outJointMatrices[i] = global * m_inverseBindMatrices[i];
		}
	}

} // namespace brassica
