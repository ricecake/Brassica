#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"
#include "terrain/TerrainManager.hpp"

namespace brassica {

	struct TerrainGenPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		std::uint32_t clipmapStorageIdx{0};
		std::uint32_t minMaxStorageIdx{0};
		std::uint32_t biomeStorageIdx{0};
		bool          forceRegeneration = true;
	};

	// S4: no kPhase -- the old comment here ("the graph validator rejects a later-phase producer
	// satisfying an earlier-phase consumer's input") described a hard phase-partition check that
	// no longer exists (removed at S3, see Graph.hpp's Compile() comment). This node is the sole
	// Create<> of TerrainClipmapTexture/TerrainMinMaxTexture/TerrainBiomeTexture, so
	// Create-before-accessor already orders it before every reader -- including
	// ParticleSystemNode's Prepare-phase nodes -- as a real resource edge, regardless of what
	// phase (if any) either side is on.
	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainGenPushConstants          push{};
		glm::vec3                        cameraPos{0.0f};
		bool                             hasUpdate{true};
		bool                             forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			physicalRegistry = services.physicalRegistry;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_gen.comp") ||
			    !aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp")) {
				spdlog::critical("TerrainGenNode shader compilation failed.");
				throw std::runtime_error("TerrainGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&aabbShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
			forceRegeneration = p.forceRegeneration;

			// Publishes this frame's numLODs onto all three keys' catalog instances -- the one
			// piece of runtime config their Desc() needs beyond FrameContext (ResourceKeys.hpp).
			// Idempotent and cheap to call every frame (Get<K> only default-constructs once).
			if (physicalRegistry) {
				auto& catalog = physicalRegistry->Catalog();
				catalog.Get<TerrainClipmapTexture>().numLODs = p.terrainGridParams.x;
				catalog.Get<TerrainMinMaxTexture>().numLODs = p.terrainGridParams.x;
				catalog.Get<TerrainBiomeTexture>().numLODs = p.terrainGridParams.x;
			}
		}

		// These three are genuinely Create<>, not Modify<>, despite this node running every
		// frame: they're pass-owned PERSISTENT resources (Lifetime, ResourceKey.hpp -- not yet
		// read by Provision, but the intent), provisioned the normal way through ProvisionTexture
		// like any other persistent resource, so "recreating" them each frame is really just
		// re-asserting the same desc, not destroying state (ProvisionTexture's desc-match early
		// return leaves the real resource untouched). TerrainTLAS keeps its own explicit .Desc():
		// it's an AccelerationStructure, which Provision() special-cases by switching on this
		// exact realization's own desc.kind before any catalog lookup runs -- see
		// SetupBuilder.hpp's AccessHandle::Desc comment.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainClipmapTexture>();
			r.Create<TerrainMinMaxTexture>();
			r.Create<TerrainBiomeTexture>();
			r.Modify<TerrainTLAS>().Desc(graph::AccelerationStructureDesc());
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout))
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet))
			};

			if (hasUpdate || forceRegeneration) {
				push.clipmapStorageIdx = ctx.StorageIndex<TerrainClipmapTexture>();
				push.minMaxStorageIdx = ctx.StorageIndex<TerrainMinMaxTexture>();
				push.biomeStorageIdx = ctx.StorageIndex<TerrainBiomeTexture>();
				push.forceRegeneration = forceRegeneration;

				std::array<vk::PushConstantRange, 1> pushConstantRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainGenPushConstants)}
				};

				render::ComputePipelineRequest request{
					.shader = &genShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = pushConstantRanges,
				};
				render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (resolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
				}

				if (boundSets[0] && boundSets[1]) {
					vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
				}

				vkCmd.pushConstants(
					resolved.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(TerrainGenPushConstants),
					&push
				);

				uint32_t groupX = (push.gridParams.w + 15) / 16;
				uint32_t groupY = (push.gridParams.w + 15) / 16;
				uint32_t groupZ = push.gridParams.x;
				vkCmd.dispatch(groupX, groupY, groupZ);
			}

			if (false && terrainAS) {
				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				terrainAS->BuildOrUpdate(
					vkCmd,
					cameraPos,
					0.5f,
					push.gridParams.x,
					pipelineLibrary,
					&aabbShader,
					boundSets[0],
					boundSets[1],
					setLayouts[0],
					setLayouts[1],
					push.gridParams
				);
				if (physicalRegistry && terrainAS->GetTLAS()) {
					physicalRegistry->RegisterImportedAccelerationStructure<TerrainTLAS>(terrainAS->GetTLAS());
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainGenNode);

} // namespace brassica
