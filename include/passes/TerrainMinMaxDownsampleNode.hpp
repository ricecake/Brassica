#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainManager.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	// Mirrors shaders/terrain_downsample.comp's push_constant block exactly. Offsets verified via
	// spirv-dis against the real compiled shader (0, 4, 8, 12, 16; size 56) rather than hand math.
	// No vec/uvec members here, so plain 4-byte-aligned scalars/array -- no padding needed, unlike
	// the old workGroupOffset-carrying layout this replaced.
	struct TerrainDownsamplePushConstants {
		std::uint32_t srcIndex{0};      // bindless SAMPLED index for TerrainMinMaxTexture (mip 0 read)
		std::uint32_t mips{10};         // SPD output mip count -- our mip 1..10 for DownsampleDim = 1024
		std::uint32_t isFullRebuild{1}; // 1 = every tile dirty (first frame/force/resize), 0 = subregion
		std::uint32_t textureDim{1024}; // sub-portion map width (1024) for downsampling
		                                // and dirty-tile test
		std::array<std::uint32_t, 10> mipIndices{};                  // bindless STORAGE indices for our mip 1..10
	};

	static_assert(offsetof(TerrainDownsamplePushConstants, srcIndex) == 0);
	static_assert(offsetof(TerrainDownsamplePushConstants, mips) == 4);
	static_assert(offsetof(TerrainDownsamplePushConstants, isFullRebuild) == 8);
	static_assert(offsetof(TerrainDownsamplePushConstants, textureDim) == 12);
	static_assert(offsetof(TerrainDownsamplePushConstants, mipIndices) == 16);
	static_assert(sizeof(TerrainDownsamplePushConstants) == 56, "TerrainDownsamplePushConstants size must be 56 bytes");

	namespace detail {

		// A single storage-buffer binding for FFX SPD's atomic counter -- node-local, not the
		// bindless catalog (that arena's bindless storage-buffer binding has no GLSL declaration
		// yet, and this buffer specifically needs `coherent`, which no bindless declaration
		// carries). Same shape as ParticleSystemNode's ParticleDescriptorSet, just one binding.
		struct TerrainCounterSet {
			vk::DescriptorSetLayout layout{nullptr};
			vk::DescriptorPool      pool{nullptr};
			vk::DescriptorSet       set{nullptr};
		};

		inline TerrainCounterSet CreateTerrainCounterSet(vk::Device device) {
			TerrainCounterSet result;

			vk::DescriptorSetLayoutBinding binding{};
			binding.setBinding(0)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute);

			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(binding);
			result.layout = device.createDescriptorSetLayout(layoutInfo);

			vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eStorageBuffer, 1};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSize).setMaxSets(1);
			result.pool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(result.pool).setSetLayouts(result.layout);
			result.set = device.allocateDescriptorSets(allocInfo).front();

			return result;
		}

		inline void DestroyTerrainCounterSet(vk::Device device, TerrainCounterSet& s) {
			if (s.pool) {
				device.destroyDescriptorPool(s.pool);
				s.pool = nullptr;
			}
			if (s.layout) {
				device.destroyDescriptorSetLayout(s.layout);
				s.layout = nullptr;
			}
			s.set = nullptr;
		}

	} // namespace detail

	// Builds TerrainMinMaxTexture's mip chain (mips 1-10) from mip 0, which TerrainGenNode writes
	// every regeneration frame. A separate node, not a second dispatch inside
	// TerrainGenNode::Execute, specifically so the barrier between "write mip 0" and "read mip 0 /
	// write mips 1-10" is synthesized by the frame graph itself
	// (PhysicalExecutionBackend::RunSchedule's per-stage Acquire/Release batches) instead of a
	// hand-rolled vkCmd.pipelineBarrier2. TerrainGenNode Creates<TerrainMinMaxTexture> (mip 0);
	// this node Modifies<TerrainMinMaxTexture> (consumes mip 0, produces mips 1-10) -- an entirely
	// ordinary producer/consumer edge, which Graph::CollectEdges orders and barriers the normal
	// way, no self-modify auto-chain or version literal involved.
	//
	// Execute always dispatches the full tile grid (every slice), but shaders/terrain_downsample.
	// comp's main() derives the same per-layer toroidal dirty region terrain_gen.comp used to
	// update mip 0 and makes every non-dirty workgroup return before it ever touches SPD -- so on
	// a typical moving-camera frame, most of the 16x16-per-slice workgroups are a handful of ALU
	// ops instead of a full downsample. isFullRebuild (the push field) forces every tile dirty on
	// the first dispatch, an explicit regeneration, or a gridParams change.
	// S4: no kPhase -- see TerrainGenNode's identical comment. This node's Modify<
	// TerrainMinMaxTexture>() is ordered after TerrainGenNode's sole Create<> of the same key via
	// Create-before-accessor, and every reader of TerrainMinMaxTexture gets a real edge from this
	// node too (the default "reader sees final state" resource edge) -- no phase needed.
	struct TerrainMinMaxDownsampleNode: render::NodeRegistrar<TerrainMinMaxDownsampleNode> {
		// FFX SPD's fixed contract: one workgroup per 64x64 source tile. For DownsampleDim = 1024 that's
		// 16x16 tiles per slice -- matches the dispatch in Execute.
		static constexpr std::uint32_t kTileSize = 64;
		static constexpr std::uint32_t kDownsampleDim = 1024;
		static constexpr std::uint32_t kOutputMips = 10; // our mip 1..10 for DownsampleDim = 1024

		render::PipelineLibrary*       pipelineLibrary = nullptr;
		ComputeShader                  downsampleShader;
		detail::TerrainCounterSet      counterSet{};
		vk::Buffer                     counterBuffer{nullptr};
		VmaAllocation                  counterAllocation{nullptr};
		VmaAllocator                   allocator{nullptr};
		TerrainDownsamplePushConstants push{};
		glm::uvec4                     gridParams{
            constants::Class::Terrain::DefaultMaxLODs,
            constants::Class::Terrain::MeshletsPerRow,
            constants::Class::Terrain::TotalMeshlets,
            constants::Class::Terrain::MapDim
        };
		bool hasUpdate{true};
		bool forceRegeneration{true};

		// Tracks whether the last dispatch actually ran a full rebuild, so a gridParams change
		// (which reprovisions TerrainMinMaxTexture with a different layer count or width --
		// PhysicalRegistry::ProvisionTexture's desc-mismatch path) or the very first dispatch
		// forces a full rebuild instead of trusting coarse mips that may not exist yet.
		bool       hasDispatched{false};
		glm::uvec4 lastDispatchedGridParams{0, 0, 0, 0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!downsampleShader.CompileComputeFromFile(services.device, "shaders/terrain_downsample.comp")) {
				spdlog::critical("TerrainMinMaxDownsampleNode shader compilation failed.");
				throw std::runtime_error("TerrainMinMaxDownsampleNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&downsampleShader);
			}

			counterSet = detail::CreateTerrainCounterSet(services.device);

			if (!services.physicalRegistry) {
				return;
			}
			services.physicalRegistry->Catalog().Get<TerrainMinMaxTexture>();
			allocator = services.physicalRegistry->GetAllocator();

			// One uint per possible clipmap slice, sized for the max rather than whatever numLODs
			// happens to be at Init time -- small enough (DefaultMaxLODs * 4 bytes) that
			// over-allocating is free, and this buffer is never resized. Host-visible/coherent
			// rather than device-local: it's written exactly once, right here, and every access
			// after that is GPU-internal to SPD's own atomic-counter protocol, which self-resets
			// (shaders/terrain_downsample.comp's SpdResetAtomicCounter) -- there's no per-frame
			// transfer to make staging it through device-local memory worth the complexity.
			vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(constants::Class::Terrain::DefaultMaxLODs) *
				sizeof(std::uint32_t);

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = bufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          rawBuffer{};
			VmaAllocation     rawAllocation{};
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(
					allocator,
					&bufferInfo,
					&allocCreateInfo,
					&rawBuffer,
					&rawAllocation,
					&allocResultInfo
				) != VK_SUCCESS) {
				spdlog::critical("TerrainMinMaxDownsampleNode: failed to allocate the atomic-counter buffer.");
				throw std::runtime_error("TerrainMinMaxDownsampleNode: failed to allocate the atomic-counter buffer.");
			}
			counterBuffer = rawBuffer;
			counterAllocation = rawAllocation;
			if (allocResultInfo.pMappedData) {
				std::memset(allocResultInfo.pMappedData, 0, static_cast<std::size_t>(bufferSize));
			}

			vk::DescriptorBufferInfo bufferDescInfo{counterBuffer, 0, bufferSize};
			vk::WriteDescriptorSet   write{};
			write.setDstSet(counterSet.set)
				.setDstBinding(0)
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferDescInfo);
			services.device.updateDescriptorSets(write, {});
		}

		void Destroy(vk::Device device) {
			downsampleShader.Destroy(device);
			detail::DestroyTerrainCounterSet(device, counterSet);
			if (counterBuffer && allocator) {
				vmaDestroyBuffer(allocator, counterBuffer, counterAllocation);
				counterBuffer = nullptr;
				counterAllocation = nullptr;
			}
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
			forceRegeneration = p.forceRegeneration;
		}

		// TerrainMinMaxTexture is catalog-converted (ResourceKeys.hpp, numLODs field published by
		// TerrainGenNode::SetFrameParams every frame) -- no .Desc() needed here. That's what used
		// to make this node's hand-built desc a real, severe hazard: two nodes realizing one key
		// with mismatched descs (a stale gridParams.x read here vs. there) silently reprovisions
		// while in use. The catalog removes the second place that number had to be read correctly.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Modify<TerrainMinMaxTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			// Same gate TerrainGenNode uses -- only recompute the chain on a frame that actually
			// regenerated mip 0. TerrainGenNode's own realization above still keeps this resource
			// alive/provisioned every frame regardless of this node's activity.
			if (!hasUpdate && !forceRegeneration) {
				return;
			}

			// A gridParams change reprovisions TerrainMinMaxTexture with a different layer count or
			// width (PhysicalRegistry::ProvisionTexture's desc-mismatch path), so last frame's
			// coarse mips are gone -- that, the very first dispatch, and an explicit
			// forceRegeneration all mean the shader must treat every tile as dirty rather than
			// trust a subregion rebuild against mips that may not exist.
			bool isFullRebuild = forceRegeneration || !hasDispatched || gridParams != lastDispatchedGridParams;

			push.srcIndex = ctx.Index<TerrainMinMaxTexture>();
			for (std::uint32_t mip = 0; mip < kOutputMips; ++mip) {
				push.mipIndices[mip] = ctx.StorageIndex<TerrainMinMaxTexture>(mip + 1);
			}
			push.mips = kOutputMips;
			push.isFullRebuild = isFullRebuild ? 1u : 0u;
			push.textureDim = kDownsampleDim;
			std::uint32_t tilesPerAxis = (kDownsampleDim + kTileSize - 1) / kTileSize;

			std::array<vk::DescriptorSetLayout, 1> extraLayouts{counterSet.layout};
			std::array<vk::DescriptorSet, 1>       extraSets{counterSet.set};
			render::ResolvedPipeline               resolved = render::BindCompute(
                ctx,
                *pipelineLibrary,
                downsampleShader,
                push,
                render::ExtraSets{extraLayouts, extraSets}
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(tilesPerAxis, tilesPerAxis, gridParams.x);

			hasDispatched = true;
			lastDispatchedGridParams = gridParams;
		}
	};

	BRASSICA_REGISTER_NODE(TerrainMinMaxDownsampleNode);

} // namespace brassica
