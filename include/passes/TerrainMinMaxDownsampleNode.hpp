#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "terrain/TerrainManager.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	// Mirrors shaders/terrain_downsample.comp's push_constant block exactly. Offsets verified via
	// spirv-dis against the real compiled shader (0, 4, 8, 16, 24; size 64) rather than hand math
	// -- GLSL's push-constant rules align uvec2 to 8 bytes, which glm::uvec2 does NOT get for
	// free in C++ (alignof(glm::uvec2) == 4 here), hence the explicit _pad0.
	struct TerrainDownsamplePushConstants {
		std::uint32_t srcIndex{0};      // bindless SAMPLED index for TerrainMinMaxTexture (mip 0 read)
		std::uint32_t mips{10};         // SPD output mip count -- our mip 1..10 for MapDim = 1024
		std::uint32_t numWorkGroups{0}; // workgroups dispatched per slice (excludes Z/slice dim)
		std::uint32_t _pad0{0};
		glm::uvec2    workGroupOffset{0, 0};        // (0,0) until incremental dirty-rect dispatch lands
		std::array<std::uint32_t, 10> mipIndices{}; // bindless STORAGE indices for our mip 1..10
	};

	static_assert(offsetof(TerrainDownsamplePushConstants, srcIndex) == 0);
	static_assert(offsetof(TerrainDownsamplePushConstants, mips) == 4);
	static_assert(offsetof(TerrainDownsamplePushConstants, numWorkGroups) == 8);
	static_assert(offsetof(TerrainDownsamplePushConstants, workGroupOffset) == 16);
	static_assert(offsetof(TerrainDownsamplePushConstants, mipIndices) == 24);
	static_assert(sizeof(TerrainDownsamplePushConstants) == 64, "TerrainDownsamplePushConstants size must be 64 bytes");

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

			vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer, 1};
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
	struct TerrainMinMaxDownsampleNode: render::NodeRegistrar<TerrainMinMaxDownsampleNode> {
		using Resources = graph::Declares<graph::Modify<TerrainMinMaxTexture>>;

		// See TerrainGenNode's identical comment: particle shaders now read terrain data at
		// SubPhase::Prepare, so every terrain-producing node has to run at or before Prepare too.
		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		// FFX SPD's fixed contract: one workgroup per 64x64 source tile. For MapDim = 1024 that's
		// 16x16 tiles per slice -- matches the dispatch in Execute.
		static constexpr std::uint32_t kTileSize = 64;
		static constexpr std::uint32_t kOutputMips = 10; // our mip 1..10 for MapDim = 1024

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
			allocator = services.physicalRegistry->GetAllocator();

			// One uint per possible clipmap slice, sized for the max rather than whatever numLODs
			// happens to be at Init time -- small enough (DefaultMaxLODs * 4 bytes) that
			// over-allocating is free, and this buffer is never resized. Host-visible/coherent
			// rather than device-local: it's written exactly once, right here, and every access
			// after that is GPU-internal to SPD's own atomic-counter protocol, which self-resets
			// (shaders/terrain_downsample.comp's SpdResetAtomicCounter) -- there's no per-frame
			// transfer to make staging it through device-local memory worth the complexity.
			vk::DeviceSize bufferSize =
				static_cast<vk::DeviceSize>(constants::Class::Terrain::DefaultMaxLODs) * sizeof(std::uint32_t);

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = bufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          rawBuffer{};
			VmaAllocation     rawAllocation{};
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &rawBuffer, &rawAllocation, &allocResultInfo) !=
			    VK_SUCCESS) {
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

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxTexture>(),
					.access = graph::AccessKind::ReadWrite,
					// Identical desc-building call to TerrainGenNode's own realization of this
					// same key -- two nodes realizing one key with mismatched descs is a real,
					// severe bug (silent reprovision-while-in-use), not just a style nit.
					.desc = TerrainMinMaxDesc(gridParams.x),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			// Same gate TerrainGenNode uses -- only recompute the chain on a frame that actually
			// regenerated mip 0. TerrainGenNode's own realization above still keeps this resource
			// alive/provisioned every frame regardless of this node's activity.
			if (!hasUpdate && !forceRegeneration) {
				return;
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)),
				counterSet.layout,
			};
			std::array<vk::DescriptorSet, 3> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet)),
				counterSet.set,
			};

			push.srcIndex = ctx.Index<TerrainMinMaxTexture>();
			for (std::uint32_t mip = 0; mip < kOutputMips; ++mip) {
				push.mipIndices[mip] = ctx.StorageIndex<TerrainMinMaxTexture>(mip + 1);
			}
			push.mips = kOutputMips;
			std::uint32_t tilesPerAxis = (gridParams.w + kTileSize - 1) / kTileSize;
			push.numWorkGroups = tilesPerAxis * tilesPerAxis;
			push.workGroupOffset = glm::uvec2(0, 0);

			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainDownsamplePushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &downsampleShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			if (boundSets[0] && boundSets[1] && boundSets[2]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(TerrainDownsamplePushConstants),
				&push
			);

			vkCmd.dispatch(tilesPerAxis, tilesPerAxis, gridParams.x);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainMinMaxDownsampleNode);

} // namespace brassica
