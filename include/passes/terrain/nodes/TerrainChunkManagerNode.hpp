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
#include "ServiceLocator.hpp"
#include "passes/RenderPhases.hpp"
#include "resources.hpp"
#include "passes/terrain/ITerrainManager.hpp"
#include "passes/terrain/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {


	namespace terrain_chunk_detail {
		inline bool IsChunkSystemEnabled() {
			if (ServiceLocator::Instance().Has<ITerrainManager>()) {
				return ServiceLocator::Instance().Get<ITerrainManager>()->GetState().useChunkSystem;
			}
			return true;
		}

		inline glm::uvec2 SplitDeviceAddress(vk::DeviceAddress addr) {
			return glm::uvec2(
				static_cast<std::uint32_t>(addr & 0xFFFFFFFFu),
				static_cast<std::uint32_t>(addr >> 32u)
			);
		}

		inline vk::DeviceAddress GetBufferDeviceAddress(
			const graph::PhysicalResourceRegistry* physReg,
			vk::Buffer buffer
		) {
			if (!physReg || !buffer) {
				return 0;
			}
			vk::BufferDeviceAddressInfo info(buffer);
			return physReg->GetDevice().getBufferAddress(info);
		}
	} // namespace terrain_chunk_detail

	// =========================================================================
	// Push Constant Structs
	// =========================================================================

	struct TerrainChunkResetPush {
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		glm::uvec2   requirementsBufferAddr{0};
		std::uint32_t maxVirtualChunks{constants::Class::Terrain::ChunkManager::MaxVirtualChunks};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::ChunkManager::MaxAllocatedChunks};
	};

	struct TerrainChunkLodPush {
		glm::uvec2   chunkLodBufferAddr{0};
		glm::uvec2   padding{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
	};

	struct TerrainChunkEvictPush {
		glm::uvec2   chunkLodBufferAddr{0};
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		glm::uvec2   padding{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
		std::uint32_t maxVirtualChunks{constants::Class::Terrain::ChunkManager::MaxVirtualChunks};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::ChunkManager::MaxAllocatedChunks};
	};

	struct TerrainChunkRequirementsPush {
		glm::uvec2   chunkLodBufferAddr{0};
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   requirementsBufferAddr{0};
		glm::uvec2   padding{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
	};

	struct TerrainChunkAllocPush {
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		glm::uvec2   requirementsBufferAddr{0};
		glm::uvec2   padding{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
		std::uint32_t maxVirtualChunks{constants::Class::Terrain::ChunkManager::MaxVirtualChunks};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::ChunkManager::MaxAllocatedChunks};
	};

	struct TerrainChunkGeneratePush {
		glm::uvec2   vertexBufferAddr{0};
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   requirementsBufferAddr{0};
		glm::uvec2   padding{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
		std::uint32_t chunkDimension{constants::Class::Terrain::ChunkManager::ChunkDimension};
	};

	struct TerrainChunkRenderPushConstants {
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   vertexBufferAddr{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::ChunkManager::NumLODs,
			constants::Class::Terrain::ChunkManager::ChunksPerRow,
			constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
			constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
		};
		float         baseChunkSize{64.0f};
		float         waterLevel{0.0f};
		std::uint32_t minMaxIndex{0};
		std::uint32_t hizIndex{0};
	};

	// =========================================================================
	// 1. TerrainChunkResetNode: Resets Page Table, Indirection Stack, & Requirements
	// =========================================================================
	struct TerrainChunkResetNode: render::NodeRegistrar<TerrainChunkResetNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            resetShader;
		bool                     initialized = false;
		bool                     forceRegeneration = true;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!resetShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_reset.comp")) {
				spdlog::critical("TerrainChunkResetNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkResetNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&resetShader);
		}

		void Destroy(vk::Device device) {
			resetShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			forceRegeneration = p.forceRegeneration;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainPageTableBuffer>();
			r.Create<TerrainIndirectionBuffer>();
			r.Create<TerrainChunkLodBuffer>();
			r.Create<TerrainChunkRequirementsBuffer>();
			r.Create<TerrainChunkVertexBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (ServiceLocator::Instance().Has<ITerrainManager>()) {
				if (!ServiceLocator::Instance().Get<ITerrainManager>()->GetState().useChunkSystem) {
					return;
				}
			}

			if (initialized && !forceRegeneration) {
				return;
			}

			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto indBuf = physReg->GetBuffer<TerrainIndirectionBuffer>();
			auto reqBuf = physReg->GetBuffer<TerrainChunkRequirementsBuffer>();

			if (!ptBuf || !indBuf || !reqBuf) return;

			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress indAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, indBuf->GetBuffer());
			vk::DeviceAddress reqAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, reqBuf->GetBuffer());

			TerrainChunkResetPush push{
				.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr),
				.indirectionBufferAddr = terrain_chunk_detail::SplitDeviceAddress(indAddr),
				.requirementsBufferAddr = terrain_chunk_detail::SplitDeviceAddress(reqAddr),
				.maxVirtualChunks = constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
				.maxAllocatedChunks = constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, resetShader, push);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t groupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);

			initialized = true;
		}
	};

	// =========================================================================
	// 2. TerrainChunkLODNode: Decides what LOD each chunk needs
	// =========================================================================
	struct TerrainChunkLODNode: render::NodeRegistrar<TerrainChunkLODNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            lodShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!lodShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_lod.comp")) {
				spdlog::critical("TerrainChunkLODNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkLODNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&lodShader);
		}

		void Destroy(vk::Device device) {
			lodShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Modify<TerrainChunkLodBuffer>();
			r.After<TerrainChunkResetNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;

			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto lodBuf = physReg->GetBuffer<TerrainChunkLodBuffer>();
			if (!lodBuf) return;

			vk::DeviceAddress lodAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, lodBuf->GetBuffer());

			TerrainChunkLodPush push{
				.chunkLodBufferAddr = terrain_chunk_detail::SplitDeviceAddress(lodAddr),
				.gridParams = glm::uvec4(
					constants::Class::Terrain::ChunkManager::NumLODs,
					constants::Class::Terrain::ChunkManager::ChunksPerRow,
					constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
					constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
				),
				.baseChunkSize = 64.0f
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, lodShader, push);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t groupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	// =========================================================================
	// 3. TerrainChunkEvictNode: Removes stale/mismatched chunk entries, frees physical slots
	// =========================================================================
	struct TerrainChunkEvictNode: render::NodeRegistrar<TerrainChunkEvictNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            evictShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!evictShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_evict.comp")) {
				spdlog::critical("TerrainChunkEvictNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkEvictNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&evictShader);
		}

		void Destroy(vk::Device device) {
			evictShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<TerrainChunkLodBuffer>();
			r.Modify<TerrainPageTableBuffer>();
			r.Modify<TerrainIndirectionBuffer>();
			r.After<TerrainChunkLODNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto lodBuf = physReg->GetBuffer<TerrainChunkLodBuffer>();
			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto indBuf = physReg->GetBuffer<TerrainIndirectionBuffer>();
			if (!lodBuf || !ptBuf || !indBuf) return;

			vk::DeviceAddress lodAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, lodBuf->GetBuffer());
			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress indAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, indBuf->GetBuffer());

			TerrainChunkEvictPush push{
				.chunkLodBufferAddr = terrain_chunk_detail::SplitDeviceAddress(lodAddr),
				.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr),
				.indirectionBufferAddr = terrain_chunk_detail::SplitDeviceAddress(indAddr),
				.gridParams = glm::uvec4(
					constants::Class::Terrain::ChunkManager::NumLODs,
					constants::Class::Terrain::ChunkManager::ChunksPerRow,
					constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
					constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
				),
				.baseChunkSize = 64.0f,
				.maxVirtualChunks = constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
				.maxAllocatedChunks = constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, evictShader, push);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t groupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	// =========================================================================
	// 4. TerrainChunkRequirementsNode: Populates missing requirements
	// =========================================================================
	struct TerrainChunkRequirementsNode: render::NodeRegistrar<TerrainChunkRequirementsNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            reqShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!reqShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_requirements.comp")) {
				spdlog::critical("TerrainChunkRequirementsNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkRequirementsNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&reqShader);
		}

		void Destroy(vk::Device device) {
			reqShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<TerrainChunkLodBuffer>();
			r.Read<TerrainPageTableBuffer>();
			r.Modify<TerrainChunkRequirementsBuffer>();
			r.After<TerrainChunkEvictNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto lodBuf = physReg->GetBuffer<TerrainChunkLodBuffer>();
			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto reqBuf = physReg->GetBuffer<TerrainChunkRequirementsBuffer>();
			if (!lodBuf || !ptBuf || !reqBuf) return;

			vk::DeviceAddress lodAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, lodBuf->GetBuffer());
			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress reqAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, reqBuf->GetBuffer());

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Clear requestCount (first 4 bytes of TerrainChunkRequirementsBuffer)
			vkCmd.fillBuffer(reqBuf->GetBuffer(), 0, 4, 0u);

			vk::MemoryBarrier b{
				vk::AccessFlagBits::eTransferWrite,
				vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
			};
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				b,
				nullptr,
				nullptr
			);

			TerrainChunkRequirementsPush push{
				.chunkLodBufferAddr = terrain_chunk_detail::SplitDeviceAddress(lodAddr),
				.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr),
				.requirementsBufferAddr = terrain_chunk_detail::SplitDeviceAddress(reqAddr),
				.gridParams = glm::uvec4(
					constants::Class::Terrain::ChunkManager::NumLODs,
					constants::Class::Terrain::ChunkManager::ChunksPerRow,
					constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
					constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
				),
				.baseChunkSize = 64.0f
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, reqShader, push);
			if (!resolved.pipeline) return;

			std::uint32_t groupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	// =========================================================================
	// 5. TerrainChunkAllocNode: Allocates physical slots into Page Table from free stack
	// =========================================================================
	struct TerrainChunkAllocNode: render::NodeRegistrar<TerrainChunkAllocNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            allocShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!allocShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_alloc.comp")) {
				spdlog::critical("TerrainChunkAllocNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkAllocNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&allocShader);
		}

		void Destroy(vk::Device device) {
			allocShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<TerrainChunkRequirementsBuffer>();
			r.Modify<TerrainPageTableBuffer>();
			r.Modify<TerrainIndirectionBuffer>();
			r.After<TerrainChunkRequirementsNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto indBuf = physReg->GetBuffer<TerrainIndirectionBuffer>();
			auto reqBuf = physReg->GetBuffer<TerrainChunkRequirementsBuffer>();
			if (!ptBuf || !indBuf || !reqBuf) return;

			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress indAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, indBuf->GetBuffer());
			vk::DeviceAddress reqAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, reqBuf->GetBuffer());

			TerrainChunkAllocPush push{
				.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr),
				.indirectionBufferAddr = terrain_chunk_detail::SplitDeviceAddress(indAddr),
				.requirementsBufferAddr = terrain_chunk_detail::SplitDeviceAddress(reqAddr),
				.gridParams = glm::uvec4(
					constants::Class::Terrain::ChunkManager::NumLODs,
					constants::Class::Terrain::ChunkManager::ChunksPerRow,
					constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
					constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
				),
				.baseChunkSize = 64.0f,
				.maxVirtualChunks = constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
				.maxAllocatedChunks = constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, allocShader, push);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t groupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	// =========================================================================
	// 6. TerrainChunkGenerateNode: Generates terrain chunk vertex data into buffer
	// =========================================================================
	struct TerrainChunkGenerateNode: render::NodeRegistrar<TerrainChunkGenerateNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            genShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_generate.comp")) {
				spdlog::critical("TerrainChunkGenerateNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkGenerateNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<TerrainChunkRequirementsBuffer>();
			r.Read<TerrainPageTableBuffer>();
			r.Modify<TerrainChunkVertexBuffer>();
			r.After<TerrainChunkAllocNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto vBuf = physReg->GetBuffer<TerrainChunkVertexBuffer>();
			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto reqBuf = physReg->GetBuffer<TerrainChunkRequirementsBuffer>();
			if (!vBuf || !ptBuf || !reqBuf) return;

			vk::DeviceAddress vAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, vBuf->GetBuffer());
			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress reqAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, reqBuf->GetBuffer());

			TerrainChunkGeneratePush push{
				.vertexBufferAddr = terrain_chunk_detail::SplitDeviceAddress(vAddr),
				.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr),
				.requirementsBufferAddr = terrain_chunk_detail::SplitDeviceAddress(reqAddr),
				.gridParams = glm::uvec4(
					constants::Class::Terrain::ChunkManager::NumLODs,
					constants::Class::Terrain::ChunkManager::ChunksPerRow,
					constants::Class::Terrain::ChunkManager::MaxVirtualChunks,
					constants::Class::Terrain::ChunkManager::MaxAllocatedChunks
				),
				.baseChunkSize = 64.0f,
				.chunkDimension = constants::Class::Terrain::ChunkManager::ChunkDimension
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Dispatch 1 workgroup per requested chunk (up to MaxVirtualChunks)
			vkCmd.dispatch(constants::Class::Terrain::ChunkManager::MaxVirtualChunks, 1, 1);
		}
	};

	// =========================================================================
	// 7. TerrainChunkRenderNode: Task & Mesh shader culling and rasterization
	// =========================================================================
	struct TerrainChunkRenderNode: render::NodeRegistrar<TerrainChunkRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*              pipelineLibrary = nullptr;
		TaskShader                             taskShader;
		MeshShader                             meshShader;
		FragmentShader                         fragShader;
		const brassica::DispatchLoaderDynamic* dls = nullptr;
		TerrainChunkRenderPushConstants        push{};
		bool                                   enabled = true;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/terrain_chunk.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/terrain_chunk.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/terrain.frag")) {
				spdlog::critical("TerrainChunkRenderNode shader compilation failed.");
				throw std::runtime_error("TerrainChunkRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.waterLevel = p.waterLevel;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferMaterial>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.Read<TerrainPageTableBuffer>();
			r.Read<TerrainChunkVertexBuffer>();
			r.Read<graph::History<HiZTexture>>();
			r.After<TerrainChunkGenerateNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!terrain_chunk_detail::IsChunkSystemEnabled()) return;
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) return;

			auto ptBuf = physReg->GetBuffer<TerrainPageTableBuffer>();
			auto vBuf = physReg->GetBuffer<TerrainChunkVertexBuffer>();
			if (!ptBuf || !vBuf) return;

			vk::DeviceAddress ptAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, ptBuf->GetBuffer());
			vk::DeviceAddress vAddr = terrain_chunk_detail::GetBufferDeviceAddress(physReg, vBuf->GetBuffer());

			push.pageTableBufferAddr = terrain_chunk_detail::SplitDeviceAddress(ptAddr);
			push.vertexBufferAddr = terrain_chunk_detail::SplitDeviceAddress(vAddr);
			push.hizIndex = ctx.Index<graph::History<HiZTexture>>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
				vk::Format::eR32G32B32A32Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Srgb,
				vk::Format::eR8G8B8A8Unorm
			};

			render::ResolvedPipeline resolved = render::BindGraphics(
				ctx,
				*pipelineLibrary,
				stages,
				kPipelineState,
				colorFormats,
				push,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
					vk::ShaderStageFlagBits::eFragment,
				vk::Format::eD32Sfloat
			);
			if (!resolved.pipeline) return;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Each chunk is rendered via 2 mesh half-workgroups, so total task items = MaxVirtualChunks * 2
			std::uint32_t taskGroupCount = (constants::Class::Terrain::ChunkManager::MaxVirtualChunks * 2u + 31u) / 32u;
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainChunkResetNode);
	BRASSICA_REGISTER_NODE(TerrainChunkLODNode);
	BRASSICA_REGISTER_NODE(TerrainChunkEvictNode);
	BRASSICA_REGISTER_NODE(TerrainChunkRequirementsNode);
	BRASSICA_REGISTER_NODE(TerrainChunkAllocNode);
	BRASSICA_REGISTER_NODE(TerrainChunkGenerateNode);
	BRASSICA_REGISTER_NODE(TerrainChunkRenderNode);

} // namespace brassica
