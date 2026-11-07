#pragma once

#include <array>
#include <cstdint>
#include <vector>

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

namespace brassica {

	struct TerrainProxyResetPush {
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		std::uint32_t maxVirtualChunks{constants::Class::Terrain::Proxy::MaxVirtualChunks};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::Proxy::MaxAllocatedChunks};
		std::uint32_t indicesPerChunk{constants::Class::Terrain::Proxy::IndicesPerChunk};
		std::uint32_t pad{0};
	};

	struct TerrainProxyCullAllocPush {
		glm::uvec2   pageTableBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		glm::vec4    sunDir{0.0f, 1.0f, 0.0f, 1500.0f}; // xyz = sun direction, w = max cascade range
		glm::uvec4   gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		};
		std::uint32_t maxVirtualChunks{constants::Class::Terrain::Proxy::MaxVirtualChunks};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::Proxy::MaxAllocatedChunks};
		std::uint32_t minMaxIndex{0};
		std::uint32_t passMode{0}; // 0 = clear, 1 = alloc, 2 = finalize
	};

	struct TerrainProxyGeneratePush {
		glm::uvec2   vertexBufferAddr{0};
		glm::uvec2   indirectionBufferAddr{0};
		glm::uvec4   gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		};
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t maxAllocatedChunks{constants::Class::Terrain::Proxy::MaxAllocatedChunks};
		std::uint32_t chunkDimension{constants::Class::Terrain::Proxy::ChunkDimension};
	};

	// Generates lower-resolution proxy geometry for terrain shadow casting using
	// compute-driven chunk allocation, an indirection buffer, and a page table.
	struct TerrainProxyNode: render::NodeRegistrar<TerrainProxyNode> {
		render::PipelineLibrary*   pipelineLibrary = nullptr;
		ComputeShader              resetShader;
		ComputeShader              cullAllocShader;
		ComputeShader              generateShader;

		glm::vec3                  cameraPos{0.0f};
		glm::vec3                  sunDir{0.0f, 1.0f, 0.0f};
		glm::uvec4                 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		};
		bool                       initialized{false};
		bool                       forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!resetShader.CompileComputeFromFile(services.device, "shaders/terrain_proxy_reset.comp") ||
			    !cullAllocShader.CompileComputeFromFile(services.device, "shaders/terrain_proxy_cull_alloc.comp") ||
			    !generateShader.CompileComputeFromFile(services.device, "shaders/terrain_proxy_generate.comp")) {
				spdlog::critical("TerrainProxyNode shader compilation failed.");
				throw std::runtime_error("TerrainProxyNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&resetShader);
			watcher.RegisterShader(&cullAllocShader);
			watcher.RegisterShader(&generateShader);
		}

		void Destroy(vk::Device device) {
			resetShader.Destroy(device);
			cullAllocShader.Destroy(device);
			generateShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			sunDir = p.sunDir;
			gridParams = p.terrainGridParams;
			forceRegeneration = p.forceRegeneration;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainProxyPageTableBuffer>();
			r.Create<TerrainProxyIndirectionBuffer>();
			r.Create<TerrainProxyVertexBuffer>();
			r.Create<TerrainProxyIndexBuffer>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}

			auto ptBuf = physReg->GetBuffer<TerrainProxyPageTableBuffer>();
			auto indBuf = physReg->GetBuffer<TerrainProxyIndirectionBuffer>();
			auto vBuf = physReg->GetBuffer<TerrainProxyVertexBuffer>();
			auto idxBuf = physReg->GetBuffer<TerrainProxyIndexBuffer>();

			if (!ptBuf || !indBuf || !vBuf || !idxBuf) {
				return;
			}

			auto getAddress = [&](auto& buf) -> vk::DeviceAddress {
				vk::Buffer vkBuf = buf->GetBuffer();
				if (!vkBuf) return 0;
				vk::BufferDeviceAddressInfo bda(vkBuf);
				return physReg->GetDevice().getBufferAddress(bda);
			};

			vk::DeviceAddress ptAddr = getAddress(ptBuf);
			vk::DeviceAddress indAddr = getAddress(indBuf);
			vk::DeviceAddress vAddr = getAddress(vBuf);

			auto splitAddr = [](vk::DeviceAddress addr) {
				return glm::uvec2(
					static_cast<std::uint32_t>(addr & 0xFFFFFFFFu),
					static_cast<std::uint32_t>(addr >> 32u)
				);
			};

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// 1. One-time or forced reset
			if (!initialized || forceRegeneration) {
				// Also populate static index buffer for the canonical quad mesh if needed
				std::vector<std::uint32_t> indices;
				constexpr std::uint32_t dim = constants::Class::Terrain::Proxy::ChunkDimension;
				indices.reserve(constants::Class::Terrain::Proxy::IndicesPerChunk);
				for (std::uint32_t y = 0; y < dim - 1; ++y) {
					for (std::uint32_t x = 0; x < dim - 1; ++x) {
						std::uint32_t v0 = y * dim + x;
						std::uint32_t v1 = y * dim + (x + 1);
						std::uint32_t v2 = (y + 1) * dim + x;
						std::uint32_t v3 = (y + 1) * dim + (x + 1);

						indices.push_back(v0);
						indices.push_back(v2);
						indices.push_back(v1);

						indices.push_back(v1);
						indices.push_back(v2);
						indices.push_back(v3);
					}
				}

				ctx.WriteSpan<TerrainProxyIndexBuffer>(
					std::span<const std::uint32_t>(indices.data(), indices.size())
				);

				TerrainProxyResetPush resetPush{
					.pageTableBufferAddr = splitAddr(ptAddr),
					.indirectionBufferAddr = splitAddr(indAddr),
					.maxVirtualChunks = constants::Class::Terrain::Proxy::MaxVirtualChunks,
					.maxAllocatedChunks = constants::Class::Terrain::Proxy::MaxAllocatedChunks,
					.indicesPerChunk = constants::Class::Terrain::Proxy::IndicesPerChunk,
					.pad = 0
				};

				render::ResolvedPipeline resResolved = render::BindCompute(ctx, *pipelineLibrary, resetShader, resetPush);
				if (resResolved.pipeline) {
					std::uint32_t groupCount = (constants::Class::Terrain::Proxy::MaxVirtualChunks + 63u) / 64u;
					vkCmd.dispatch(groupCount, 1, 1);

					vk::MemoryBarrier b{
						vk::AccessFlagBits::eShaderWrite,
						vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
					};
					vkCmd.pipelineBarrier(
						vk::PipelineStageFlagBits::eComputeShader,
						vk::PipelineStageFlagBits::eComputeShader,
						vk::DependencyFlags{},
						b,
						nullptr,
						nullptr
					);
				}
				initialized = true;
			}

			// 2. Cull & Allocate Chunks via Page Table and Indirection Buffer
			TerrainProxyCullAllocPush cullPush{
				.pageTableBufferAddr = splitAddr(ptAddr),
				.indirectionBufferAddr = splitAddr(indAddr),
				.sunDir = glm::vec4(sunDir, 2500.0f),
				.gridParams = gridParams,
				.maxVirtualChunks = constants::Class::Terrain::Proxy::MaxVirtualChunks,
				.maxAllocatedChunks = constants::Class::Terrain::Proxy::MaxAllocatedChunks,
				.minMaxIndex = ctx.Index<TerrainMinMaxTexture>(),
				.passMode = 0 // Clear active counter
			};

			render::ResolvedPipeline cullResolved = render::BindCompute(ctx, *pipelineLibrary, cullAllocShader, cullPush);
			if (cullResolved.pipeline) {
				// Pass 0: Clear active chunk counter
				vkCmd.dispatch(1, 1, 1);

				vk::MemoryBarrier b0{
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
				};
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					b0,
					nullptr,
					nullptr
				);

				// Pass 1: Evaluate & allocate virtual chunks
				cullPush.passMode = 1;
				vkCmd.pushConstants(
					cullResolved.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(TerrainProxyCullAllocPush),
					&cullPush
				);
				std::uint32_t totalChunks = std::min(gridParams.z, constants::Class::Terrain::Proxy::MaxVirtualChunks);
				std::uint32_t groupCount = (totalChunks + 63u) / 64u;
				vkCmd.dispatch(groupCount, 1, 1);

				// Pass 2: Finalize indirect draw/dispatch args
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					b0,
					nullptr,
					nullptr
				);

				cullPush.passMode = 2;
				vkCmd.pushConstants(
					cullResolved.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(TerrainProxyCullAllocPush),
					&cullPush
				);
				vkCmd.dispatch(1, 1, 1);

				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					b0,
					nullptr,
					nullptr
				);
			}

			// 3. Generate Proxy Chunk Vertices for Active Chunks
			TerrainProxyGeneratePush genPush{
				.vertexBufferAddr = splitAddr(vAddr),
				.indirectionBufferAddr = splitAddr(indAddr),
				.gridParams = gridParams,
				.clipmapIndex = ctx.Index<TerrainClipmapTexture>(),
				.minMaxIndex = ctx.Index<TerrainMinMaxTexture>(),
				.maxAllocatedChunks = constants::Class::Terrain::Proxy::MaxAllocatedChunks,
				.chunkDimension = constants::Class::Terrain::Proxy::ChunkDimension
			};

			render::ResolvedPipeline genResolved = render::BindCompute(ctx, *pipelineLibrary, generateShader, genPush);
			if (genResolved.pipeline) {
				// Dispatch across up to MaxAllocatedChunks (each workgroup checks if within activeChunkCount)
				vkCmd.dispatch(constants::Class::Terrain::Proxy::MaxAllocatedChunks, 1, 1);
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainProxyNode);

} // namespace brassica
