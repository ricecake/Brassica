#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

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
#include "vk_mem_alloc.h"

namespace brassica {

	struct HiZDownsamplePushConstants {
		std::uint32_t srcIndex{0};       // bindless SAMPLED index for GBufferDepth (mip 0 read)
		std::uint32_t mips{10};          // SPD output mips
		std::uint32_t numWorkGroups{0};  // workgroups dispatched
		std::uint32_t _pad0{0};
		glm::uvec2    workGroupOffset{0, 0};        // (0,0)
		std::array<std::uint32_t, 12> mipIndices{}; // bindless STORAGE indices for mips 1..12
	};

	static_assert(offsetof(HiZDownsamplePushConstants, srcIndex) == 0);
	static_assert(offsetof(HiZDownsamplePushConstants, mips) == 4);
	static_assert(offsetof(HiZDownsamplePushConstants, numWorkGroups) == 8);
	static_assert(offsetof(HiZDownsamplePushConstants, _pad0) == 12);
	static_assert(offsetof(HiZDownsamplePushConstants, workGroupOffset) == 16);
	static_assert(offsetof(HiZDownsamplePushConstants, mipIndices) == 24);
	static_assert(sizeof(HiZDownsamplePushConstants) == 72, "HiZDownsamplePushConstants size must be 72 bytes");

	namespace detail {

		struct HiZCounterSet {
			vk::DescriptorSetLayout layout{nullptr};
			vk::DescriptorPool      pool{nullptr};
			vk::DescriptorSet       set{nullptr};
		};

		inline HiZCounterSet CreateHiZCounterSet(vk::Device device) {
			HiZCounterSet result;

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

		inline void DestroyHiZCounterSet(vk::Device device, HiZCounterSet& s) {
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

	// Builds HiZTexture's mip chain (min/max depth pyramid) from GBufferDepth mip 0.
	// Executed in SubPhase::LightPreparation (Phase 500) so HiZ depth data is available for
	// light clustering, occlusion culling, reflections, and screen space shadow ray marching.
	struct HiZDownsampleNode: render::NodeRegistrar<HiZDownsampleNode> {
		using Resources = graph::Declares<graph::Read<GBufferDepth>, graph::Create<HiZTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		static constexpr std::uint32_t kTileSize = 64;

		render::PipelineLibrary*   pipelineLibrary = nullptr;
		ComputeShader              downsampleShader;
		detail::HiZCounterSet      counterSet{};
		vk::Buffer                 counterBuffer{nullptr};
		VmaAllocation              counterAllocation{nullptr};
		VmaAllocator               allocator{nullptr};
		HiZDownsamplePushConstants push{};

		std::uint32_t width{0};
		std::uint32_t height{0};
		std::uint32_t outputMips{0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!downsampleShader.CompileComputeFromFile(services.device, "shaders/hiz_downsample.comp")) {
				spdlog::critical("HiZDownsampleNode shader compilation failed.");
				throw std::runtime_error("HiZDownsampleNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&downsampleShader);
			}

			counterSet = detail::CreateHiZCounterSet(services.device);

			if (!services.physicalRegistry) {
				return;
			}
			allocator = services.physicalRegistry->GetAllocator();

			vk::DeviceSize bufferSize = sizeof(std::uint32_t);

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
				spdlog::critical("HiZDownsampleNode: failed to allocate the atomic-counter buffer.");
				throw std::runtime_error("HiZDownsampleNode: failed to allocate the atomic-counter buffer.");
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
			detail::DestroyHiZCounterSet(device, counterSet);
			if (counterBuffer && allocator) {
				vmaDestroyBuffer(allocator, counterBuffer, counterAllocation);
				counterBuffer = nullptr;
				counterAllocation = nullptr;
			}
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;
			std::uint32_t maxDim = (std::max)(width, height);
			outputMips = maxDim > 0 ? static_cast<std::uint32_t>(std::floor(std::log2(static_cast<float>(maxDim)))) : 1;
			outputMips = (std::min)(outputMips, 12u);

			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Read,
					.desc = graph::DepthBufferDesc(width, height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HiZTexture>(),
					.access = graph::AccessKind::Write,
					.desc = HiZTextureDesc(width, height, outputMips + 1),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (width == 0 || height == 0 || outputMips == 0) {
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

			push.srcIndex = ctx.Index<GBufferDepth>();
			for (std::uint32_t mip = 0; mip < outputMips; ++mip) {
				push.mipIndices[mip] = ctx.StorageIndex<HiZTexture>(mip + 1);
			}
			push.mips = outputMips;
			std::uint32_t tilesX = (width + kTileSize - 1) / kTileSize;
			std::uint32_t tilesY = (height + kTileSize - 1) / kTileSize;
			push.numWorkGroups = tilesX * tilesY;
			push.workGroupOffset = glm::uvec2(0, 0);

			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(HiZDownsamplePushConstants)}
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
				sizeof(HiZDownsamplePushConstants),
				&push
			);

			vkCmd.dispatch(tilesX, tilesY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(HiZDownsampleNode);

} // namespace brassica
