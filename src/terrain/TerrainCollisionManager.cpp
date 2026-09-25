#include "terrain/TerrainCollisionManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if BRASSICA_HAS_VULKAN
	#include "Engine.hpp"
	#include "Shader.hpp"
	#include "graph/PhysicalRegistry.hpp"
	#include "render/PipelineLibrary.hpp"
	#include "spdlog/spdlog.h"
#endif

namespace brassica {

	struct TerrainQueryPushConstants {
		uint32_t  width{256};
		uint32_t  height{256};
		glm::vec2 minWorldPos{0.0f, 0.0f};
		glm::vec2 texelSize{1.0f, 1.0f};
		float     planetRadius{600000.0f};
	};

	void TerrainCollisionManager::UpdateBufferCPU(const TerrainQueryRange& range) {
		m_range = range;
		uint32_t w = std::max(1u, range.width);
		uint32_t h = std::max(1u, range.height);
		size_t   total = static_cast<size_t>(w) * h;

		m_cpuBuffer.resize(total);

		glm::vec2 minPos = range.GetMinWorldPos();
		glm::vec2 texelSize = range.GetTexelSize();

		for (uint32_t z = 0; z < h; ++z) {
			float worldZ = minPos.y + static_cast<float>(z) * texelSize.y;
			for (uint32_t x = 0; x < w; ++x) {
				float     worldX = minPos.x + static_cast<float>(x) * texelSize.x;
				glm::vec4 sample = TerrainClipmap::SampleTerrain(worldX, worldZ, std::min(texelSize.x, texelSize.y));
				size_t    idx = static_cast<size_t>(z) * w + x;
				m_cpuBuffer[idx] = sample;
			}
		}

		m_hasData = true;
	}

	bool TerrainCollisionManager::UpdateBufferGPU([[maybe_unused]] Engine& engine, const TerrainQueryRange& range) {
#if !BRASSICA_HAS_VULKAN
		UpdateBufferCPU(range);
		return true;
#else
		vk::Device device = engine.GetDevice();
		if (!device) {
			UpdateBufferCPU(range);
			return false;
		}

		try {
			uint32_t     w = std::max(1u, range.width);
			uint32_t     h = std::max(1u, range.height);
			size_t       totalSamples = static_cast<size_t>(w) * h;
			VkDeviceSize bufferSize = totalSamples * sizeof(glm::vec4);

			VmaAllocator allocator = engine.GetAllocator();
			auto         pipelineLibrary = engine.GetService<render::PipelineLibrary>();

			// 1. Create GPU Storage Buffer
			VkBufferCreateInfo ssboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			ssboInfo.size = bufferSize;
			ssboInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

			VmaAllocationCreateInfo ssboAllocInfo{};
			ssboAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

			VkBuffer      ssboBuffer = VK_NULL_HANDLE;
			VmaAllocation ssboAllocation = nullptr;
			if (vmaCreateBuffer(allocator, &ssboInfo, &ssboAllocInfo, &ssboBuffer, &ssboAllocation, nullptr) != VK_SUCCESS) {
				spdlog::warn("vmaCreateBuffer failed for TerrainCollisionManager; falling back to CPU.");
				UpdateBufferCPU(range);
				return false;
			}

			// 2. Create Host Staging Buffer
			VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			stagingInfo.size = bufferSize;
			stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

			VmaAllocationCreateInfo stagingAllocInfo{};
			stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          stagingBuffer = VK_NULL_HANDLE;
			VmaAllocation     stagingAllocation = nullptr;
			VmaAllocationInfo stagingResultInfo{};
			if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingResultInfo) != VK_SUCCESS) {
				vmaDestroyBuffer(allocator, ssboBuffer, ssboAllocation);
				UpdateBufferCPU(range);
				return false;
			}

			// 3. Compile Compute Shader & Build Pipeline
			ComputeShader queryShader;
			if (!queryShader.CompileComputeFromFile(device, "shaders/terrain_query.comp")) {
				spdlog::warn("shaders/terrain_query.comp compilation failed; falling back to CPU.");
				vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
				vmaDestroyBuffer(allocator, ssboBuffer, ssboAllocation);
				UpdateBufferCPU(range);
				return false;
			}

			// Descriptor set layout: Binding 0 = TerrainQuerySSBO (Storage Buffer)
			vk::DescriptorSetLayoutBinding ssboBinding{};
			ssboBinding.setBinding(0)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute);

			std::array<vk::DescriptorSetLayoutBinding, 1> queryBindings{ssboBinding};
			vk::DescriptorSetLayoutCreateInfo             queryLayoutInfo{};
			queryLayoutInfo.setBindings(queryBindings);
			vk::DescriptorSetLayout querySetLayout = device.createDescriptorSetLayout(queryLayoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1}
			};
			vk::DescriptorPoolCreateInfo queryPoolInfo{};
			queryPoolInfo.setPoolSizes(poolSizes);
			queryPoolInfo.setMaxSets(1);
			vk::DescriptorPool queryPool = device.createDescriptorPool(queryPoolInfo);

			std::array<vk::DescriptorSetLayout, 1> allocLayouts{querySetLayout};
			vk::DescriptorSetAllocateInfo          allocInfo{queryPool, allocLayouts};
			vk::DescriptorSet                      querySet = device.allocateDescriptorSets(allocInfo).front();

			vk::DescriptorBufferInfo descriptorBufInfo{ssboBuffer, 0, bufferSize};
			vk::WriteDescriptorSet   write{};
			write.setDstSet(querySet)
				.setDstBinding(0)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(descriptorBufInfo);
			device.updateDescriptorSets(write, nullptr);

			std::array<vk::DescriptorSetLayout, 1> setLayouts{querySetLayout};
			std::array<vk::PushConstantRange, 1>   pushRanges{
				  vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainQueryPushConstants)}
			};

			render::ComputePipelineRequest pipelineReq{
				.shader = &queryShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRanges
			};
			render::ResolvedPipeline resolvedPipeline = pipelineLibrary->ResolveCached(pipelineReq);

			// 4. Record and Submit Commands
			uint32_t                  queueFamily = 0;
			vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlagBits::eTransient, queueFamily};
			vk::CommandPool           pool = device.createCommandPool(poolInfo);

			vk::CommandBufferAllocateInfo cmdInfo{pool, vk::CommandBufferLevel::ePrimary, 1};
			vk::CommandBuffer             cmd = device.allocateCommandBuffers(cmdInfo).front();

			cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

			if (resolvedPipeline.pipeline) {
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolvedPipeline.pipeline);
			}

			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolvedPipeline.layout, 0, querySet, nullptr);

			TerrainQueryPushConstants push{
				.width = w,
				.height = h,
				.minWorldPos = range.GetMinWorldPos(),
				.texelSize = range.GetTexelSize(),
				.planetRadius = 600000.0f
			};
			cmd.pushConstants(resolvedPipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainQueryPushConstants), &push);

			uint32_t groupX = (w + 15) / 16;
			uint32_t groupY = (h + 15) / 16;
			cmd.dispatch(groupX, groupY, 1);

			vk::BufferMemoryBarrier2 toTransfer{};
			toTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setBuffer(ssboBuffer)
				.setOffset(0)
				.setSize(bufferSize);
			vk::DependencyInfo depTransfer{};
			depTransfer.setBufferMemoryBarriers(toTransfer);
			cmd.pipelineBarrier2(depTransfer);

			vk::BufferCopy copyRegion{0, 0, bufferSize};
			cmd.copyBuffer(ssboBuffer, stagingBuffer, copyRegion);

			cmd.end();

			vk::Queue queue = engine.GetInstance() ? device.getQueue(queueFamily, 0) : vk::Queue{};
			if (queue) {
				vk::SubmitInfo submitInfo{};
				submitInfo.setCommandBuffers(cmd);
				queue.submit(submitInfo);
				queue.waitIdle();
			}

			vmaInvalidateAllocation(allocator, stagingAllocation, 0, bufferSize);

			m_cpuBuffer.resize(totalSamples);
			if (stagingResultInfo.pMappedData) {
				std::memcpy(m_cpuBuffer.data(), stagingResultInfo.pMappedData, bufferSize);
			}

			queryShader.Destroy(device);
			device.destroyDescriptorPool(queryPool);
			device.destroyDescriptorSetLayout(querySetLayout);
			device.destroyCommandPool(pool);
			vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
			vmaDestroyBuffer(allocator, ssboBuffer, ssboAllocation);

			m_range = range;
			m_hasData = true;
			return true;
		} catch (const std::exception& err) {
			spdlog::warn("GPU terrain query readback error: {}; falling back to CPU.", err.what());
			UpdateBufferCPU(range);
			return false;
		}
#endif
	}

	bool TerrainCollisionManager::UpdateBuffer(Engine* engine, const TerrainQueryRange& range) {
		if (engine) {
			return UpdateBufferGPU(*engine, range);
		} else {
			UpdateBufferCPU(range);
			return true;
		}
	}

	float TerrainCollisionManager::GetHeightAt(float worldX, float worldZ) const {
		if (!m_hasData || m_cpuBuffer.empty()) {
			return TerrainClipmap::SampleTerrain(worldX, worldZ, 0.5f).r;
		}

		glm::vec2 minPos = m_range.GetMinWorldPos();
		glm::vec2 texelSize = m_range.GetTexelSize();

		if (texelSize.x <= 0.0f || texelSize.y <= 0.0f) {
			return TerrainClipmap::SampleTerrain(worldX, worldZ, 0.5f).r;
		}

		float u = (worldX - minPos.x) / texelSize.x;
		float v = (worldZ - minPos.y) / texelSize.y;

		uint32_t w = m_range.width;
		uint32_t h = m_range.height;

		if (u < 0.0f || u > static_cast<float>(w - 1) || v < 0.0f || v > static_cast<float>(h - 1)) {
			return TerrainClipmap::SampleTerrain(worldX, worldZ, std::min(texelSize.x, texelSize.y)).r;
		}

		int x0 = static_cast<int>(std::floor(u));
		int z0 = static_cast<int>(std::floor(v));
		int x1 = std::min(x0 + 1, static_cast<int>(w) - 1);
		int z1 = std::min(z0 + 1, static_cast<int>(h) - 1);

		float tx = u - static_cast<float>(x0);
		float tz = v - static_cast<float>(z0);

		float h00 = m_cpuBuffer[z0 * w + x0].r;
		float h10 = m_cpuBuffer[z0 * w + x1].r;
		float h01 = m_cpuBuffer[z1 * w + x0].r;
		float h11 = m_cpuBuffer[z1 * w + x1].r;

		float h0 = glm::mix(h00, h10, tx);
		float h1 = glm::mix(h01, h11, tx);
		return glm::mix(h0, h1, tz);
	}

	glm::vec3 TerrainCollisionManager::GetNormalAt(float worldX, float worldZ) const {
		if (!m_hasData || m_cpuBuffer.empty()) {
			glm::vec4 sample = TerrainClipmap::SampleTerrain(worldX, worldZ, 0.5f);
			return glm::vec3(sample.g, sample.b, sample.a);
		}

		glm::vec2 minPos = m_range.GetMinWorldPos();
		glm::vec2 texelSize = m_range.GetTexelSize();

		if (texelSize.x <= 0.0f || texelSize.y <= 0.0f) {
			glm::vec4 sample = TerrainClipmap::SampleTerrain(worldX, worldZ, 0.5f);
			return glm::vec3(sample.g, sample.b, sample.a);
		}

		float u = (worldX - minPos.x) / texelSize.x;
		float v = (worldZ - minPos.y) / texelSize.y;

		uint32_t w = m_range.width;
		uint32_t h = m_range.height;

		if (u < 0.0f || u > static_cast<float>(w - 1) || v < 0.0f || v > static_cast<float>(h - 1)) {
			glm::vec4 sample = TerrainClipmap::SampleTerrain(worldX, worldZ, std::min(texelSize.x, texelSize.y));
			return glm::vec3(sample.g, sample.b, sample.a);
		}

		int x0 = static_cast<int>(std::floor(u));
		int z0 = static_cast<int>(std::floor(v));
		int x1 = std::min(x0 + 1, static_cast<int>(w) - 1);
		int z1 = std::min(z0 + 1, static_cast<int>(h) - 1);

		float tx = u - static_cast<float>(x0);
		float tz = v - static_cast<float>(z0);

		auto getNorm = [&](int x, int z) {
			const auto& s = m_cpuBuffer[z * w + x];
			return glm::vec3(s.g, s.b, s.a);
		};

		glm::vec3 n00 = getNorm(x0, z0);
		glm::vec3 n10 = getNorm(x1, z0);
		glm::vec3 n01 = getNorm(x0, z1);
		glm::vec3 n11 = getNorm(x1, z1);

		glm::vec3 n0 = glm::mix(n00, n10, tx);
		glm::vec3 n1 = glm::mix(n01, n11, tx);
		return glm::normalize(glm::mix(n0, n1, tz));
	}

	bool TerrainCollisionManager::CheckAndResolveCameraCollision(glm::vec3& cameraPosition, float minDistanceAboveTerrain) const {
		float groundHeight = GetHeightAt(cameraPosition.x, cameraPosition.z);
		float minAllowedY = groundHeight + minDistanceAboveTerrain;
		if (cameraPosition.y < minAllowedY) {
			cameraPosition.y = minAllowedY;
			return true;
		}
		return false;
	}

} // namespace brassica
