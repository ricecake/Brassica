#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "terrain/TerrainMapExporter.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "Engine.hpp"
#include "Shader.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "render/PipelineLibrary.hpp"
#include "spdlog/spdlog.h"
#include "terrain/TerrainClipmap.hpp"

namespace brassica {

	struct TerrainMapPushConstants {
		uint32_t storageImageIdx{0};
		uint32_t width{4096};
		uint32_t height{2048};
		float    planetRadius{600000.0f};
	};

	bool TerrainMapExporter::ExportCPU(const std::string& outputPath, uint32_t width, uint32_t height) {
		std::vector<uint8_t> pixels(width * height * 4);

		float planetRadius = 600000.0f; // 600km
		float pi = std::numbers::pi_v<float>;

		for (uint32_t y = 0; y < height; ++y) {
			float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
			float worldZ = (0.5f - v) * (pi * planetRadius);

			for (uint32_t x = 0; x < width; ++x) {
				float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
				float worldX = (u - 0.5f) * (2.0f * pi * planetRadius);

				glm::vec4 terrainSample = TerrainClipmap::SampleTerrain(worldX, worldZ, 0.5f);
				float     h = terrainSample.r;
				float     nx = terrainSample.g;
				float     ny = terrainSample.b;
				float     nz = terrainSample.a;

				uint8_t r = 0, g = 0, b = 0, a = 255;
				TerrainMapColorConfig::GetColor(h, nx, ny, nz, r, g, b, a);

				size_t idx = (y * width + x) * 4;
				pixels[idx + 0] = r;
				pixels[idx + 1] = g;
				pixels[idx + 2] = b;
				pixels[idx + 3] = a;
			}
		}

		int result = stbi_write_png(
			outputPath.c_str(),
			static_cast<int>(width),
			static_cast<int>(height),
			4,
			pixels.data(),
			static_cast<int>(width * 4)
		);

		if (result == 0) {
			spdlog::error("Failed to write terrain map PNG to '{}'", outputPath);
			return false;
		}

		spdlog::info("Successfully exported CPU terrain map ({}x{}) to '{}'", width, height, outputPath);
		return true;
	}

	bool TerrainMapExporter::ExportGPU(Engine& engine, const std::string& outputPath, uint32_t width, uint32_t height) {
		vk::Device device = engine.GetDevice();
		if (!device) {
			spdlog::warn("Vulkan device null; falling back to CPU terrain map exporter.");
			return ExportCPU(outputPath, width, height);
		}

		try {
			VmaAllocator allocator = engine.GetAllocator();
			auto&        physicalRegistry = engine.GetPhysicalRegistry();
			auto         pipelineLibrary = engine.GetService<render::PipelineLibrary>();

			// 1. Create Storage Image
			vk::ImageCreateInfo imageInfo{};
			imageInfo.setImageType(vk::ImageType::e2D)
				.setFormat(vk::Format::eR32G32B32A32Sfloat)
				.setExtent(vk::Extent3D{width, height, 1})
				.setMipLevels(1)
				.setArrayLayers(1)
				.setSamples(vk::SampleCountFlagBits::e1)
				.setTiling(vk::ImageTiling::eOptimal)
				.setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

			VkImage       rawImage = VK_NULL_HANDLE;
			VmaAllocation imageAllocation = nullptr;
			VkImageCreateInfo rawImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
			if (vmaCreateImage(allocator, &rawImageInfo, &allocCreateInfo, &rawImage, &imageAllocation, nullptr) != VK_SUCCESS) {
				spdlog::warn("vmaCreateImage failed for terrain map; falling back to CPU.");
				return ExportCPU(outputPath, width, height);
			}

			vk::Image image(rawImage);

			vk::ImageViewCreateInfo viewInfo{};
			viewInfo.setImage(image)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(vk::Format::eR32G32B32A32Sfloat)
				.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			vk::ImageView imageView = device.createImageView(viewInfo);

			struct TerrainMapExportKey {};
			graph::ResourceDesc storageDesc{};
			storageDesc.kind = graph::ResourceDesc::Kind::Image2D;
			storageDesc.width = width;
			storageDesc.height = height;
			storageDesc.formatCode = static_cast<uint32_t>(vk::Format::eR32G32B32A32Sfloat);
			storageDesc.usageMask = static_cast<uint32_t>(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);

			physicalRegistry.RegisterImportedTexture<TerrainMapExportKey>(
				image,
				imageView,
				storageDesc,
				vk::ImageLayout::eUndefined,
				false
			);

			uint32_t storageIdx = physicalRegistry.StorageIndexOf(graph::IdOf<TerrainMapExportKey>());

			// 2. Create Host-Visible Staging Buffer
			VkDeviceSize       bufferSize = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = bufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

			VmaAllocationCreateInfo stagingAllocInfo{};
			stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          stagingBuffer = VK_NULL_HANDLE;
			VmaAllocation     stagingAllocation = nullptr;
			VmaAllocationInfo stagingResultInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingResultInfo) != VK_SUCCESS) {
				device.destroyImageView(imageView);
				vmaDestroyImage(allocator, image, imageAllocation);
				return ExportCPU(outputPath, width, height);
			}

			// 3. Compile Compute Shader & Build Pipeline
			ComputeShader mapShader;
			if (!mapShader.CompileComputeFromFile(device, "shaders/terrain_map.comp")) {
				spdlog::warn("shaders/terrain_map.comp compilation failed; falling back to CPU.");
				vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
				device.destroyImageView(imageView);
				vmaDestroyImage(allocator, image, imageAllocation);
				return ExportCPU(outputPath, width, height);
			}

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				physicalRegistry.GetFrameDescriptorSetLayout(),
				physicalRegistry.GetBindlessDescriptorSetLayout()
			};
			std::array<vk::PushConstantRange, 1> pushRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainMapPushConstants)}
			};

			render::ComputePipelineRequest pipelineReq{
				.shader = &mapShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRanges
			};
			render::ResolvedPipeline resolvedPipeline = pipelineLibrary->ResolveCached(pipelineReq);

			// 4. Record and Submit Commands
			uint32_t queueFamily = 0;
			vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlagBits::eTransient, queueFamily};
			vk::CommandPool pool = device.createCommandPool(poolInfo);
			vk::CommandBufferAllocateInfo cmdInfo{pool, vk::CommandBufferLevel::ePrimary, 1};
			vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdInfo).front();

			cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

			vk::ImageMemoryBarrier2 toGeneral{};
			toGeneral.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
				.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setDstAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
				.setOldLayout(vk::ImageLayout::eUndefined)
				.setNewLayout(vk::ImageLayout::eGeneral)
				.setImage(image)
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo depGeneral{};
			depGeneral.setImageMemoryBarriers(toGeneral);
			cmd.pipelineBarrier2(depGeneral);

			if (resolvedPipeline.pipeline) {
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolvedPipeline.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				physicalRegistry.GetFrameDescriptorSet(),
				physicalRegistry.GetBindlessDescriptorSet()
			};
			if (boundSets[0] && boundSets[1]) {
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolvedPipeline.layout, 0, boundSets, nullptr);
			}

			TerrainMapPushConstants push{storageIdx, width, height, 600000.0f};
			cmd.pushConstants(resolvedPipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainMapPushConstants), &push);

			uint32_t groupX = (width + 15) / 16;
			uint32_t groupY = (height + 15) / 16;
			cmd.dispatch(groupX, groupY, 1);

			vk::ImageMemoryBarrier2 toTransfer{};
			toTransfer.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
				.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
				.setOldLayout(vk::ImageLayout::eGeneral)
				.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setImage(image)
				.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
			vk::DependencyInfo depTransfer{};
			depTransfer.setImageMemoryBarriers(toTransfer);
			cmd.pipelineBarrier2(depTransfer);

			vk::BufferImageCopy copyRegion{};
			copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
				.setImageExtent(vk::Extent3D{width, height, 1});
			cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, stagingBuffer, copyRegion);

			cmd.end();

			vk::Queue queue = engine.GetInstance() ? device.getQueue(queueFamily, 0) : vk::Queue{};
			if (queue) {
				vk::SubmitInfo submitInfo{};
				submitInfo.setCommandBuffers(cmd);
				queue.submit(submitInfo);
				queue.waitIdle();
			}

			vmaInvalidateAllocation(allocator, stagingAllocation, 0, bufferSize);

			// Convert float pixels to 8-bit RGBA
			std::vector<uint8_t> pixels(width * height * 4);
			const float* floatData = static_cast<const float*>(stagingResultInfo.pMappedData);
			if (floatData) {
				for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
					pixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(floatData[i * 4 + 0] * 255.0f, 0.0f, 255.0f));
					pixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(floatData[i * 4 + 1] * 255.0f, 0.0f, 255.0f));
					pixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(floatData[i * 4 + 2] * 255.0f, 0.0f, 255.0f));
					pixels[i * 4 + 3] = 255;
				}
			}

			mapShader.Destroy(device);
			device.destroyCommandPool(pool);
			vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
			device.destroyImageView(imageView);
			vmaDestroyImage(allocator, image, imageAllocation);

			int stbiRes = stbi_write_png(
				outputPath.c_str(),
				static_cast<int>(width),
				static_cast<int>(height),
				4,
				pixels.data(),
				static_cast<int>(width * 4)
			);

			if (stbiRes == 0) {
				spdlog::error("Failed to write GPU terrain map PNG to '{}'", outputPath);
				return ExportCPU(outputPath, width, height);
			}

			spdlog::info("Successfully exported GPU terrain map ({}x{}) to '{}'", width, height, outputPath);
			return true;
		} catch (const std::exception& err) {
			spdlog::warn("GPU terrain map export error: {}; falling back to CPU.", err.what());
			return ExportCPU(outputPath, width, height);
		}
	}

} // namespace brassica
