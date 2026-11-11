#include "TextureDumpService.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <limits>

#include <glm/gtc/packing.hpp>
#include <spdlog/spdlog.h>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "stb_image_write.h"

namespace brassica {

	namespace {

		// The three float/half-float formats every currently-registered world-state texture
		// actually uses (confirmed against resources/ResourceTypes.hpp's WeatherBiomeImage2D/
		// TerrainLODImage2D/FixedImage2D CRTP bases) -- not a general arbitrary-vk::Format
		// decoder. Add a case here if a future dumpable texture needs one; Dump() fails loudly
		// (logged, not silently wrong output) for anything not listed.
		struct FormatInfo {
			std::uint32_t channels = 0;
			std::uint32_t bytesPerComponent = 0;
			bool          isHalfFloat = false;
		};

		bool LookupFormat(vk::Format format, FormatInfo& out) {
			switch (format) {
			case vk::Format::eR32G32B32A32Sfloat:
				out = {4, 4, false};
				return true;
			case vk::Format::eR32G32Sfloat:
				out = {2, 4, false};
				return true;
			case vk::Format::eR16G16B16A16Sfloat:
				out = {4, 2, true};
				return true;
			default:
				return false;
			}
		}

		float ReadComponent(const std::uint8_t* pixelBase, std::uint32_t component, const FormatInfo& fmt) {
			if (fmt.isHalfFloat) {
				std::uint16_t half;
				std::memcpy(&half, pixelBase + component * 2, sizeof(half));
				return glm::unpackHalf1x16(half);
			}
			float value;
			std::memcpy(&value, pixelBase + component * 4, sizeof(value));
			return value;
		}

		std::string Sanitize(const std::string& label) {
			std::string out = label;
			std::replace_if(out.begin(), out.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
			return out;
		}

		std::string Timestamp() {
			auto        now = std::chrono::system_clock::now();
			std::time_t t = std::chrono::system_clock::to_time_t(now);
			std::tm     tmBuf{};
#if defined(_WIN32)
			localtime_s(&tmBuf, &t);
#else
			localtime_r(&t, &tmBuf);
#endif
			char buf[32];
			std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmBuf);
			return std::string(buf);
		}

	} // namespace

	void TextureDumpService::Init(
		vk::Device device, vk::Queue queue, std::uint32_t queueFamily, VmaAllocator allocator,
		graph::PhysicalResourceRegistry& registry
	) {
		m_device = device;
		m_queue = queue;
		m_queueFamily = queueFamily;
		m_allocator = allocator;
		m_registry = &registry;
	}

	bool TextureDumpService::Dump(const TextureDumpEntry& entry, const std::string& outputDir) {
		if (!m_registry || !m_device || !m_queue) {
			spdlog::error("TextureDumpService::Dump('{}'): not initialized.", entry.label);
			return false;
		}

		std::shared_ptr<graph::PhysicalTexture> tex = m_registry->GetTexture(entry.id);
		if (!tex || !tex->GetImage()) {
			spdlog::error("TextureDumpService::Dump('{}'): resource is not currently provisioned.", entry.label);
			return false;
		}
		if (!tex->HasDefinedContents()) {
			spdlog::error("TextureDumpService::Dump('{}'): resource has no defined contents yet.", entry.label);
			return false;
		}

		const graph::ResourceDesc& desc = tex->GetDesc();
		FormatInfo                 fmt{};
		auto                       format = static_cast<vk::Format>(desc.formatCode);
		if (!LookupFormat(format, fmt)) {
			spdlog::error(
				"TextureDumpService::Dump('{}'): unsupported format ({}) -- add a case to LookupFormat if you need "
				"this one.",
				entry.label, static_cast<int>(format)
			);
			return false;
		}

		std::uint32_t width = desc.width;
		std::uint32_t height = desc.height;
		if (desc.layers > 1 || desc.depth > 1 || desc.mips > 1) {
			spdlog::info(
				"TextureDumpService::Dump('{}'): resource has {} layer(s)/{} depth/{} mip(s) -- dumping only array "
				"layer 0, depth slice 0, mip 0.",
				entry.label, desc.layers, desc.depth, desc.mips
			);
		}

		VkDeviceSize bytesPerPixel = static_cast<VkDeviceSize>(fmt.channels) * fmt.bytesPerComponent;
		VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * height * bytesPerPixel;

		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = bufferSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

		VmaAllocationCreateInfo stagingAllocInfo{};
		stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          stagingBuffer = VK_NULL_HANDLE;
		VmaAllocation     stagingAllocation = nullptr;
		VmaAllocationInfo stagingResultInfo{};
		if (vmaCreateBuffer(m_allocator, &bufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingResultInfo) !=
		    VK_SUCCESS) {
			spdlog::error("TextureDumpService::Dump('{}'): failed to create staging buffer.", entry.label);
			return false;
		}

		// Captured before this function's own hand-rolled transition touches tracked state --
		// same reasoning TerrainReadbackNode::Execute uses: this image's tracked (layout,
		// stage, access) describes every subresource, but only array layer 0/mip 0 is actually
		// touched here, so the other subresources must come back exactly as they were.
		vk::ImageLayout         priorLayout = tex->GetCurrentLayout();
		vk::PipelineStageFlags2 priorStage = tex->GetLastStage();
		vk::AccessFlags2        priorAccess = tex->GetLastAccess();
		vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

		vk::CommandPoolCreateInfo     poolInfo{vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily};
		vk::CommandPool               pool = m_device.createCommandPool(poolInfo);
		vk::CommandBufferAllocateInfo cmdInfo{pool, vk::CommandBufferLevel::ePrimary, 1};
		vk::CommandBuffer             cmd = m_device.allocateCommandBuffers(cmdInfo).front();

		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		vk::ImageMemoryBarrier2 toTransferSrc{};
		toTransferSrc.setSrcStageMask(priorStage ? priorStage : vk::PipelineStageFlagBits2::eTopOfPipe)
			.setSrcAccessMask(priorAccess)
			.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer)
			.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setOldLayout(priorLayout)
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setImage(tex->GetImage())
			.setSubresourceRange(range);
		vk::DependencyInfo dep1{};
		dep1.setImageMemoryBarriers(toTransferSrc);
		cmd.pipelineBarrier2(dep1);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setBufferRowLength(width)
			.setBufferImageHeight(height)
			.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
			.setImageExtent(vk::Extent3D{width, height, 1});
		cmd.copyImageToBuffer(tex->GetImage(), vk::ImageLayout::eTransferSrcOptimal, vk::Buffer(stagingBuffer), copyRegion);

		// Back to exactly what every other subresource is still actually in -- see
		// TerrainReadbackNode::Execute's identical comment for why this can't just leave the
		// touched subresource (and the tracked whole-image state along with it) at
		// eTransferSrcOptimal.
		vk::ImageMemoryBarrier2 backToPrior{};
		backToPrior.setSrcStageMask(vk::PipelineStageFlagBits2::eAllTransfer)
			.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setDstStageMask(priorStage ? priorStage : vk::PipelineStageFlagBits2::eTopOfPipe)
			.setDstAccessMask(priorAccess)
			.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setNewLayout(priorLayout)
			.setImage(tex->GetImage())
			.setSubresourceRange(range);
		vk::DependencyInfo dep2{};
		dep2.setImageMemoryBarriers(backToPrior);
		cmd.pipelineBarrier2(dep2);

		cmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(cmd);
		m_queue.submit(submitInfo);
		m_queue.waitIdle();

		tex->SetCurrentLayout(priorLayout);
		tex->SetLastStageAccess(priorStage | vk::PipelineStageFlagBits2::eAllTransfer, priorAccess | vk::AccessFlagBits2::eTransferRead);

		m_device.destroyCommandPool(pool);

		vmaInvalidateAllocation(m_allocator, stagingAllocation, 0, bufferSize);
		const auto* bytes = static_cast<const std::uint8_t*>(stagingResultInfo.pMappedData);
		if (!bytes) {
			spdlog::error("TextureDumpService::Dump('{}'): staging buffer was never mapped.", entry.label);
			vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
			return false;
		}

		// Per-channel min/max over the whole image, then linearly remap to [0,255] -- these are
		// arbitrary simulation values (temperature, biome ids, encoded heights...), not a
		// pre-normalized color, so a fixed clamp range (TerrainMapExporter's own simpler
		// approach, tuned for its own always-[0,1] output) would often just render solid black
		// or white. A degenerate all-equal channel maps to a flat mid-grey rather than NaN/div
		// by zero.
		std::size_t                       pixelCount = static_cast<std::size_t>(width) * height;
		std::array<float, 4>              channelMin{};
		std::array<float, 4>              channelMax{};
		channelMin.fill(std::numeric_limits<float>::max());
		channelMax.fill(std::numeric_limits<float>::lowest());
		for (std::size_t p = 0; p < pixelCount; ++p) {
			const std::uint8_t* pixelBase = bytes + p * bytesPerPixel;
			for (std::uint32_t c = 0; c < fmt.channels; ++c) {
				float v = ReadComponent(pixelBase, c, fmt);
				channelMin[c] = std::min(channelMin[c], v);
				channelMax[c] = std::max(channelMax[c], v);
			}
		}

		std::vector<std::uint8_t> pixels(pixelCount * fmt.channels);
		for (std::size_t p = 0; p < pixelCount; ++p) {
			const std::uint8_t* pixelBase = bytes + p * bytesPerPixel;
			for (std::uint32_t c = 0; c < fmt.channels; ++c) {
				float v = ReadComponent(pixelBase, c, fmt);
				float range = channelMax[c] - channelMin[c];
				float normalized = (range > 1e-12f) ? (v - channelMin[c]) / range : 0.5f;
				pixels[p * fmt.channels + c] = static_cast<std::uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
			}
		}

		vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);

		std::error_code ec;
		std::filesystem::create_directories(outputDir, ec);
		std::string path = outputDir + "/" + Sanitize(entry.label) + "_" + Timestamp() + ".png";

		int stbiRes = stbi_write_png(
			path.c_str(), static_cast<int>(width), static_cast<int>(height), static_cast<int>(fmt.channels), pixels.data(),
			static_cast<int>(width * fmt.channels)
		);
		if (stbiRes == 0) {
			spdlog::error("TextureDumpService::Dump('{}'): failed to write PNG to '{}'.", entry.label, path);
			return false;
		}

		spdlog::info("TextureDumpService::Dump('{}'): wrote {}x{} to '{}'.", entry.label, width, height, path);
		return true;
	}

} // namespace brassica
