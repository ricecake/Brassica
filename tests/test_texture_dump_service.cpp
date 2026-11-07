#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cstring>
#include <filesystem>
#include <vector>

#include "doctest/doctest.h"

#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "resources/ResourceTypes.hpp"
#include "stb_image.h"
#include "TextureDumpService.hpp"

using namespace brassica;
using namespace brassica::graph;

namespace {

	// Test-only resource key, same shape as every real FixedImage2D-derived key (e.g.
	// TerrainShadowMap) -- exists only so RegisterImportedTexture<K>()/RegisterDumpable<K>()
	// have a real compile-time key to work with.
	struct TestDumpTexture: FixedImage2D<TestDumpTexture> {
		static constexpr std::uint32_t       kWidth = 4;
		static constexpr std::uint32_t       kHeight = 4;
		static constexpr vk::Format          kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eTransferSrc |
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	};

} // namespace

// Exercises TextureDumpService::Dump's real GPU path end to end -- the one part build success
// alone can't verify: the hand-rolled transfer-src/transfer-dst barrier dance, the actual
// copyImageToBuffer, and the float-to-PNG conversion. Real device via MinimalDevice (plain
// image + transfer work, no mesh shader/ray query needed).
TEST_CASE("TextureDumpService dumps a real GPU texture to a PNG with the right relative brightness") {
	testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device   vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();

	constexpr std::uint32_t kW = 4;
	constexpr std::uint32_t kH = 4;

	// Known data: a horizontal gradient (column 0 dimmest, column 3 brightest), replicated
	// across all 4 channels and every row.
	std::vector<float> src(static_cast<std::size_t>(kW) * kH * 4);
	for (std::uint32_t y = 0; y < kH; ++y) {
		for (std::uint32_t x = 0; x < kW; ++x) {
			float       v = static_cast<float>(x) / static_cast<float>(kW - 1);
			std::size_t base = (static_cast<std::size_t>(y) * kW + x) * 4;
			src[base + 0] = v;
			src[base + 1] = v;
			src[base + 2] = v;
			src[base + 3] = 1.0f;
		}
	}

	vk::ImageCreateInfo imageInfo{};
	imageInfo.setImageType(vk::ImageType::e2D)
		.setFormat(vk::Format::eR32G32B32A32Sfloat)
		.setExtent(vk::Extent3D{kW, kH, 1})
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setTiling(vk::ImageTiling::eOptimal)
		.setUsage(
			vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
		);

	VmaAllocationCreateInfo imageAllocInfo{};
	imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	VkImage           rawImage = VK_NULL_HANDLE;
	VmaAllocation     imageAllocation = nullptr;
	VkImageCreateInfo rawImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
	REQUIRE(vmaCreateImage(allocator, &rawImageInfo, &imageAllocInfo, &rawImage, &imageAllocation, nullptr) == VK_SUCCESS);
	vk::Image image(rawImage);

	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.setImage(image)
		.setViewType(vk::ImageViewType::e2D)
		.setFormat(vk::Format::eR32G32B32A32Sfloat)
		.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	vk::ImageView view = vkDevice.createImageView(viewInfo);

	VkDeviceSize       bufferSize = static_cast<VkDeviceSize>(src.size()) * sizeof(float);
	VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo stagingAllocInfo{};
	stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VkBuffer          stagingBuffer = VK_NULL_HANDLE;
	VmaAllocation     stagingAllocation = nullptr;
	VmaAllocationInfo stagingResultInfo{};
	REQUIRE(
		vmaCreateBuffer(allocator, &bufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingResultInfo) ==
		VK_SUCCESS
	);
	std::memcpy(stagingResultInfo.pMappedData, src.data(), bufferSize);

	// Upload: transition to transferDst, copy, leave it there -- TextureDumpService::Dump reads
	// whatever "prior" layout it's told the image is really in, so resting at
	// eTransferDstOptimal here is just as valid a starting point as any other real layout would
	// be, and skips a pointless extra transition this test doesn't otherwise need.
	vk::CommandPool pool =
		vkDevice.createCommandPool(vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, device.GetQueueFamily()});
	vk::CommandBuffer cmd =
		vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}).front();
	cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	vk::ImageMemoryBarrier2 toDst{};
	toDst.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
		.setDstStageMask(vk::PipelineStageFlagBits2::eAllTransfer)
		.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
		.setOldLayout(vk::ImageLayout::eUndefined)
		.setNewLayout(vk::ImageLayout::eTransferDstOptimal)
		.setImage(image)
		.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
	vk::DependencyInfo dep{};
	dep.setImageMemoryBarriers(toDst);
	cmd.pipelineBarrier2(dep);

	vk::BufferImageCopy copyRegion{};
	copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
		.setImageExtent(vk::Extent3D{kW, kH, 1});
	cmd.copyBufferToImage(vk::Buffer(stagingBuffer), image, vk::ImageLayout::eTransferDstOptimal, copyRegion);

	cmd.end();
	vk::SubmitInfo submitInfo{};
	submitInfo.setCommandBuffers(cmd);
	device.GetQueue().submit(submitInfo);
	device.GetQueue().waitIdle();
	vkDevice.destroyCommandPool(pool);
	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

	std::filesystem::path outDir = std::filesystem::temp_directory_path() / "brassica_test_texture_dump";
	std::error_code       ec;
	std::filesystem::remove_all(outDir, ec);

	{
		PhysicalResourceRegistry registry(vkDevice, allocator);

		ResourceDesc desc{};
		desc.kind = ResourceDesc::Kind::Image2D;
		desc.width = kW;
		desc.height = kH;
		desc.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat);
		desc.usageMask = static_cast<std::uint32_t>(imageInfo.usage);

		registry.RegisterImportedTexture<TestDumpTexture>(image, view, desc, vk::ImageLayout::eTransferDstOptimal, true);

		TextureDumpService service;
		service.Init(vkDevice, device.GetQueue(), device.GetQueueFamily(), allocator, registry);
		service.RegisterDumpable<TestDumpTexture>("Test Gradient");
		REQUIRE(service.Entries().size() == 1);

		bool ok = service.Dump(service.Entries()[0], outDir.string());
		REQUIRE(ok);

		// Dump() doesn't hand back the exact (timestamped) filename it chose -- scan for it.
		std::filesystem::path foundPath;
		for (const auto& dirEntry : std::filesystem::directory_iterator(outDir)) {
			if (dirEntry.path().extension() == ".png") {
				foundPath = dirEntry.path();
				break;
			}
		}
		REQUIRE(!foundPath.empty());

		int            w = 0;
		int            h = 0;
		int            channels = 0;
		unsigned char* pixels = stbi_load(foundPath.string().c_str(), &w, &h, &channels, 0);
		REQUIRE(pixels != nullptr);
		CHECK(w == static_cast<int>(kW));
		CHECK(h == static_cast<int>(kH));
		CHECK(channels == 4);

		// Dump() normalizes each channel independently by its own min/max across the image, so
		// absolute values don't survive -- only relative brightness does. Column 0 (known
		// dimmest) must decode darker than column 3 (known brightest), and since the known data
		// spans the full [0,1] range, the normalized extremes should land at/near 0 and 255.
		int leftR = pixels[(0 * kW + 0) * static_cast<std::uint32_t>(channels) + 0];
		int rightR = pixels[(0 * kW + 3) * static_cast<std::uint32_t>(channels) + 0];
		CHECK(leftR < rightR);
		CHECK(leftR <= 5);
		CHECK(rightR >= 250);

		stbi_image_free(pixels);
		registry.Reset();
	}

	std::filesystem::remove_all(outDir, ec);
	vkDevice.destroyImageView(view);
	vmaDestroyImage(allocator, image, imageAllocation);

	CHECK(device.GetValidationErrorCount() == 0);
}
