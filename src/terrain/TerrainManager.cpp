#include "terrain/TerrainManager.hpp"

#include <algorithm>
#include <cstring>

#include "spdlog/spdlog.h"

namespace {
	inline glm::vec2 octWrap(glm::vec2 v) {
		return (glm::vec2(1.0f) - glm::abs(glm::vec2(v.y, v.x))) * glm::vec2(v.x >= 0.0f ? 1.0f : -1.0f, v.y >= 0.0f ? 1.0f : -1.0f);
	}

	inline glm::vec3 octahedralUVToDirection(glm::vec2 uv) {
		glm::vec2 oct = uv * 2.0f - 1.0f;
		glm::vec3 dir = glm::vec3(oct.x, 1.0f - std::abs(oct.x) - std::abs(oct.y), oct.y);
		if (dir.y < 0.0f) {
			glm::vec2 wrapped = octWrap(glm::vec2(dir.x, dir.z));
			dir.x = wrapped.x;
			dir.z = wrapped.y;
		}
		float len = glm::length(dir);
		return len > 1e-6f ? dir / len : glm::vec3(0.0f, 1.0f, 0.0f);
	}
} // namespace

namespace brassica {

	void TerrainManager::Init(vk::Device dev, VmaAllocator alloc, uint32_t lods, float baseTexel) {
		device = dev;
		allocator = alloc;
		numLODs = lods;
		baseTexelSize = baseTexel;

		if (!device) {
			return;
		}

		// Comfortably covers the 128x128 RGBA32F (256KB) ground-height-and-normal window this is
		// actually used for -- generous rather than tight-fitted, matching the staging buffers
		// elsewhere in this codebase.
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = kReadbackStagingBytes;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		// RANDOM, not SEQUENTIAL_WRITE: the CPU *reads* this buffer (PollReadbackData's
		// vmaInvalidateAllocation + memcpy) -- SEQUENTIAL_WRITE is a hint for CPU writes, and
		// could steer VMA toward uncached or write-combined (BAR) memory, which is fine for a
		// write-only staging buffer but actively bad for one the CPU reads back from every
		// time a readback lands.
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          buf{VK_NULL_HANDLE};
		VmaAllocationInfo allocationInfo{};
		if (vmaCreateBuffer(
				allocator,
				&bufferInfo,
				&allocCreateInfo,
				&buf,
				&readbackStagingAllocation,
				&allocationInfo
			) == VK_SUCCESS) {
			readbackStagingBuffer = buf;
			readbackStagingMapped = allocationInfo.pMappedData;
		} else {
			spdlog::error("TerrainManager: failed to allocate readback staging buffer.");
		}
	}

	void TerrainManager::Cleanup() {
		if (!device) {
			return;
		}
		if (readbackStagingBuffer && readbackStagingAllocation) {
			vmaDestroyBuffer(allocator, readbackStagingBuffer, readbackStagingAllocation);
			readbackStagingBuffer = nullptr;
			readbackStagingAllocation = VK_NULL_HANDLE;
			readbackStagingMapped = nullptr;
		}
	}

	void TerrainManager::PlanReadback(glm::vec2 camPos2D) {
		hasPendingReadback = false;

		if (readbackInFlight || !readbackStagingBuffer) {
			return;
		}
		if (hasReadbackData) {
			float distFromLastTrigger = glm::distance(camPos2D, cachedReadbackCamPos);
			if (distFromLastTrigger < 16.0f) {
				return;
			}
		}

		constexpr int kReadbackWidth = 128;
		constexpr int kReadbackHeight = 128;
		int           dim = static_cast<int>(constants::Class::Terrain::MapDim); // 1024
		float         texelSize = baseTexelSize > 0.0001f ? baseTexelSize : 0.5f;
		glm::vec2     camGrid = glm::floor(camPos2D / texelSize);

		int camU = static_cast<int>(camGrid.x) % dim;
		int camV = static_cast<int>(camGrid.y) % dim;
		if (camU < 0)
			camU += dim;
		if (camV < 0)
			camV += dim;

		int minU = std::clamp(camU - kReadbackWidth / 2, 0, dim - kReadbackWidth);
		int minV = std::clamp(camV - kReadbackHeight / 2, 0, dim - kReadbackHeight);

		int g_x0 = static_cast<int>(camGrid.x) - (camU - minU);
		int g_y0 = static_cast<int>(camGrid.y) - (camV - minV);

		glm::vec2 originWorld(
			(static_cast<float>(g_x0) + 0.5f) * texelSize,
			(static_cast<float>(g_y0) + 0.5f) * texelSize
		);

		size_t requiredBytes = static_cast<size_t>(kReadbackWidth) * kReadbackHeight * sizeof(glm::vec4);
		if (requiredBytes > kReadbackStagingBytes) {
			spdlog::warn("TerrainManager: requested readback size {} bytes exceeds staging capacity.", requiredBytes);
			return;
		}

		pendingRegion = TerrainReadbackRegion{
			.arrayLayer = 0, // LOD 0
			.mipLevel = 0,
			.offset = vk::Offset2D{minU, minV},
			.extent = vk::Extent2D{static_cast<uint32_t>(kReadbackWidth), static_cast<uint32_t>(kReadbackHeight)},
			.originWorld = originWorld,
			.texelSize = texelSize,
			.camPos = camPos2D,
		};
		hasPendingReadback = true;
	}

	void TerrainManager::PollReadbackData(vk::Device dev, vk::Semaphore frameTimelineSemaphore) {
		if (!readbackInFlight || !frameTimelineSemaphore) {
			return;
		}

		uint64_t currentValue = dev.getSemaphoreCounterValue(frameTimelineSemaphore);
		if (currentValue >= pendingReadbackFrameIndex + 1) {
			readbackInFlight = false;

			cachedReadbackWidth = pendingRegion.extent.width;
			cachedReadbackHeight = pendingRegion.extent.height;
			size_t pixelCount = static_cast<size_t>(cachedReadbackWidth) * cachedReadbackHeight;
			cachedReadbackData.resize(pixelCount);
			if (readbackStagingMapped) {
				vmaInvalidateAllocation(allocator, readbackStagingAllocation, 0, pixelCount * sizeof(glm::vec4));
				std::memcpy(cachedReadbackData.data(), readbackStagingMapped, pixelCount * sizeof(glm::vec4));
			}
			cachedReadbackOriginWorld = pendingRegion.originWorld;
			cachedReadbackTexelSize = pendingRegion.texelSize;
			cachedReadbackCamPos = pendingRegion.camPos;
			hasReadbackData = true;
		}
	}

	float TerrainManager::GetInterpolatedGroundHeight(glm::vec2 worldXZ, float fallback) const {
		if (!hasReadbackData || cachedReadbackData.empty() || cachedReadbackWidth < 1 || cachedReadbackHeight < 1) {
			return fallback;
		}

		if (cachedReadbackWidth == 1 && cachedReadbackHeight == 1) {
			return cachedReadbackData[0].x;
		}

		float ts = (cachedReadbackTexelSize > 1e-5f) ? cachedReadbackTexelSize : 0.5f;
		float gx = (worldXZ.x - cachedReadbackOriginWorld.x) / ts;
		float gz = (worldXZ.y - cachedReadbackOriginWorld.y) / ts;

		int i0 = static_cast<int>(std::floor(gx));
		int j0 = static_cast<int>(std::floor(gz));

		float fx = gx - static_cast<float>(i0);
		float fz = gz - static_cast<float>(j0);

		int maxI = static_cast<int>(cachedReadbackWidth) - 1;
		int maxJ = static_cast<int>(cachedReadbackHeight) - 1;

		int i0_clamped = std::clamp(i0, 0, std::max(0, maxI - 1));
		int j0_clamped = std::clamp(j0, 0, std::max(0, maxJ - 1));

		int i1 = std::min(i0_clamped + 1, maxI);
		int j1 = std::min(j0_clamped + 1, maxJ);

		float fx_clamped = std::clamp(fx, 0.0f, 1.0f);
		float fz_clamped = std::clamp(fz, 0.0f, 1.0f);

		uint32_t w = cachedReadbackWidth;
		float    h00 = cachedReadbackData[j0_clamped * w + i0_clamped].x;
		float    h10 = cachedReadbackData[j0_clamped * w + i1].x;
		float    h01 = cachedReadbackData[j1 * w + i0_clamped].x;
		float    h11 = cachedReadbackData[j1 * w + i1].x;

		float top = glm::mix(h00, h10, fx_clamped);
		float bot = glm::mix(h01, h11, fx_clamped);
		return glm::mix(top, bot, fz_clamped);
	}

	float TerrainManager::GetInterpolatedGroundHeight(float fallback) const {
		return GetInterpolatedGroundHeight(cachedReadbackCamPos, fallback);
	}

	// Same grid/weights as GetInterpolatedGroundHeight above, just reading normal.xyz (.yzw)
	// instead of height (.x) and renormalizing afterward -- bilinear-blending four unit normals
	// doesn't generally stay unit length.
	glm::vec3 TerrainManager::GetInterpolatedGroundNormal(glm::vec2 worldXZ, glm::vec3 fallback) const {
		if (!hasReadbackData || cachedReadbackData.empty() || cachedReadbackWidth < 1 || cachedReadbackHeight < 1) {
			return fallback;
		}

		if (cachedReadbackWidth == 1 && cachedReadbackHeight == 1) {
			const glm::vec4& p = cachedReadbackData[0];
			return octahedralUVToDirection(glm::vec2(p.z, p.w));
		}

		float ts = (cachedReadbackTexelSize > 1e-5f) ? cachedReadbackTexelSize : 0.5f;
		float gx = (worldXZ.x - cachedReadbackOriginWorld.x) / ts;
		float gz = (worldXZ.y - cachedReadbackOriginWorld.y) / ts;

		int i0 = static_cast<int>(std::floor(gx));
		int j0 = static_cast<int>(std::floor(gz));

		float fx = gx - static_cast<float>(i0);
		float fz = gz - static_cast<float>(j0);

		int maxI = static_cast<int>(cachedReadbackWidth) - 1;
		int maxJ = static_cast<int>(cachedReadbackHeight) - 1;

		int i0_clamped = std::clamp(i0, 0, std::max(0, maxI - 1));
		int j0_clamped = std::clamp(j0, 0, std::max(0, maxJ - 1));

		int i1 = std::min(i0_clamped + 1, maxI);
		int j1 = std::min(j0_clamped + 1, maxJ);

		float fx_clamped = std::clamp(fx, 0.0f, 1.0f);
		float fz_clamped = std::clamp(fz, 0.0f, 1.0f);

		uint32_t  w = cachedReadbackWidth;
		auto      normalOf = [](const glm::vec4& p) { return octahedralUVToDirection(glm::vec2(p.z, p.w)); };
		glm::vec3 n00 = normalOf(cachedReadbackData[j0_clamped * w + i0_clamped]);
		glm::vec3 n10 = normalOf(cachedReadbackData[j0_clamped * w + i1]);
		glm::vec3 n01 = normalOf(cachedReadbackData[j1 * w + i0_clamped]);
		glm::vec3 n11 = normalOf(cachedReadbackData[j1 * w + i1]);

		glm::vec3 top = glm::mix(n00, n10, fx_clamped);
		glm::vec3 bot = glm::mix(n01, n11, fx_clamped);
		glm::vec3 blended = glm::mix(top, bot, fz_clamped);
		return glm::length(blended) > 1e-6f ? glm::normalize(blended) : fallback;
	}

	glm::vec3 TerrainManager::GetInterpolatedGroundNormal(glm::vec3 fallback) const {
		return GetInterpolatedGroundNormal(cachedReadbackCamPos, fallback);
	}

	float TerrainManager::GetCachedGroundHeight(float fallback) const {
		return GetInterpolatedGroundHeight(cachedReadbackCamPos, fallback);
	}

	float TerrainManager::GetCachedGroundHeight(glm::vec2 worldXZ, float fallback) const {
		return GetInterpolatedGroundHeight(worldXZ, fallback);
	}

} // namespace brassica
