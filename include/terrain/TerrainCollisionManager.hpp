#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

#include "terrain/TerrainClipmap.hpp"

namespace brassica {

	class Engine;

	struct TerrainQueryRange {
		glm::vec2 center{0.0f, 0.0f};       // Center world coordinate (X, Z)
		glm::vec2 extent{1000.0f, 1000.0f}; // Size in X and Z (world units)
		uint32_t  width{256};               // Grid samples along X
		uint32_t  height{256};              // Grid samples along Z

		[[nodiscard]] glm::vec2 GetMinWorldPos() const {
			return center - extent * 0.5f;
		}

		[[nodiscard]] glm::vec2 GetMaxWorldPos() const {
			return center + extent * 0.5f;
		}

		[[nodiscard]] glm::vec2 GetTexelSize() const {
			float sx = (width > 1) ? (extent.x / static_cast<float>(width - 1)) : extent.x;
			float sz = (height > 1) ? (extent.y / static_cast<float>(height - 1)) : extent.y;
			return glm::vec2(sx, sz);
		}
	};

	class TerrainCollisionManager {
	public:
		TerrainCollisionManager() = default;
		~TerrainCollisionManager() = default;

		// Fills CPU buffer using CPU TerrainClipmap::SampleTerrain (for fallback/unit tests)
		void UpdateBufferCPU(const TerrainQueryRange& range);

		// Fills CPU buffer using GPU compute shader readback (when Engine & Vulkan device active)
		bool UpdateBufferGPU(Engine& engine, const TerrainQueryRange& range);

		// Dispatches GPU readback if possible, or falls back to CPU update
		bool UpdateBuffer(Engine* engine, const TerrainQueryRange& range);

		// CPU Collision Queries
		[[nodiscard]] float     GetHeightAt(float worldX, float worldZ) const;
		[[nodiscard]] glm::vec3 GetNormalAt(float worldX, float worldZ) const;

		// Returns true if camera position was adjusted to prevent moving below terrain
		bool CheckAndResolveCameraCollision(glm::vec3& cameraPosition, float minDistanceAboveTerrain = 2.0f) const;

		// Accessors
		[[nodiscard]] const TerrainQueryRange&     GetRange() const { return m_range; }
		[[nodiscard]] const std::vector<glm::vec4>& GetCPUBuffer() const { return m_cpuBuffer; }
		[[nodiscard]] bool                         HasData() const { return m_hasData && !m_cpuBuffer.empty(); }

	private:
		TerrainQueryRange      m_range{};
		std::vector<glm::vec4> m_cpuBuffer{}; // xyzw = height, nx, ny, nz
		bool                   m_hasData{false};
	};

} // namespace brassica
