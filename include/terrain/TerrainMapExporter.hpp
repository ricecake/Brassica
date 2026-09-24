#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include <glm/glm.hpp>
#include "vulkan/vulkan.hpp"

namespace brassica {

	class Engine;

	struct TerrainMapColorConfig {
		static void GetColor(
			float    height,
			float    normalX,
			float    normalY,
			float    normalZ,
			uint8_t& r,
			uint8_t& g,
			uint8_t& b,
			uint8_t& a
		) {
			glm::vec3 normal = glm::normalize(glm::vec3(normalX, normalY, normalZ));
			glm::vec3 lightDir = glm::normalize(glm::vec3(-0.6f, 1.2f, 0.8f));
			float     hillshade = std::clamp(glm::dot(normal, lightDir), 0.25f, 1.2f);

			glm::vec3 color;
			if (height < 0.0f) {
				float     depth = -height;
				float     t = std::clamp(depth / 400.0f, 0.0f, 1.0f);
				glm::vec3 shallowColor(0.12f, 0.55f, 0.85f);
				glm::vec3 deepColor(0.02f, 0.12f, 0.38f);
				glm::vec3 baseWater = glm::mix(shallowColor, deepColor, t);
				color = baseWater * std::clamp(hillshade * 0.9f + 0.1f, 0.4f, 1.1f);
			} else {
				if (height < 100.0f) {
					color = glm::mix(glm::vec3(0.18f, 0.55f, 0.22f), glm::vec3(0.35f, 0.62f, 0.22f), height / 100.0f);
				} else if (height < 350.0f) {
					color = glm::mix(
						glm::vec3(0.35f, 0.62f, 0.22f),
						glm::vec3(0.68f, 0.62f, 0.28f),
						(height - 100.0f) / 250.0f
					);
				} else if (height < 650.0f) {
					color = glm::mix(
						glm::vec3(0.68f, 0.62f, 0.28f),
						glm::vec3(0.55f, 0.42f, 0.30f),
						(height - 350.0f) / 300.0f
					);
				} else if (height < 900.0f) {
					color = glm::mix(
						glm::vec3(0.55f, 0.42f, 0.30f),
						glm::vec3(0.45f, 0.43f, 0.42f),
						(height - 650.0f) / 250.0f
					);
				} else {
					color = glm::mix(
						glm::vec3(0.45f, 0.43f, 0.42f),
						glm::vec3(0.95f, 0.95f, 0.98f),
						std::clamp((height - 900.0f) / 100.0f, 0.0f, 1.0f)
					);
				}
				color *= hillshade;
			}

			r = static_cast<uint8_t>(std::clamp(color.r * 255.0f, 0.0f, 255.0f));
			g = static_cast<uint8_t>(std::clamp(color.g * 255.0f, 0.0f, 255.0f));
			b = static_cast<uint8_t>(std::clamp(color.b * 255.0f, 0.0f, 255.0f));
			a = 255;
		}
	};

	class TerrainMapExporter {
	public:
		static bool ExportGPU(
			Engine&            engine,
			const std::string& outputPath,
			uint32_t           width = 4096,
			uint32_t           height = 2048
		);
		static bool ExportCPU(
			const std::string& outputPath,
			uint32_t           width = 1024,
			uint32_t           height = 512
		);
	};

} // namespace brassica
