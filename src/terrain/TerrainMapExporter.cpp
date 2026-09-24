#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "terrain/TerrainMapExporter.hpp"

#include <cmath>
#include <numbers>
#include <vector>

#include "Engine.hpp"
#include "spdlog/spdlog.h"
#include "terrain/TerrainClipmap.hpp"

namespace brassica {

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
			return ExportCPU(outputPath, width, height);
		} catch (const std::exception& err) {
			spdlog::warn("GPU terrain map export error: {}; falling back to CPU.", err.what());
			return ExportCPU(outputPath, width, height);
		}
	}

} // namespace brassica
