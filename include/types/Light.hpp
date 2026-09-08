#pragma once

#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace brassica {

	enum class LightType : int32_t {
		Point = 0,
		Directional = 1,
		Spot = 2,
		Emissive = 3,
		Flash = 4
	};

	constexpr int32_t LIGHT_FLAG_CASTS_SHADOW = 1;
	constexpr int32_t LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
	constexpr int32_t LIGHT_FLAG_CAMERA_RELATIVE = 4;
	constexpr int32_t LIGHT_FLAG_CLOUD_EMISSIVE = 8;

	struct alignas(16) LightGPU {
		alignas(16) glm::vec3 position{0.0f};
		alignas(4) float intensity{1.0f};
		alignas(16) glm::vec3 color{1.0f};
		alignas(4) int32_t type{static_cast<int32_t>(LightType::Point)};
		alignas(16) glm::vec3 direction{0.0f, -1.0f, 0.0f};
		alignas(4) float innerCutoff{0.0f};
		alignas(4) float outerCutoff{0.0f};
		alignas(4) int32_t flags{0};
		alignas(4) float padding[2]{0.0f, 0.0f};
	};

	struct DirectionalLight {
		glm::vec3 direction{0.0f, -1.0f, 0.0f};
		glm::vec3 color{1.0f, 0.95f, 0.9f};
		float     intensity{2.5f};

		float azimuth{135.0f};  // degrees, 0 is North (+Z), 90 is East (+X)
		float elevation{45.0f}; // degrees, 0 is horizon, 90 is zenith (+Y)

		glm::vec3 GetLightDir() const { return -glm::normalize(direction); }
		glm::vec3 GetRadiance() const { return color * intensity; }

		void UpdateDirectionFromAngles() {
			float radAzimuth = glm::radians(azimuth);
			float radElevation = glm::radians(elevation);

			glm::vec3 pos;
			pos.x = std::cos(radElevation) * std::sin(radAzimuth);
			pos.y = std::sin(radElevation);
			pos.z = std::cos(radElevation) * std::cos(radAzimuth);

			direction = -glm::normalize(pos);
		}

		static void GetAnglesFromDirection(const glm::vec3& dir, float& outAzimuth, float& outElevation) {
			glm::vec3 d = -glm::normalize(dir);
			outElevation = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
			outAzimuth = glm::degrees(std::atan2(d.x, d.z));
			if (outAzimuth < 0.0f) {
				outAzimuth += 360.0f;
			}
		}

		static DirectionalLight
		CreateFromAngles(float az, float el, float intens = 2.5f, const glm::vec3& col = glm::vec3(1.0f, 0.95f, 0.9f)) {
			DirectionalLight l;
			l.azimuth = az;
			l.elevation = el;
			l.intensity = intens;
			l.color = col;
			l.UpdateDirectionFromAngles();
			return l;
		}

		static DirectionalLight
		CreateFromDir(const glm::vec3& dir, float intens = 2.5f, const glm::vec3& col = glm::vec3(1.0f, 0.95f, 0.9f)) {
			DirectionalLight l;
			l.direction = glm::normalize(dir);
			l.color = col;
			l.intensity = intens;
			GetAnglesFromDirection(l.direction, l.azimuth, l.elevation);
			return l;
		}

		LightGPU ToGPU() const {
			LightGPU gpu;
			gpu.position = glm::vec3(0.0f);
			gpu.intensity = intensity;
			gpu.color = color;
			gpu.type = static_cast<int32_t>(LightType::Directional);
			gpu.direction = direction;
			gpu.flags = LIGHT_FLAG_VOLUMETRIC_SHADOW;
			return gpu;
		}
	};

	struct GlobalLightingData {
		DirectionalLight sun{DirectionalLight::CreateFromAngles(135.0f, 45.0f, 3.0f, glm::vec3(1.0f, 0.98f, 0.92f))};
		DirectionalLight moon{DirectionalLight::CreateFromAngles(315.0f, -45.0f, 0.2f, glm::vec3(0.5f, 0.6f, 0.8f))};
		glm::vec3        ambientLight{0.05f, 0.05f, 0.08f};
		float            time{0.0f};
		float            dayTime{12.0f}; // 0.0 - 24.0 hours
		float            skyExposure{1.0f};
		float            starExposure{1.0f};
		float            sunAureoleStrength{0.5f};
		float            cirrusOpacity{0.3f};

		void UpdateSunMoonFromTime(float timeInHours) {
			dayTime = timeInHours;
			float dayFraction = timeInHours / 24.0f;
			float sunAngle = (dayFraction - 0.25f) * 2.0f * glm::pi<float>();

			sun.azimuth = 90.0f + 180.0f * dayFraction;
			sun.elevation = glm::degrees(std::sin(sunAngle)) * 75.0f / 90.0f;
			sun.UpdateDirectionFromAngles();

			moon.azimuth = sun.azimuth + 180.0f;
			if (moon.azimuth >= 360.0f)
				moon.azimuth -= 360.0f;
			moon.elevation = -sun.elevation;
			moon.UpdateDirectionFromAngles();
		}
	};

} // namespace brassica
