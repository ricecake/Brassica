#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	enum LightType {
		POINT_LIGHT = 0,
		DIRECTIONAL_LIGHT = 1,
		SPOT_LIGHT = 2,
		EMISSIVE_LIGHT = 3, // Glowing object - point light with emissive surface
		FLASH_LIGHT = 4     // Explosion/flash - very bright, rapid falloff
	};

	constexpr int32_t LIGHT_FLAG_CASTS_SHADOW = 1;
	constexpr int32_t LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
	constexpr int32_t LIGHT_FLAG_CAMERA_RELATIVE = 4;
	constexpr int32_t LIGHT_FLAG_CLOUD_EMISSIVE = 8;

	struct alignas(16) LightGPU {
		glm::vec3 position{0.0f};               // offset 0, 12 bytes
		float     intensity{0.0f};              // offset 12, 4 bytes
		glm::vec3 color{1.0f};                  // offset 16, 12 bytes
		int32_t   type{POINT_LIGHT};            // offset 28, 4 bytes
		glm::vec3 direction{0.0f, -1.0f, 0.0f}; // offset 32, 12 bytes
		float     innerCutoff{0.0f};            // offset 44, 4 bytes
		float     outerCutoff{0.0f};            // offset 48, 4 bytes
		int32_t   flags{0};                     // offset 52, 4 bytes
		float     _padding[2]{0.0f, 0.0f};      // offset 56, 8 bytes
	}; // Total: 64 bytes

	static_assert(sizeof(LightGPU) == 64, "LightGPU size must be 64 bytes");

	constexpr uint32_t MAX_LIGHTS = 1024;

	struct alignas(16) LightsSSBOData {
		uint32_t count{0};
		uint32_t padding[3]{0, 0, 0};
		LightGPU lights[MAX_LIGHTS];
	};

	struct alignas(16) ClusterGPU {
		uint32_t count{0};
		uint32_t lightIndices[64]{0};
		uint32_t padding[3]{0, 0, 0};
	}; // Total: 272 bytes

	static_assert(sizeof(ClusterGPU) == 272, "ClusterGPU size must be 272 bytes");

	constexpr uint32_t TOTAL_CLUSTERS = 16 * 9 * 24 + 1; // 3457 clusters

	enum class LightBehaviorType { NONE, BLINK, PULSE, EASE_IN, EASE_OUT, EASE_IN_OUT, FLICKER, MORSE };

	struct LightBehavior {
		LightBehaviorType type = LightBehaviorType::NONE;
		float             period = 1.0f;
		float             amplitude = 1.0f;
		float             dutyCycle = 0.5f;
		float             flickerIntensity = 0.0f; // 0-5
		std::string       message;
		float             timer = 0.0f;
		bool              loop = true;

		// Internal state
		std::vector<bool> morseSequence;
		int               morseIndex = -1;
	};

	struct Light {
		int       id = -1;
		glm::vec3 position{0.0f};
		float     intensity = 1.0f;
		float     baseIntensity = 1.0f; // Original intensity before behaviors
		glm::vec3 color{1.0f};
		int       type = POINT_LIGHT;
		glm::vec3 direction{0.0f, -1.0f, 0.0f};

		// For directional lights, angles in degrees
		float azimuth = 0.0f;    // 0 is North (+Z), 90 is East (+X)
		float elevation = 45.0f; // 0 is horizon, 90 is zenith (+Y)

		float innerCutoff = 0.0f;
		float outerCutoff = 0.0f;

		// Flags
		bool castsShadow = false;
		bool volumetricShadow = false;
		bool cameraRelative = false;
		bool cloudEmissive = false;

		int shadowMapIndex = -1;

		glm::vec3 lastPosition{0.0f};
		glm::vec3 lastDirection{0.0f, -1.0f, 0.0f};

		LightBehavior behavior;
		bool          autoRemove = false;

		LightGPU ToGPU() const {
			LightGPU gpu;
			gpu.position = position;
			gpu.intensity = intensity;
			gpu.color = color;
			gpu.type = type;
			gpu.direction = direction;
			gpu.innerCutoff = innerCutoff;
			gpu.outerCutoff = outerCutoff;
			gpu.flags = 0;
			if (castsShadow)
				gpu.flags |= LIGHT_FLAG_CASTS_SHADOW;
			if (volumetricShadow)
				gpu.flags |= LIGHT_FLAG_VOLUMETRIC_SHADOW;
			if (cameraRelative)
				gpu.flags |= LIGHT_FLAG_CAMERA_RELATIVE;
			if (cloudEmissive)
				gpu.flags |= LIGHT_FLAG_CLOUD_EMISSIVE;
			return gpu;
		}

		void UpdateDirectionFromAngles() {
			float radAzimuth = glm::radians(azimuth);
			float radElevation = glm::radians(elevation);

			glm::vec3 sunPos;
			sunPos.x = glm::cos(radElevation) * glm::sin(radAzimuth);
			sunPos.y = glm::sin(radElevation);
			sunPos.z = glm::cos(radElevation) * glm::cos(radAzimuth);

			direction = -glm::normalize(sunPos);
		}

		static void GetAnglesFromDirection(const glm::vec3& dir, float& azimuth, float& elevation) {
			glm::vec3 d = -glm::normalize(dir);
			elevation = glm::degrees(glm::asin(glm::clamp(d.y, -1.0f, 1.0f)));
			azimuth = glm::degrees(glm::atan(d.x, d.z));
			if (azimuth < 0.0f) {
				azimuth += 360.0f;
			}
		}

		void SetBlink(float period, float dutyCycle = 0.5f) {
			behavior.type = LightBehaviorType::BLINK;
			behavior.period = period;
			behavior.dutyCycle = dutyCycle;
		}

		void SetPulse(float period, float amplitude = 1.0f) {
			behavior.type = LightBehaviorType::PULSE;
			behavior.period = period;
			behavior.amplitude = amplitude;
		}

		void SetEaseIn(float duration) {
			behavior.type = LightBehaviorType::EASE_IN;
			behavior.period = duration;
			behavior.timer = 0.0f;
		}

		void SetEaseOut(float duration) {
			behavior.type = LightBehaviorType::EASE_OUT;
			behavior.period = duration;
			behavior.timer = 0.0f;
			behavior.loop = false;
		}

		void SetEaseInOut(float duration) {
			behavior.type = LightBehaviorType::EASE_IN_OUT;
			behavior.period = duration;
			behavior.timer = 0.0f;
		}

		void SetFlicker(float intensity = 1.0f) {
			behavior.type = LightBehaviorType::FLICKER;
			behavior.flickerIntensity = intensity;
		}

		void SetMorse(const std::string& msg, float unitTime = 0.2f) {
			behavior.type = LightBehaviorType::MORSE;
			behavior.message = msg;
			behavior.period = unitTime;
			behavior.morseIndex = -1; // Trigger sequence generation
		}

		static Light
		Create(const glm::vec3& pos, float intens, const glm::vec3& col, bool shadows = false, float range = 0.0f) {
			Light l;
			l.position = pos;
			l.intensity = intens;
			l.baseIntensity = intens;
			l.color = col;
			l.type = POINT_LIGHT;
			l.direction = glm::vec3(0.0f, -1.0f, 0.0f);
			l.innerCutoff = 0.0f;
			l.outerCutoff = range;
			l.castsShadow = shadows;
			l.volumetricShadow = false;
			l.cameraRelative = false;
			l.shadowMapIndex = -1;
			l.lastPosition = l.position;
			l.lastDirection = l.direction;
			l.behavior.type = LightBehaviorType::NONE;
			return l;
		}

		static Light CreatePoint(
			const glm::vec3& pos,
			float            intens,
			const glm::vec3& col,
			float            range = 0.0f,
			bool             shadows = false
		) {
			return Create(pos, intens, col, shadows, range);
		}

		static Light
		CreateDirectional(float azimuth, float elevation, float intens, const glm::vec3& col, bool shadows = false) {
			Light l = Create(glm::vec3(0.0f), intens, col, shadows);
			l.type = DIRECTIONAL_LIGHT;
			l.azimuth = azimuth;
			l.elevation = elevation;
			l.UpdateDirectionFromAngles();
			l.volumetricShadow = true;
			return l;
		}

		static Light CreateDirectional(const glm::vec3& dir, const glm::vec3& col, float intens, bool shadows = false) {
			Light l = Create(glm::vec3(0.0f), intens, col, shadows);
			l.type = DIRECTIONAL_LIGHT;
			l.direction = glm::normalize(dir);
			GetAnglesFromDirection(l.direction, l.azimuth, l.elevation);
			l.volumetricShadow = true;
			return l;
		}

		static Light CreateSpot(
			const glm::vec3& pos,
			const glm::vec3& dir,
			float            intens,
			const glm::vec3& col,
			float            innerAngle,
			float            outerAngle,
			bool             shadows = false
		) {
			Light l = Create(pos, intens, col, shadows);
			l.type = SPOT_LIGHT;
			l.direction = dir;
			l.innerCutoff = glm::cos(glm::radians(innerAngle));
			l.outerCutoff = glm::cos(glm::radians(outerAngle));
			return l;
		}

		static Light CreateEmissive(
			const glm::vec3& pos,
			float            intens,
			const glm::vec3& col,
			float            emissiveRadius = 1.0f,
			bool             shadows = false
		) {
			Light l = Create(pos, intens, col, shadows);
			l.type = EMISSIVE_LIGHT;
			l.innerCutoff = emissiveRadius;
			l.outerCutoff = 0.0f;
			return l;
		}

		static Light CreateFlash(
			const glm::vec3& pos,
			float            intens,
			const glm::vec3& col,
			float            radius = 50.0f,
			float            falloffExp = 2.0f
		) {
			Light l = Create(pos, intens, col, false);
			l.type = FLASH_LIGHT;
			l.innerCutoff = radius;
			l.outerCutoff = falloffExp;
			return l;
		}
	};

} // namespace brassica
