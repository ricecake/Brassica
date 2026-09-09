#pragma once

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

	constexpr int LIGHT_FLAG_CASTS_SHADOW = 1;
	constexpr int LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
	constexpr int LIGHT_FLAG_CAMERA_RELATIVE = 4;
	constexpr int LIGHT_FLAG_CLOUD_EMISSIVE = 8;

	struct alignas(16) AmbientProbe {
		glm::vec4 sh_coeffs[9]; // rgb = coefficients, w = unused
	};

	/**
	 * @brief GPU-compatible light data for UBO/SSBO upload (std140 layout).
	 * Size: 64 bytes.
	 */
	struct alignas(16) LightGPU {
		glm::vec3 position{0.0f};     // offset 0,  12 bytes
		float     intensity{0.0f};    // offset 12, 4 bytes
		glm::vec3 color{1.0f};        // offset 16, 12 bytes
		int32_t   type{POINT_LIGHT};  // offset 28, 4 bytes
		glm::vec3 direction{0.0f, -1.0f, 0.0f}; // offset 32, 12 bytes
		float     inner_cutoff{0.0f}; // offset 44, 4 bytes
		float     outer_cutoff{0.0f}; // offset 48, 4 bytes
		int32_t   flags{0};           // offset 52, 4 bytes
		float     _padding[2]{0.0f, 0.0f}; // offset 56, 8 bytes
	};

	static_assert(sizeof(LightGPU) == 64, "LightGPU struct size must be 64 bytes");

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
	};

	/**
	 * @brief Complete lighting UBO data for GPU upload (std140 layout).
	 * Total size: 1744 bytes.
	 */
	struct alignas(16) LightingUbo {
		int32_t   num_lights{0};                     // offset 0, 4 bytes
		float     world_scale{1.0f};                 // offset 4, 4 bytes
		float     day_time{8.0f};                    // offset 8, 4 bytes (0-24)
		float     night_factor{0.0f};                // offset 12, 4 bytes (0-1)
		alignas(16) glm::vec3 view_pos{0.0f};        // offset 16, 12 bytes
		float     cloudShadowIntensity{1.0f};        // offset 28, 4 bytes
		alignas(16) glm::vec3 ambient_light{0.1f};   // offset 32, 12 bytes
		float     time{0.0f};                        // offset 44, 4 bytes
		alignas(16) glm::vec3 view_dir{0.0f, 0.0f, -1.0f}; // offset 48, 12 bytes
		float     cloudAltitude{1500.0f};            // offset 60, 4 bytes
		float     cloudThickness{1000.0f};           // offset 64, 4 bytes
		float     cloudDensity{0.5f};                // offset 68, 4 bytes
		float     cloudCoverage{0.5f};               // offset 72, 4 bytes
		float     cloudWarp{0.2f};                   // offset 76, 4 bytes
		float     cloudPhaseG1{0.8f};                // offset 80, 4 bytes
		float     cloudPhaseG2{-0.3f};               // offset 84, 4 bytes
		float     cloudPhaseAlpha{0.5f};             // offset 88, 4 bytes
		float     cloudPhaseIsotropic{0.0f};         // offset 92, 4 bytes
		float     cloudPowderScale{1.0f};            // offset 96, 4 bytes
		float     cloudPowderMultiplier{1.0f};       // offset 100, 4 bytes
		float     cloudPowderLocalScale{1.0f};       // offset 104, 4 bytes
		float     cloudShadowOpticalDepthMultiplier{1.0f}; // offset 108, 4 bytes
		float     cloudShadowStepMultiplier{1.0f};   // offset 112, 4 bytes
		float     cloudSunLightScale{1.0f};          // offset 116, 4 bytes
		float     cloudMoonLightScale{1.0f};         // offset 120, 4 bytes
		float     cloudBeerPowderMix{0.5f};          // offset 124, 4 bytes
		float     cloudFlowSpeed{1.0f};              // offset 128, 4 bytes
		float     cloudFlowDirection{0.0f};          // offset 132, 4 bytes
		float     cloudFlowHeightScale{1.0f};        // offset 136, 4 bytes
		float     cloudCurlStrength{0.5f};           // offset 140, 4 bytes
		float     cloudCurlFrequency{0.01f};         // offset 144, 4 bytes
		float     sunAureoleStrength{1.0f};          // offset 148, 4 bytes
		float     cirrusOpacity{0.2f};               // offset 152, 4 bytes
		float     zNear{0.1f};                       // offset 156, 4 bytes
		float     zFar{32768.0f};                    // offset 160, 4 bytes
		float     _pad_clouds3_a{0.0f};              // offset 164, 4 bytes
		float     _pad_clouds3_b{0.0f};              // offset 168, 4 bytes
		float     _pad_clouds3_c{0.0f};              // offset 172, 4 bytes
		alignas(16) glm::vec4 _pad_clouds3_vec[3]{}; // offset 176, 48 bytes
		float     _pad_cloud_shadow_mat[16]{};       // offset 224, 64 bytes
		alignas(16) glm::mat4 view{1.0f};           // offset 288, 64 bytes
		alignas(16) glm::mat4 projection{1.0f};     // offset 352, 64 bytes
		alignas(16) glm::vec3 lightningColor{1.0f}; // offset 416, 12 bytes
		float     lightningPulse{0.0f};              // offset 428, 4 bytes
		float     skyExposure{1.0f};                 // offset 432, 4 bytes
		float     starExposure{1.0f};                // offset 436, 4 bytes
		float     terrainExposure{1.0f};             // offset 440, 4 bytes
		float     _pad_exposure{0.0f};               // offset 444, 4 bytes
		alignas(16) glm::vec4 sh_coeffs[81]{};      // offset 448, 1296 bytes
	};

	static_assert(sizeof(LightingUbo) == 1744, "LightingUbo size mismatch");

	enum class LightBehaviorType { NONE, BLINK, PULSE, EASE_IN, EASE_OUT, EASE_IN_OUT, FLICKER, MORSE };

	struct LightBehavior {
		LightBehaviorType type = LightBehaviorType::NONE;
		float             period = 1.0f;
		float             amplitude = 1.0f;
		float             duty_cycle = 0.5f;
		float             flicker_intensity = 0.0f;
		std::string       message;
		float             timer = 0.0f;
		bool              loop = true;

		// Internal state
		std::vector<bool> morse_sequence;
		int               morse_index = -1;
	};

	struct Light {
		int       id = -1;
		glm::vec3 position{0.0f};
		float     intensity{0.0f};
		float     base_intensity{0.0f};
		glm::vec3 color{1.0f};
		int       type{POINT_LIGHT};
		glm::vec3 direction{0.0f, -1.0f, 0.0f};

		float azimuth = 0.0f;
		float elevation = 45.0f;

		float inner_cutoff{0.0f};
		float outer_cutoff{0.0f};

		bool casts_shadow = false;
		bool volumetric_shadow = false;
		bool camera_relative = false;
		bool cloud_emissive = false;

		int shadow_map_index = -1;

		glm::vec3 last_position{0.0f};
		glm::vec3 last_direction{0.0f, -1.0f, 0.0f};

		LightBehavior behavior;
		bool          auto_remove = false;

		LightGPU ToGPU() const {
			LightGPU gpu;
			gpu.position = position;
			gpu.intensity = intensity;
			gpu.color = color;
			gpu.type = type;
			gpu.direction = direction;
			gpu.inner_cutoff = inner_cutoff;
			gpu.outer_cutoff = outer_cutoff;
			gpu.flags = 0;
			if (casts_shadow) gpu.flags |= LIGHT_FLAG_CASTS_SHADOW;
			if (volumetric_shadow) gpu.flags |= LIGHT_FLAG_VOLUMETRIC_SHADOW;
			if (camera_relative) gpu.flags |= LIGHT_FLAG_CAMERA_RELATIVE;
			if (cloud_emissive) gpu.flags |= LIGHT_FLAG_CLOUD_EMISSIVE;
			return gpu;
		}

		void UpdateDirectionFromAngles() {
			float rad_azimuth = glm::radians(azimuth);
			float rad_elevation = glm::radians(elevation);

			glm::vec3 sun_pos;
			sun_pos.x = glm::cos(rad_elevation) * glm::sin(rad_azimuth);
			sun_pos.y = glm::sin(rad_elevation);
			sun_pos.z = glm::cos(rad_elevation) * glm::cos(rad_azimuth);

			direction = -glm::normalize(sun_pos);
		}

		static void GetAnglesFromDirection(const glm::vec3& dir, float& azimuth, float& elevation) {
			glm::vec3 d = -glm::normalize(dir);
			elevation = glm::degrees(glm::asin(glm::clamp(d.y, -1.0f, 1.0f)));
			azimuth = glm::degrees(glm::atan(d.x, d.z));
			if (azimuth < 0.0f) {
				azimuth += 360.0f;
			}
		}

		void SetBlink(float period_val, float duty_cycle_val = 0.5f) {
			behavior.type = LightBehaviorType::BLINK;
			behavior.period = period_val;
			behavior.duty_cycle = duty_cycle_val;
		}

		void SetPulse(float period_val, float amplitude_val = 1.0f) {
			behavior.type = LightBehaviorType::PULSE;
			behavior.period = period_val;
			behavior.amplitude = amplitude_val;
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

		void SetFlicker(float intensity_val = 1.0f) {
			behavior.type = LightBehaviorType::FLICKER;
			behavior.flicker_intensity = intensity_val;
		}

		void SetMorse(const std::string& msg, float unit_time = 0.2f) {
			behavior.type = LightBehaviorType::MORSE;
			behavior.message = msg;
			behavior.period = unit_time;
			behavior.morse_index = -1;
		}

		static Light Create(const glm::vec3& pos, float intens, const glm::vec3& col, bool shadows = false, float range = 0.0f) {
			Light l;
			l.position = pos;
			l.intensity = intens;
			l.base_intensity = intens;
			l.color = col;
			l.type = POINT_LIGHT;
			l.direction = glm::vec3(0.0f, -1.0f, 0.0f);
			l.inner_cutoff = 0.0f;
			l.outer_cutoff = range;
			l.casts_shadow = shadows;
			l.volumetric_shadow = false;
			l.camera_relative = false;
			l.shadow_map_index = -1;
			l.last_position = l.position;
			l.last_direction = l.direction;
			l.behavior.type = LightBehaviorType::NONE;
			return l;
		}

		static Light CreatePoint(const glm::vec3& pos, float intens, const glm::vec3& col, float range = 0.0f, bool shadows = false) {
			return Create(pos, intens, col, shadows, range);
		}

		static Light CreateDirectional(float azimuth_val, float elevation_val, float intens, const glm::vec3& col, bool shadows = false) {
			Light l = Create(glm::vec3(0.0f), intens, col, shadows);
			l.type = DIRECTIONAL_LIGHT;
			l.azimuth = azimuth_val;
			l.elevation = elevation_val;
			l.UpdateDirectionFromAngles();
			l.volumetric_shadow = true;
			return l;
		}

		static Light CreateDirectional(const glm::vec3& dir, const glm::vec3& col, float intens, bool shadows = false) {
			Light l = Create(glm::vec3(0.0f), intens, col, shadows);
			l.type = DIRECTIONAL_LIGHT;
			l.direction = glm::normalize(dir);
			GetAnglesFromDirection(l.direction, l.azimuth, l.elevation);
			l.volumetric_shadow = true;
			return l;
		}

		static Light CreateSpot(
			const glm::vec3& pos,
			const glm::vec3& dir,
			float            intens,
			const glm::vec3& col,
			float            inner_angle,
			float            outer_angle,
			bool             shadows = false
		) {
			Light l = Create(pos, intens, col, shadows);
			l.type = SPOT_LIGHT;
			l.direction = dir;
			l.inner_cutoff = glm::cos(glm::radians(inner_angle));
			l.outer_cutoff = glm::cos(glm::radians(outer_angle));
			return l;
		}

		static Light CreateEmissive(
			const glm::vec3& pos,
			float            intens,
			const glm::vec3& col,
			float            emissive_radius = 1.0f,
			bool             shadows = false
		) {
			Light l = Create(pos, intens, col, shadows);
			l.type = EMISSIVE_LIGHT;
			l.inner_cutoff = emissive_radius;
			l.outer_cutoff = 0.0f;
			return l;
		}

		static Light CreateFlash(
			const glm::vec3& pos,
			float            intens,
			const glm::vec3& col,
			float            radius = 50.0f,
			float            falloff_exp = 2.0f
		) {
			Light l = Create(pos, intens, col, false);
			l.type = FLASH_LIGHT;
			l.inner_cutoff = radius;
			l.outer_cutoff = falloff_exp;
			return l;
		}
	};

	struct alignas(16) ShadowLightData {
		glm::mat4 light_space_matrix{1.0f};
		glm::vec3 position{0.0f};
		float     padding1{0.0f};
		int32_t   shadow_map_index{-1};
		int32_t   padding2{0}, padding3{0}, padding4{0};
	};

} // namespace brassica
