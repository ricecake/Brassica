#pragma once

#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"

#include "types/Light.hpp"

namespace brassica {

	struct DayNightCycle {
		bool      enabled = true;
		float     time = 8.0f;       // 0.0 - 24.0 (12.0 is noon)
		float     speed = 0.0125f;   // Rate of time passage
		bool      paused = false;
		float     night_factor = 0.0f; // 0.0 (day) to 1.0 (night)
		float     moon_offset = 6.0f; // Hours offset from sun
		float     moon_azimuth = 70.0f;

		float     moon_phase_days = 0.0f;
		float     lunar_albedo = 0.08f;
		float     lunar_month = 2.0f;
		glm::vec3 moon_tint = glm::vec3(0.95f, 0.93f, 0.88f);
	};

	class LightManager {
	public:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		LightManager();
		~LightManager();

		int                       AddLight(const Light& light);
		void                      RemoveLight(int id);
		Light*                    GetLight(int id);
		std::vector<Light>&       GetLights();
		const std::vector<Light>& GetLights() const;

		void Update(float deltaTime);

		glm::vec3 GetAmbientLight() const { return ambientLight; }
		void      SetAmbientLight(const glm::vec3& ambient) { ambientLight = ambient; }

		// SH Probe Tuning & Exposure Controls
		float GetProbeScaling() const { return probeScaling; }
		void  SetProbeScaling(float scaling) { probeScaling = scaling; }
		float GetSkyExposure() const { return skyExposure; }
		void  SetSkyExposure(float exp) { skyExposure = exp; }
		float GetStarExposure() const { return starExposure; }
		void  SetStarExposure(float exp) { starExposure = exp; }
		float GetTerrainExposure() const { return terrainExposure; }
		void  SetTerrainExposure(float exp) { terrainExposure = exp; }
		float GetProbeConvergenceSpeed() const { return probeConvergenceSpeed; }
		void  SetProbeConvergenceSpeed(float speed) { probeConvergenceSpeed = speed; }
		int   GetProbeRayCountMultiplier() const { return probeRayCountMultiplier; }
		void  SetProbeRayCountMultiplier(int multiplier) { probeRayCountMultiplier = multiplier; }

		DayNightCycle&       GetDayNightCycle() { return cycle; }
		const DayNightCycle& GetDayNightCycle() const { return cycle; }

		std::vector<Light*> GetShadowCastingLights();
		int                 GetShadowCastingLightCount() const;

		// GPU Buffer Management
		bool InitGpuResources(vk::Device device, VmaAllocator allocator);
		void CleanupGpuResources();
		void UpdateGpuBuffers(
			uint32_t         activeFrame,
			const glm::mat4& view,
			const glm::mat4& projection,
			const glm::vec3& cameraPos,
			const glm::vec3& cameraDir,
			float            zNear = 0.1f,
			float            zFar = 32768.0f,
			float            time = 0.0f
		);

		vk::Buffer GetLightingUboBuffer(uint32_t activeFrame) const {
			return lightingUboBuffers[activeFrame % FRAME_OVERLAP];
		}
		vk::Buffer GetLightsSsboBuffer(uint32_t activeFrame) const {
			return lightsSsboBuffers[activeFrame % FRAME_OVERLAP];
		}

		const LightingUbo&    GetLightingUbo() const { return lightingUbo; }
		LightingUbo&          GetLightingUbo() { return lightingUbo; }
		const LightsSSBOData& GetLightsSSBOData() const { return lightsSsboData; }

	private:
		std::vector<Light> _lights{
			Light::CreateDirectional(0.0f, 45.0f, 10.0f, {1.0f, 1.0f, 1.0f}, true),
			Light::CreateDirectional(180.0f, -45.0f, 1.0f, {0.8f, 0.9f, 1.0f}, true)
		};
		glm::vec3     ambientLight{0.2f, 0.2f, 0.2f};
		DayNightCycle cycle;
		int           nextLightId = 1;

		float probeScaling = 1.0f;
		float skyExposure = 1.0f;
		float starExposure = 1.0f;
		float terrainExposure = 1.0f;
		float probeConvergenceSpeed = 0.50f;
		int   probeRayCountMultiplier = 1;

		LightingUbo    lightingUbo{};
		LightsSSBOData lightsSsboData{};

		vk::Device   device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};

		VkBuffer      lightingUboBuffers[FRAME_OVERLAP]{VK_NULL_HANDLE, VK_NULL_HANDLE};
		VmaAllocation lightingUboAllocations[FRAME_OVERLAP]{VK_NULL_HANDLE, VK_NULL_HANDLE};
		void*         lightingUboMapped[FRAME_OVERLAP]{nullptr, nullptr};

		VkBuffer      lightsSsboBuffers[FRAME_OVERLAP]{VK_NULL_HANDLE, VK_NULL_HANDLE};
		VmaAllocation lightsSsboAllocations[FRAME_OVERLAP]{VK_NULL_HANDLE, VK_NULL_HANDLE};
		void*         lightsSsboMapped[FRAME_OVERLAP]{nullptr, nullptr};
	};

} // namespace brassica
