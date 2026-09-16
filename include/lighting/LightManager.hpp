#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "lighting/Light.hpp"
#include "types/ubo/LightingUBO.hpp"

namespace brassica {

	class LightManager {
	public:
		LightManager();

		struct DayNightCycle {
			bool  enabled = true;
			float time = 8.0f;       // 0.0 - 24.0 (12.0 is noon)
			float speed = 0.0125f;   // Rate of time passage
			bool  paused = false;
			float nightFactor = 0.0f; // 0.0 (day) to 1.0 (night)
			float moonOffset = 6.0f; // Hours offset from sun
			float moonAzimuth = 70.0f;

			float     moonPhaseDays = 0.0f;
			float     lunarAlbedo = 0.08f;
			float     lunarMonth = 2.0f;
			glm::vec3 moonTint = glm::vec3(0.95f, 0.93f, 0.88f);
		};

		int                 AddLight(const Light& light);
		void                RemoveLight(int id);
		Light*              GetLight(int id);
		const Light*        GetLight(int id) const;
		std::vector<Light>& GetLights();
		const std::vector<Light>& GetLights() const;
		void                Update(float deltaTime);

		glm::vec3 GetAmbientLight() const { return _ambientLight; }
		void      SetAmbientLight(const glm::vec3& ambient) { _ambientLight = ambient; }

		float GetSkyExposure() const { return _skyExposure; }
		void  SetSkyExposure(float exp) { _skyExposure = exp; }
		float GetStarExposure() const { return _starExposure; }
		void  SetStarExposure(float exp) { _starExposure = exp; }
		float GetTerrainExposure() const { return _terrainExposure; }
		void  SetTerrainExposure(float exp) { _terrainExposure = exp; }

		DayNightCycle&       GetDayNightCycle() { return _cycle; }
		const DayNightCycle& GetDayNightCycle() const { return _cycle; }

		std::vector<Light*> GetShadowCastingLights();
		int                 GetShadowCastingLightCount() const;

		LightingUBO    GetLightingUBO() const;
		LightsSSBOData GetLightsSSBOData() const;

	private:
		std::vector<Light> _lights{
			Light::CreateDirectional(0.0f, 45.0f, 10.0f, {1.0f, 1.0f, 1.0f}, true),
			Light::CreateDirectional(180.0f, -45.0f, 1.0f, {0.8f, 0.9f, 1.0f}, true)
		};
		glm::vec3     _ambientLight{0.15f, 0.15f, 0.2f};
		DayNightCycle _cycle;
		int           _nextLightId = 1;

		float _skyExposure = 1.0f;
		float _starExposure = 1.0f;
		float _terrainExposure = 1.0f;
	};

} // namespace brassica
