#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "constants.h"
#include "lighting/Light.hpp"
#include "types/ubo/LightingUBO.hpp"

namespace brassica {

	class ILightManager : public IManager {
	public:
		~ILightManager() override = default;

		struct DayNightCycle {
			bool  enabled = true;
			float time = constants::Class::Lighting::DefaultCycleTime;     // 0.0 - 24.0 (12.0 is noon)
			float speed = constants::Class::Lighting::DefaultCycleSpeed; // Rate of time passage
			bool  paused = false;
			float nightFactor = 0.0f; // 0.0 (day) to 1.0 (night)
			float moonOffset = constants::Class::Lighting::DefaultMoonOffset;  // Hours offset from sun
			float moonAzimuth = constants::Class::Lighting::DefaultMoonAzimuthBase;

			float     moonPhaseDays = 0.0f;
			float     lunarAlbedo = constants::Class::Lighting::DefaultLunarAlbedo;
			float     lunarMonth = constants::Class::Lighting::DefaultLunarMonth;
			glm::vec3 moonTint = constants::Class::Lighting::DefaultMoonColor();
		};

		virtual int                       AddLight(const Light& light) = 0;
		virtual void                      RemoveLight(int id) = 0;
		virtual Light*                    GetLight(int id) = 0;
		virtual const Light*              GetLight(int id) const = 0;
		virtual std::vector<Light>&       GetLights() = 0;
		virtual const std::vector<Light>& GetLights() const = 0;
		virtual void                      Update(float deltaTime) = 0;

		virtual glm::vec3 GetAmbientLight() const = 0;
		virtual void      SetAmbientLight(const glm::vec3& ambient) = 0;

		virtual float GetSkyExposure() const = 0;
		virtual void  SetSkyExposure(float exp) = 0;

		virtual float GetStarExposure() const = 0;
		virtual void  SetStarExposure(float exp) = 0;

		virtual float GetTerrainExposure() const = 0;
		virtual void  SetTerrainExposure(float exp) = 0;

		virtual DayNightCycle&       GetDayNightCycle() = 0;
		virtual const DayNightCycle& GetDayNightCycle() const = 0;

		virtual std::vector<Light*> GetShadowCastingLights() = 0;
		virtual int                 GetShadowCastingLightCount() const = 0;

		virtual LightingUBO    GetLightingUBO() const = 0;
		virtual LightsSSBOData GetLightsSSBOData() const = 0;
	};

} // namespace brassica
