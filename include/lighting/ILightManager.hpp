#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "constants.h"
#include "IManager.hpp"
#include "lighting/Light.hpp"
#include "types/ubo/LightingUBO.hpp"

namespace brassica {

	struct LightManagerState {
		glm::vec3 ambientLight{constants::Class::Lighting::DefaultAmbientLight()};
		float     skyExposure = constants::Class::Lighting::DefaultSkyExposure;
		float     starExposure = constants::Class::Lighting::DefaultStarExposure;
		float     terrainExposure = constants::Class::Lighting::DefaultTerrainExposure;

		bool      cycleEnabled = true;
		float     cycleTime = constants::Class::Lighting::DefaultCycleTime;
		float     cycleSpeed = constants::Class::Lighting::DefaultCycleSpeed;
		bool      cyclePaused = false;
		float     moonOffset = constants::Class::Lighting::DefaultMoonOffset;
		float     moonAzimuth = constants::Class::Lighting::DefaultMoonAzimuthBase;
		float     lunarAlbedo = constants::Class::Lighting::DefaultLunarAlbedo;
		float     lunarMonth = constants::Class::Lighting::DefaultLunarMonth;
		glm::vec3 moonTint{constants::Class::Lighting::DefaultMoonColor()};

		auto GetReflection() const {
			return std::make_tuple(
				MakeColorField("ambientLight", "Ambient Light", &LightManagerState::ambientLight),
				MakeField("skyExposure", "Sky Exposure", &LightManagerState::skyExposure, 0.0f, 10.0f, UIHint::Slider),
				MakeField(
					"starExposure",
					"Star Exposure",
					&LightManagerState::starExposure,
					0.0f,
					10.0f,
					UIHint::Slider
				),
				MakeField(
					"terrainExposure",
					"Terrain Exposure",
					&LightManagerState::terrainExposure,
					0.0f,
					10.0f,
					UIHint::Slider
				),
				MakeField("cycleEnabled", "Enable Day/Night Cycle", &LightManagerState::cycleEnabled),
				MakeField("cycleTime", "Time of Day (24h)", &LightManagerState::cycleTime, 0.0f, 24.0f, UIHint::Slider),
				MakeField("cycleSpeed", "Cycle Speed", &LightManagerState::cycleSpeed, 0.0f, 100.0f, UIHint::Slider),
				MakeField("cyclePaused", "Pause Day/Night Cycle", &LightManagerState::cyclePaused),
				MakeField(
					"moonOffset",
					"Moon Offset (Hours)",
					&LightManagerState::moonOffset,
					-12.0f,
					12.0f,
					UIHint::Slider
				),
				MakeField(
					"moonAzimuth",
					"Moon Azimuth Base",
					&LightManagerState::moonAzimuth,
					0.0f,
					360.0f,
					UIHint::Slider
				),
				MakeField("lunarAlbedo", "Lunar Albedo", &LightManagerState::lunarAlbedo, 0.0f, 1.0f, UIHint::Slider),
				MakeField(
					"lunarMonth",
					"Lunar Month (Days)",
					&LightManagerState::lunarMonth,
					1.0f,
					100.0f,
					UIHint::Slider
				),
				MakeColorField("moonTint", "Moon Tint Color", &LightManagerState::moonTint)
			);
		}
	};

	class ILightManager: public ManagerBase<ILightManager, LightManagerState> {
	public:
		using State = LightManagerState;

		~ILightManager() override = default;

		std::string GetManagerName() const override { return "LightManager"; }

		struct DayNightCycle {
			bool  enabled = true;
			float time = constants::Class::Lighting::DefaultCycleTime;   // 0.0 - 24.0 (12.0 is noon)
			float speed = constants::Class::Lighting::DefaultCycleSpeed; // Rate of time passage
			bool  paused = false;
			float nightFactor = 0.0f;                                         // 0.0 (day) to 1.0 (night)
			float moonOffset = constants::Class::Lighting::DefaultMoonOffset; // Hours offset from sun
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
