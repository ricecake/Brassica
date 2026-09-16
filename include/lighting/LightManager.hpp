#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "lighting/ILightManager.hpp"

namespace brassica {

	class LightManager: public ILightManager {
	public:
		LightManager();
		~LightManager() override = default;

		void Initialize() override { m_initialized = true; }

		void Shutdown() override { m_initialized = false; }

		int                       AddLight(const Light& light) override;
		void                      RemoveLight(int id) override;
		Light*                    GetLight(int id) override;
		const Light*              GetLight(int id) const override;
		std::vector<Light>&       GetLights() override;
		const std::vector<Light>& GetLights() const override;
		void                      Update(float deltaTime) override;

		glm::vec3 GetAmbientLight() const override { return _ambientLight; }

		void SetAmbientLight(const glm::vec3& ambient) override { _ambientLight = ambient; }

		float GetSkyExposure() const override { return _skyExposure; }

		void SetSkyExposure(float exp) override { _skyExposure = exp; }

		float GetStarExposure() const override { return _starExposure; }

		void SetStarExposure(float exp) override { _starExposure = exp; }

		float GetTerrainExposure() const override { return _terrainExposure; }

		void SetTerrainExposure(float exp) override { _terrainExposure = exp; }

		DayNightCycle& GetDayNightCycle() override { return _cycle; }

		const DayNightCycle& GetDayNightCycle() const override { return _cycle; }

		std::vector<Light*> GetShadowCastingLights() override;
		int                 GetShadowCastingLightCount() const override;

		LightingUBO    GetLightingUBO() const override;
		LightsSSBOData GetLightsSSBOData() const override;

	private:
		std::vector<Light> _lights{
			Light::CreateDirectional(
				constants::Class::Lighting::DefaultSunAzimuth,
				constants::Class::Lighting::DefaultSunElevation,
				constants::Class::Lighting::DefaultSunIntensity,
				constants::Class::Lighting::DefaultSunColor(),
				true
			),
			Light::CreateDirectional(
				constants::Class::Lighting::DefaultMoonAzimuth,
				constants::Class::Lighting::DefaultMoonElevation,
				constants::Class::Lighting::DefaultMoonIntensity,
				constants::Class::Lighting::DefaultMoonColor(),
				true
			)
		};
		glm::vec3     _ambientLight{constants::Class::Lighting::DefaultAmbientLight()};
		DayNightCycle _cycle;
		int           _nextLightId = 1;

		float _skyExposure = constants::Class::Lighting::DefaultSkyExposure;
		float _starExposure = constants::Class::Lighting::DefaultStarExposure;
		float _terrainExposure = constants::Class::Lighting::DefaultTerrainExposure;
	};

} // namespace brassica
