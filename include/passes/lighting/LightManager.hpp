#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "passes/lighting/ILightManager.hpp"
#include "passes/RenderPhases.hpp"
#include "resources/core.hpp"

namespace brassica {

	class LightManager: public ILightManager {
	public:
		LightManager();
		~LightManager() override = default;

		void Initialize() override { m_initialized = true; }

		void Shutdown() override { m_initialized = false; }

		State GetState() const override {
			State state;
			state.ambientLight = _ambientLight;
			state.skyExposure = _skyExposure;
			state.starExposure = _starExposure;
			state.terrainExposure = _terrainExposure;
			state.enableGTAO = _enableGTAO;
			state.gtaoRadius = _gtaoRadius;
			state.gtaoIntensity = _gtaoIntensity;
			state.gtaoDirectInfluence = _gtaoDirectInfluence;
			state.debugVisualizeGTAO = _debugVisualizeGTAO;

			state.cycleEnabled = _cycle.enabled;
			state.cycleTime = _cycle.time;
			state.cycleSpeed = _cycle.speed;
			state.cyclePaused = _cycle.paused;
			state.moonOffset = _cycle.moonOffset;
			state.moonAzimuth = _cycle.moonAzimuth;
			state.lunarAlbedo = _cycle.lunarAlbedo;
			state.lunarMonth = _cycle.lunarMonth;
			state.moonTint = _cycle.moonTint;
			return state;
		}

		void SetState(const State& state) override {
			_ambientLight = state.ambientLight;
			_skyExposure = state.skyExposure;
			_starExposure = state.starExposure;
			_terrainExposure = state.terrainExposure;
			_enableGTAO = state.enableGTAO;
			_gtaoRadius = state.gtaoRadius;
			_gtaoIntensity = state.gtaoIntensity;
			_gtaoDirectInfluence = state.gtaoDirectInfluence;
			_debugVisualizeGTAO = state.debugVisualizeGTAO;

			_cycle.enabled = state.cycleEnabled;
			_cycle.time = state.cycleTime;
			_cycle.speed = state.cycleSpeed;
			_cycle.paused = state.cyclePaused;
			_cycle.moonOffset = state.moonOffset;
			_cycle.moonAzimuth = state.moonAzimuth;
			_cycle.lunarAlbedo = state.lunarAlbedo;
			_cycle.lunarMonth = state.lunarMonth;
			_cycle.moonTint = state.moonTint;
		}

		int                       AddLight(const Light& light) override;
		void                      RemoveLight(int id) override;
		Light*                    GetLight(int id) override;
		const Light*              GetLight(int id) const override;
		std::vector<Light>&       GetLights() override;
		const std::vector<Light>& GetLights() const override;
		using ILightManager::Update;
		void                      Update(float deltaTime) override;
		void                      Update(float deltaTime, const glm::vec3& cameraPosition) override;

		glm::vec3 GetAmbientLight() const override { return _ambientLight; }

		void SetAmbientLight(const glm::vec3& ambient) override { _ambientLight = ambient; }

		float GetSkyExposure() const override { return _skyExposure; }

		void SetSkyExposure(float exp) override { _skyExposure = exp; }

		float GetStarExposure() const override { return _starExposure; }

		void SetStarExposure(float exp) override { _starExposure = exp; }

		float GetTerrainExposure() const override { return _terrainExposure; }

		void SetTerrainExposure(float exp) override { _terrainExposure = exp; }

		bool IsGTAOEnabled() const override { return _enableGTAO; }

		void SetGTAOEnabled(bool enable) override { _enableGTAO = enable; }

		float GetGTAORadius() const override { return _gtaoRadius; }

		void SetGTAORadius(float radius) override { _gtaoRadius = radius; }

		float GetGTAOIntensity() const override { return _gtaoIntensity; }

		void SetGTAOIntensity(float intensity) override { _gtaoIntensity = intensity; }

		float GetGTAODirectInfluence() const override { return _gtaoDirectInfluence; }

		void SetGTAODirectInfluence(float influence) override { _gtaoDirectInfluence = influence; }

		bool IsGTAODebugVisualizeEnabled() const override { return _debugVisualizeGTAO; }

		void SetGTAODebugVisualizeEnabled(bool enable) override { _debugVisualizeGTAO = enable; }

		DayNightCycle& GetDayNightCycle() override { return _cycle; }

		const DayNightCycle& GetDayNightCycle() const override { return _cycle; }

		std::vector<Light*> GetShadowCastingLights() override;
		int                 GetShadowCastingLightCount() const override;

		LightingUBO       GetLightingUBO() const override;
		CpuLightsSSBOData GetCpuLightsSSBOData() const override;

		// Registers this frame's CPU-lights upload node into the graph, mirroring
		// MaterialManager::RegisterBufferNode exactly -- SyncBufferNode only calls SetData()
		// (which is what actually marks the node dirty) when something about the lights
		// genuinely changed since the last call, so an idle frame with the cycle paused and no
		// lights added/removed/mutated uploads nothing at all.
		void RegisterBufferNode(graph::Graph& frameGraph);

	private:
		void SyncBufferNode();

		graph::PredefinedBufferNode<CpuLightsBuffer, CpuLightsSSBOData, SubPhase::Prepare> m_bufferNode;
		CpuLightsSSBOData                                                                 _cpuLightsCache{};

		// Set by AddLight/RemoveLight/the non-const GetLight (a caller holding that pointer may
		// mutate the light directly) and by Update() whenever the day/night cycle is enabled
		// (sun/moon direction is recomputed from cycle time + camera position every such call,
		// so treating that as "changed" is the safe default -- see RegisterBufferNode's comment
		// for why an exact dirty/clean distinction isn't attempted beyond this).
		bool _lightsDirty = true;

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
		bool  _enableGTAO = true;
		float _gtaoRadius = constants::Class::Lighting::DefaultGTAORadius;
		float _gtaoIntensity = constants::Class::Lighting::DefaultGTAOIntensity;
		float _gtaoDirectInfluence = constants::Class::Lighting::DefaultGTAODirectInfluence;
		bool  _debugVisualizeGTAO = false;
	};

} // namespace brassica
