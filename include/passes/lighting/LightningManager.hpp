#pragma once

#include <string>
#include <tuple>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "ManagerReflection.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	// GPU-driven lightning (shaders/climate_ca.comp's trigger, shaders/lightning_age.comp's
	// reserved-light-slot write, shaders/lightning.task/.mesh/.frag's draw) -- this manager is
	// purely a settings surface mirrored into AtmosphereUBO (ApplyTo), same pattern as
	// CloudLightingManager. Ported from boidish's LightningManager/LightningWidget "Global
	// Settings" panel (external/boidish/include/lightning_manager.h,
	// include/ui/LightningWidget.h) -- settings only. Brassica's lightning has no CPU
	// TriggerStrike() path (strikes are claimed/written entirely on the GPU), so boidish's
	// manual-trigger buttons and active-strike list aren't ported.
	//
	// Replaces the old CPU-only LightningManager/LightningStrike/LightningType (dead code: its
	// TriggerStrike() had zero production callers, only a now-deleted test exercised it, and the
	// lightningColor it fed into LightingUBO had no shader reader).
	struct LightningManagerState {
		// Multiplies the per-cell trigger probability in climate_ca.comp. Moved here from
		// CloudLightingManagerState, which now only holds cloud-rendering settings.
		float frequencyScale{1.0f};
		// Scales bolt brightness, flash-light intensity, and localized cloud glow strength
		// together -- boidish's single Intensity Multiplier. Renamed from
		// CloudLightingManagerState's lightningFlashIntensity now that it drives the bolt mesh
		// and cloud glow too, not just the flash light.
		float intensityScale{1.0f};
		// Metres. climate_ca.comp only rolls a strike for cells within this of the camera's XZ
		// position -- strikes used to spawn planet-wide, which was most of why lightning read as
		// "always firing" (see climate_ca.comp's trigger comment for the rate derivation).
		float spawnRadius{60000.0f};
		// climate_ca.comp's stormFactor = smoothstep(stormThreshold, 1.0, lightningPropensity) --
		// how convective a cell needs to be before it can trigger at all.
		float stormThreshold{0.35f};
		// Metres. The cloud glow's falloff radius from the strike (cloud_raymarch.comp) --
		// without this, any live strike anywhere lit every visible cloud on screen.
		float cloudGlowRadius{4000.0f};
		// Metres. The cloud-to-cloud FLASH light's radius (lightning_age.comp) -- was
		// effectively 0 before (a FLASH/POINT innerCutoff-as-radius field mismatch), so it never
		// lit anything. Matches boidish's own cloud-to-cloud flash radius (lightning_manager.cpp's
		// Light::CreateFlash(..., 500.0f)) -- helpers/lighting.glsl's FLASH falloff
		// (1/(1+d/r)^exp, hard cutoff past 2r) barely drops at all for d << r, so this has to be
		// the real human-scale illumination radius, not a glow-radius-sized number like
		// cloudGlowRadius above (that was the original mistake here: 6000 read as a near-
		// uniform, falloff-less ball out to several km, instead of a local flash).
		float cloudFlashRadius{500.0f};
		// Debug override: forces stormFactor to 1.0 regardless of real weather, so strikes are
		// visible without waiting for a storm. GPU equivalent of boidish's LightningWidget
		// "Manual Auto-trigger".
		bool ignoreWeather{false};
		// Fraction of triggered strikes that become cloud-to-ground instead of cloud-to-cloud
		// (climate_ca.comp) -- boidish had separate BOLT/FORK/CLOUD_TO_CLOUD types manually
		// triggered per-call; this is the GPU-driven equivalent of picking which type a given
		// automatic trigger becomes.
		float groundStrikeChance{0.3f};
		// Metres. The cloud-to-ground FLASH light's radius, placed at the resolved ground point.
		// Matches boidish's own ground-strike flash radius (lightning_manager.cpp's
		// Light::CreateFlash(..., 300.0f)) -- same falloff-shape reasoning as cloudFlashRadius
		// above applies here too.
		float groundFlashRadius{300.0f};
		// Multiplies the rolled strike duration -- boidish's Lifetime Multiplier.
		float lifetimeScale{1.0f};
		// Color variety (climate_ca.comp picks the final color at trigger time -- folk-
		// meteorology color associations for artistic effect, not physically modeled). Base is
		// the existing blue-white; humid/dusty are blend targets driven by that cell's own
		// airMoisture+rain and aerosol, respectively.
		glm::vec3 baseColor{0.8f, 0.85f, 1.0f};
		glm::vec3 humidColor{0.75f, 0.7f, 1.0f};
		glm::vec3 dustyColor{1.0f, 0.85f, 0.6f};
		// 0..1. Per-strike brightness jitter strength on top of the weather-driven color above
		// -- not a blend toward literal random noise, which reads as broken rather than varied.
		float colorVariance{0.15f};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField(
					"frequencyScale",
					"Frequency Multiplier",
					&LightningManagerState::frequencyScale,
					0.0f,
					50.0f,
					UIHint::Slider
				),
				MakeField(
					"intensityScale",
					"Intensity Multiplier",
					&LightningManagerState::intensityScale,
					0.0f,
					10.0f,
					UIHint::Slider
				),
				MakeField(
					"spawnRadius", "Spawn Radius (m)", &LightningManagerState::spawnRadius, 0.0f, 200000.0f, UIHint::Slider
				),
				MakeField(
					"stormThreshold",
					"Storm Threshold",
					&LightningManagerState::stormThreshold,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudGlowRadius",
					"Cloud Glow Radius (m)",
					&LightningManagerState::cloudGlowRadius,
					100.0f,
					20000.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudFlashRadius",
					"Cloud Flash Radius (m)",
					&LightningManagerState::cloudFlashRadius,
					100.0f,
					20000.0f,
					UIHint::Slider
				),
				MakeField("ignoreWeather", "Ignore Weather (Debug)", &LightningManagerState::ignoreWeather),
				MakeField(
					"groundStrikeChance",
					"Ground Strike Chance",
					&LightningManagerState::groundStrikeChance,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"groundFlashRadius",
					"Ground Flash Radius (m)",
					&LightningManagerState::groundFlashRadius,
					100.0f,
					20000.0f,
					UIHint::Slider
				),
				MakeField(
					"lifetimeScale", "Lifetime Multiplier", &LightningManagerState::lifetimeScale, 0.1f, 5.0f, UIHint::Slider
				),
				MakeColorField("baseColor", "Base Color", &LightningManagerState::baseColor),
				MakeColorField("humidColor", "Humid Color", &LightningManagerState::humidColor),
				MakeColorField("dustyColor", "Dusty Color", &LightningManagerState::dustyColor),
				MakeField(
					"colorVariance", "Color Variance", &LightningManagerState::colorVariance, 0.0f, 1.0f, UIHint::Slider
				)
			);
		}
	};

	class LightningManager: public ManagerBase<LightningManager, LightningManagerState> {
	public:
		using State = LightningManagerState;

		void Initialize() override { m_initialized = true; }

		[[nodiscard]] std::string GetManagerName() const override { return "LightningManager"; }

		[[nodiscard]] State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

		// Copies this manager's settings into the shared AtmosphereUBO mirror -- same pattern as
		// CloudLightingManager::ApplyTo.
		void ApplyTo(AtmospherePushConstants& push) const {
			push.lightningFrequencyScale = m_state.frequencyScale;
			push.lightningIntensityScale = m_state.intensityScale;
			push.lightningSpawnRadius = m_state.spawnRadius;
			push.lightningStormThreshold = m_state.stormThreshold;
			push.lightningCloudGlowRadius = m_state.cloudGlowRadius;
			push.lightningCloudFlashRadius = m_state.cloudFlashRadius;
			push.lightningIgnoreWeather = m_state.ignoreWeather ? 1u : 0u;
			push.lightningGroundStrikeChance = m_state.groundStrikeChance;
			push.lightningGroundFlashRadius = m_state.groundFlashRadius;
			push.lightningLifetimeScale = m_state.lifetimeScale;
			push.lightningBaseColor = m_state.baseColor;
			push.lightningHumidColor = m_state.humidColor;
			push.lightningDustyColor = m_state.dustyColor;
			push.lightningColorVariance = m_state.colorVariance;
		}

	private:
		State m_state{};
	};

} // namespace brassica
