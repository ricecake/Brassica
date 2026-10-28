#pragma once

#include <tuple>

#include "IManager.hpp"
#include "ManagerReflection.hpp"

namespace brassica {

	// Mirrors the subset of boidish's AtmosphereEffect cloud-lighting Set/Get surface that has
	// a direct attachment point in cloud_render.frag today -- phase function, powder/multi-
	// scatter blend, self-shadow strength, per-light scale, and cirrus opacity (shared with
	// CirrusNode, which already consumes it; CloudRenderPushConstants had an unused duplicate
	// field of the same name that nothing ever read).
	//
	// Deliberately NOT included here (no attachment point in the current shader, flagging
	// rather than silently dropping): cloudPowderMultiplier/cloudPowderLocalScale (the powder
	// formula here doesn't match boidish's beerPowder shape), cloudShadowStepMultiplier (no
	// raymarched-shadow-step fallback exists -- shadow comes entirely from the beer-opacity
	// map), curlStrength/curlFrequency (needs multi-layer noise sampling Brassica's single
	// volume doesn't have), colorVarianceScale/Strength and sunAureoleStrength (cosmetic
	// effects with nothing to hook into yet -- no per-cloud tint pass, no sun-disc/aureole
	// render). cloudFlowSpeed/Direction/HeightScale (wind-driven advection) is handled
	// separately -- it needs new sampling logic, not just a constant swap.
	struct CloudLightingManagerState {
		float cloudPhaseG1{0.8f};
		float cloudPhaseG2{-0.2f};
		float cloudPhaseAlpha{0.5f};
		float cloudPhaseIsotropic{0.52f};

		float cloudPowderScale{300.0f};
		float cloudBeerPowderMix{0.5f};

		float cloudShadowOpticalDepthMultiplier{1.0f};

		float cloudSunLightScale{1.0f};
		float cloudMoonLightScale{1.0f};

		float cirrusOpacity{0.3f};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField(
					"cloudPhaseG1", "Phase G1 (forward lobe)", &CloudLightingManagerState::cloudPhaseG1, -0.999f, 0.999f, UIHint::Slider
				),
				MakeField(
					"cloudPhaseG2", "Phase G2 (back lobe)", &CloudLightingManagerState::cloudPhaseG2, -0.999f, 0.999f, UIHint::Slider
				),
				MakeField(
					"cloudPhaseAlpha",
					"Phase Lobe Blend",
					&CloudLightingManagerState::cloudPhaseAlpha,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudPhaseIsotropic",
					"Phase Isotropic Mix",
					&CloudLightingManagerState::cloudPhaseIsotropic,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudPowderScale", "Powder Scale", &CloudLightingManagerState::cloudPowderScale, 0.0f, 2000.0f, UIHint::Slider
				),
				MakeField(
					"cloudBeerPowderMix",
					"Beer-Powder Mix",
					&CloudLightingManagerState::cloudBeerPowderMix,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudShadowOpticalDepthMultiplier",
					"Self-Shadow Strength",
					&CloudLightingManagerState::cloudShadowOpticalDepthMultiplier,
					0.0f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudSunLightScale",
					"Sun Light Scale",
					&CloudLightingManagerState::cloudSunLightScale,
					0.0f,
					3.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudMoonLightScale",
					"Moon Light Scale",
					&CloudLightingManagerState::cloudMoonLightScale,
					0.0f,
					3.0f,
					UIHint::Slider
				),
				MakeField(
					"cirrusOpacity", "Cirrus Opacity", &CloudLightingManagerState::cirrusOpacity, 0.0f, 1.0f, UIHint::Slider
				)
			);
		}
	};

	class CloudLightingManager: public ManagerBase<CloudLightingManager, CloudLightingManagerState> {
	public:
		using State = CloudLightingManagerState;

		void Initialize() override { m_initialized = true; }

		[[nodiscard]] std::string GetManagerName() const override { return "CloudLightingManager"; }

		[[nodiscard]] State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

	private:
		State m_state{};
	};

} // namespace brassica
