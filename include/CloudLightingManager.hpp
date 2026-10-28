#pragma once

#include <tuple>

#include "IManager.hpp"
#include "ManagerReflection.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	// Mirrors the subset of boidish's AtmosphereEffect cloud-lighting Set/Get surface that has
	// a direct attachment point in cloud_render.frag today -- phase function, powder/multi-
	// scatter blend, self-shadow strength, per-light scale, and cirrus opacity (shared with
	// CirrusNode, which already consumes it; CloudRenderPushConstants had an unused duplicate
	// field of the same name that nothing ever read).
	//
	// cloudFlowSpeed/Direction/HeightScale drive actual wind-advected sampling now (see
	// getCloudAdvectionSpeed, cloud_render.frag) -- clouds didn't drift at all before this.
	//
	// Deliberately NOT included here (no attachment point in the current shader, flagging
	// rather than silently dropping): cloudPowderMultiplier/cloudPowderLocalScale (the powder
	// formula here doesn't match boidish's beerPowder shape), cloudShadowStepMultiplier (no
	// raymarched-shadow-step fallback exists -- shadow comes entirely from the beer-opacity
	// map), curlStrength/curlFrequency (needs multi-layer noise sampling Brassica's single
	// volume doesn't have), colorVarianceScale/Strength and sunAureoleStrength (cosmetic
	// effects with nothing to hook into yet -- no per-cloud tint pass, no sun-disc/aureole
	// render).
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

		float cloudFlowSpeed{0.25f};
		float cloudFlowDirection{3.14159265f}; // radians; boidish's default is pi (180deg)
		float cloudFlowHeightScale{0.015f};

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
				),
				MakeField(
					"cloudFlowSpeed", "Cloud Flow Speed", &CloudLightingManagerState::cloudFlowSpeed, 0.0f, 2.0f, UIHint::Slider
				),
				MakeField(
					"cloudFlowDirection",
					"Cloud Flow Direction (rad)",
					&CloudLightingManagerState::cloudFlowDirection,
					0.0f,
					6.28318f,
					UIHint::Slider
				),
				MakeField(
					"cloudFlowHeightScale",
					"Cloud Flow Height Shear",
					&CloudLightingManagerState::cloudFlowHeightScale,
					0.0f,
					0.1f,
					UIHint::Slider
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

		// Copies this manager's settings into the shared AtmosphereUBO mirror (now the common
		// settings UBO, not atmosphere-only) -- same pattern as AtmosphereManager::ApplyTo.
		void ApplyTo(AtmospherePushConstants& push) const {
			push.cloudPhaseG1 = m_state.cloudPhaseG1;
			push.cloudPhaseG2 = m_state.cloudPhaseG2;
			push.cloudPhaseAlpha = m_state.cloudPhaseAlpha;
			push.cloudPhaseIsotropic = m_state.cloudPhaseIsotropic;
			push.cloudPowderScale = m_state.cloudPowderScale;
			push.cloudBeerPowderMix = m_state.cloudBeerPowderMix;
			push.cloudShadowOpticalDepthMultiplier = m_state.cloudShadowOpticalDepthMultiplier;
			push.cloudSunLightScale = m_state.cloudSunLightScale;
			push.cloudMoonLightScale = m_state.cloudMoonLightScale;
			push.cirrusOpacity = m_state.cirrusOpacity;
			push.cloudFlowSpeed = m_state.cloudFlowSpeed;
			push.cloudFlowDirection = m_state.cloudFlowDirection;
			push.cloudFlowHeightScale = m_state.cloudFlowHeightScale;
		}

	private:
		State m_state{};
	};

} // namespace brassica
