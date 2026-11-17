#pragma once

#include <tuple>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "ManagerReflection.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	// Mirrors the subset of boidish's AtmosphereEffect cloud-lighting Set/Get surface that has
	// a direct attachment point in cloud_render.frag today -- phase function, powder/multi-
	// scatter blend, self-shadow strength, per-light scale, cirrus opacity (shared with
	// CirrusNode, which already consumes it; CloudRenderPushConstants had an unused duplicate
	// field of the same name that nothing ever read), final color tint, per-channel extinction,
	// and albedo.
	//
	// ambientBeerPowderScale is Brassica's own separate ambient-glow term (cloud_render.frag's
	// `1.0 - 1.0/(1.0+25*stepDensity*scale)`) -- NOT the same quantity as boidish's own
	// cloud_powder_scale_, which only feeds beerPowder()'s d-scale term and is exposed here as
	// directBeerPowderScale instead, so the two don't collide under one name.
	//
	// cloudFlowSpeed/Direction/HeightScale drive actual wind-advected sampling now (see
	// getCloudAdvectionSpeed, cloud_render.frag) -- clouds didn't drift at all before this.
	//
	// Deliberately NOT included (dead even in boidish itself, not a real parity target):
	// cloudShadowStepMultiplier (zero GLSL consumers anywhere in boidish's own shader tree)
	// and curlStrength/curlFrequency (only referenced from boidish's shaders/archive/, outside
	// its live include chain, in 3 mutually-inconsistent formulas). colorVarianceScale/Strength
	// belongs on AtmosphereManager instead -- it's a sky/scattering-coefficient shimmer, not a
	// cloud setting, despite boidish grouping it under the same UI panel.
	struct CloudLightingManagerState {
		float cloudPhaseG1{0.8f};
		float cloudPhaseG2{-0.2f};
		float cloudPhaseAlpha{0.5f};
		float cloudPhaseIsotropic{0.52f};

		// Brassica's separate ambient-glow powder term (see comment above).
		float ambientBeerPowderScale{300.0f};
		// beerPowder()'s own terms (shaders/common.glsl) -- previously hardcoded literals
		// (0.1, 50, 2.67) with no settings surface at all.
		float directBeerPowderScale{0.1f};
		float cloudPowderMultiplier{50.0f};
		float cloudPowderLocalScale{2.67f};
		float cloudBeerPowderMix{0.5f};

		float cloudShadowOpticalDepthMultiplier{1.0f};

		float cloudSunLightScale{1.0f};
		float cloudMoonLightScale{1.0f};

		float cirrusOpacity{0.3f};

		float cloudFlowSpeed{0.25f};
		float cloudFlowDirection{3.14159265f}; // radians; boidish's default is pi (180deg)
		float cloudFlowHeightScale{0.015f};

		// Final flat tint on the cloud march's accumulated light (boidish's cloudColorUniform).
		// Defaults to a no-op (1,1,1), not boidish's own tuned {0.95,0.95,1.0} -- Brassica's
		// extinction formula is its own adaptation, not boidish's verbatim, so boidish's tuned
		// defaults have no particular reason to look right here; this way the port doesn't
		// silently change today's look, and the slider is there once someone wants to dial it in.
		glm::vec3 cloudColor{1.0f, 1.0f, 1.0f};
		// Per-channel extinction: rgbExtinction = cloudExtinctionColor * cloudExtinctionScale,
		// multiplied into marchClouds()'s density->optical-depth->transmittance chain. Same
		// no-op-default reasoning as cloudColor -- boidish's own default (0.372) would silently
		// cut extinction to ~37% of today's look the moment this landed.
		float     cloudExtinctionScale{1.0f};
		glm::vec3 cloudExtinctionColor{1.0f, 1.0f, 1.0f};
		// Scales the cloud's reflected light (direct + ambient), not the lightning emissive
		// glow. boidish also multiplies by a per-sample albedo from calculateCloudDensity(),
		// which Brassica's live density function has no equivalent of -- flat-uniform-only.
		// Same no-op-default reasoning as cloudColor/cloudExtinctionScale.
		glm::vec3 cloudAlbedo{1.0f, 1.0f, 1.0f};

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
					"ambientBeerPowderScale",
					"Ambient Powder Scale",
					&CloudLightingManagerState::ambientBeerPowderScale,
					0.0f,
					2000.0f,
					UIHint::Slider
				),
				MakeField(
					"directBeerPowderScale",
					"Beer-Powder Depth Scale",
					&CloudLightingManagerState::directBeerPowderScale,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudPowderMultiplier",
					"Beer-Powder Multiplier",
					&CloudLightingManagerState::cloudPowderMultiplier,
					0.0f,
					100.0f,
					UIHint::Slider
				),
				MakeField(
					"cloudPowderLocalScale",
					"Beer-Powder Local Scale",
					&CloudLightingManagerState::cloudPowderLocalScale,
					0.0f,
					10.0f,
					UIHint::Slider
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
				),
				MakeColorField("cloudColor", "Cloud Color", &CloudLightingManagerState::cloudColor),
				MakeField(
					"cloudExtinctionScale",
					"Extinction Scale",
					&CloudLightingManagerState::cloudExtinctionScale,
					0.001f,
					1.0f,
					UIHint::Slider
				),
				MakeColorField(
					"cloudExtinctionColor", "Extinction Balance", &CloudLightingManagerState::cloudExtinctionColor
				),
				MakeColorField("cloudAlbedo", "Cloud Albedo", &CloudLightingManagerState::cloudAlbedo)
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
			push.ambientBeerPowderScale = m_state.ambientBeerPowderScale;
			push.directBeerPowderScale = m_state.directBeerPowderScale;
			push.cloudPowderMultiplier = m_state.cloudPowderMultiplier;
			push.cloudPowderLocalScale = m_state.cloudPowderLocalScale;
			push.cloudBeerPowderMix = m_state.cloudBeerPowderMix;
			push.cloudShadowOpticalDepthMultiplier = m_state.cloudShadowOpticalDepthMultiplier;
			push.cloudSunLightScale = m_state.cloudSunLightScale;
			push.cloudMoonLightScale = m_state.cloudMoonLightScale;
			push.cirrusOpacity = m_state.cirrusOpacity;
			push.cloudFlowSpeed = m_state.cloudFlowSpeed;
			push.cloudFlowDirection = m_state.cloudFlowDirection;
			push.cloudFlowHeightScale = m_state.cloudFlowHeightScale;
			push.cloudColor = m_state.cloudColor;
			push.cloudExtinctionScale = m_state.cloudExtinctionScale;
			push.cloudExtinctionColor = m_state.cloudExtinctionColor;
			push.cloudAlbedo = m_state.cloudAlbedo;
		}

	private:
		State m_state{};
	};

} // namespace brassica
