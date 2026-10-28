#pragma once

#include <tuple>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "ManagerReflection.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	// Mirrors the tunable subset of boidish's AtmosphereEffect Set/Get surface (the
	// "comprehensive cloud lighting interface" the clouds-rework effort asked to pull over).
	// Water-specific fields on AtmospherePushConstants (waterLevel, waterScatteringBase, etc.)
	// are deliberately not exposed here -- they're a separate underwater-rendering concern with
	// already-carefully-derived defaults (see AtmospherePushConstants.hpp's own comments), not
	// cloud/sky optics.
	//
	// rayleighScattering/mieScattering/mieExtinction/ozoneAbsorption are tiny-magnitude physical
	// coefficients (~1e-3) -- deliberately left out of GetReflection() below, since the generic
	// reflected UI's ColorEdit3 on raw values this small shows as unusable near-black.
	// AtmosphereManager::DrawUI() draws these by hand instead, with the same x1000/0.001
	// display-scale convention boidish's EnvironmentWidget.cpp uses for ozone specifically.
	struct AtmosphereManagerState {
		float     hazeDensity{0.015f};
		float     hazeHeight{20.0f};
		glm::vec3 hazeColor{0.6f, 0.7f, 0.8f};

		float rayleighScale{1.1f};
		float mieScale{0.35f};
		float mieAnisotropy{0.8f};
		float multiScatScale{1.0f};
		float atmosphereHeight{100.0f};
		float rayleighScaleHeight{8.0f};
		float mieScaleHeight{1.2f};

		glm::vec3 rayleighScattering{5.802e-3f, 13.558e-3f, 33.100e-3f};
		float     mieScattering{3.996e-3f};
		float     mieExtinction{4.440e-3f};
		glm::vec3 ozoneAbsorption{0.650e-3f, 1.881e-3f, 0.085e-3f};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("hazeDensity", "Haze Density", &AtmosphereManagerState::hazeDensity, 0.0f, 10.0f, UIHint::Slider),
				MakeField("hazeHeight", "Haze Height (km)", &AtmosphereManagerState::hazeHeight, 1.0f, 150.0f, UIHint::Slider),
				MakeColorField("hazeColor", "Haze Color", &AtmosphereManagerState::hazeColor),
				MakeField("rayleighScale", "Rayleigh Scale", &AtmosphereManagerState::rayleighScale, 0.0f, 3.0f, UIHint::Slider),
				MakeField("mieScale", "Mie Scale", &AtmosphereManagerState::mieScale, 0.0f, 4.0f, UIHint::Slider),
				MakeField(
					"mieAnisotropy", "Mie Anisotropy", &AtmosphereManagerState::mieAnisotropy, 0.0f, 0.999f, UIHint::Slider
				),
				MakeField(
					"multiScatScale",
					"Multi-Scatter Scale",
					&AtmosphereManagerState::multiScatScale,
					0.0f,
					2.0f,
					UIHint::Slider
				),
				MakeField(
					"atmosphereHeight",
					"Atmosphere Height (km)",
					&AtmosphereManagerState::atmosphereHeight,
					30.0f,
					150.0f,
					UIHint::Slider
				),
				MakeField(
					"rayleighScaleHeight",
					"Rayleigh Scale Height",
					&AtmosphereManagerState::rayleighScaleHeight,
					4.0f,
					10.0f,
					UIHint::Slider
				),
				MakeField(
					"mieScaleHeight", "Mie Scale Height", &AtmosphereManagerState::mieScaleHeight, 1.0f, 10.0f, UIHint::Slider
				)
			);
		}
	};

	class AtmosphereManager: public ManagerBase<AtmosphereManager, AtmosphereManagerState> {
	public:
		using State = AtmosphereManagerState;

		void Initialize() override { m_initialized = true; }

		[[nodiscard]] std::string GetManagerName() const override { return "AtmosphereManager"; }

		[[nodiscard]] State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

		// Custom (not ManagerBase's default): reflected fields plus hand-rolled scaled color
		// pickers for the tiny-magnitude coefficients -- see the struct comment above.
		bool DrawUI() override;

		// Copies this manager's settings into the live push-constant struct Engine uploads each
		// frame. Water-specific fields on push are left untouched.
		void ApplyTo(AtmospherePushConstants& push) const {
			push.hazeDensity = m_state.hazeDensity;
			push.hazeHeight = m_state.hazeHeight;
			push.hazeColor = m_state.hazeColor;
			push.rayleighScale = m_state.rayleighScale;
			push.mieScale = m_state.mieScale;
			push.mieAnisotropy = m_state.mieAnisotropy;
			push.atmosphereHeight = m_state.atmosphereHeight;
			push.rayleighScaleHeight = m_state.rayleighScaleHeight;
			push.mieScaleHeight = m_state.mieScaleHeight;
			push.rayleighScatteringBase = m_state.rayleighScattering;
			push.mieScatteringBase = m_state.mieScattering;
			push.mieExtinctionBase = m_state.mieExtinction;
			push.ozoneAbsorptionBase = m_state.ozoneAbsorption;
		}

		[[nodiscard]] float GetMultiScatScale() const { return m_state.multiScatScale; }

	private:
		State m_state{};
	};

} // namespace brassica
