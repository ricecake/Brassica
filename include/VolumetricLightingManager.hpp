#pragma once

#include <tuple>

#include "IManager.hpp"
#include "ManagerReflection.hpp"

namespace brassica {

	// VolumetricFroxelGenNode (shaders/volumetric_froxel_gen.comp) had no settings surface at
	// all -- fog density/falloff/base-height were hardcoded directly in its own SetFrameParams,
	// and its mie term had no humidity coupling despite already sampling weatherMoisture for
	// other purposes (rain density/tint). This is the volumetric-lighting system's equivalent
	// of AtmosphereManager: a real place to tune the fog "look," plus the one new physical
	// term (mieHumidityBoost).
	struct VolumetricLightingManagerState {
		float fogDensity{0.003f};
		float fogHeightFalloff{0.003f};
		float fogBaseHeight{0.0f};
		float mieHumidityBoost{5.0f};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField(
					"fogDensity", "Fog Density", &VolumetricLightingManagerState::fogDensity, 0.0f, 0.05f, UIHint::Slider
				),
				MakeField(
					"fogHeightFalloff",
					"Fog Height Falloff",
					&VolumetricLightingManagerState::fogHeightFalloff,
					0.0f,
					0.05f,
					UIHint::Slider
				),
				MakeField(
					"fogBaseHeight",
					"Fog Base Height (m)",
					&VolumetricLightingManagerState::fogBaseHeight,
					-500.0f,
					2000.0f,
					UIHint::Slider
				),
				MakeField(
					"mieHumidityBoost",
					"Mie Humidity Boost",
					&VolumetricLightingManagerState::mieHumidityBoost,
					0.0f,
					15.0f,
					UIHint::Slider
				)
			);
		}
	};

	class VolumetricLightingManager: public ManagerBase<VolumetricLightingManager, VolumetricLightingManagerState> {
	public:
		using State = VolumetricLightingManagerState;

		void Initialize() override { m_initialized = true; }

		[[nodiscard]] std::string GetManagerName() const override { return "VolumetricLightingManager"; }

		[[nodiscard]] State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

	private:
		State m_state{};
	};

} // namespace brassica
