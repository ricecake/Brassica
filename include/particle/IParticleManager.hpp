#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "IManager.hpp"

namespace brassica {

	struct ParticleManagerState {
		bool          enabled{true};
		std::uint32_t activeParticles{8192};
		std::uint32_t maxParticles{8192};
		float         birdProportion{0.34f};
		float         fishProportion{0.33f};
		float         fireflyProportion{0.33f};
		bool          enableLights{true};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("enabled", "Enable Particles", &ParticleManagerState::enabled),
				MakeField(
					"activeParticles",
					"Active Particle Count",
					&ParticleManagerState::activeParticles,
					0u,
					8192u,
					UIHint::Slider
				),
				MakeField("birdProportion", "Bird Proportion", &ParticleManagerState::birdProportion, 0.0f, 1.0f, UIHint::Slider),
				MakeField("fishProportion", "Fish Proportion", &ParticleManagerState::fishProportion, 0.0f, 1.0f, UIHint::Slider),
				MakeField(
					"fireflyProportion",
					"Firefly Proportion",
					&ParticleManagerState::fireflyProportion,
					0.0f,
					1.0f,
					UIHint::Slider
				),
				MakeField("enableLights", "Enable Lights Creation", &ParticleManagerState::enableLights)
			);
		}
	};

	class IParticleManager: public ManagerBase<IParticleManager, ParticleManagerState> {
	public:
		using State = ParticleManagerState;

		~IParticleManager() override = default;

		[[nodiscard]] std::string GetManagerName() const override { return "ParticleManager"; }

		[[nodiscard]] virtual bool          IsEnabled() const = 0;
		virtual void                        SetEnabled(bool enabled) = 0;
		[[nodiscard]] virtual std::uint32_t GetActiveParticles() const = 0;
		virtual void                        SetActiveParticles(std::uint32_t count) = 0;
		[[nodiscard]] virtual std::uint32_t GetMaxParticles() const = 0;
		virtual void                        SetMaxParticles(std::uint32_t maxCount) = 0;
		[[nodiscard]] virtual float         GetBirdProportion() const = 0;
		virtual void                        SetBirdProportion(float prop) = 0;
		[[nodiscard]] virtual float         GetFishProportion() const = 0;
		virtual void                        SetFishProportion(float prop) = 0;
		[[nodiscard]] virtual float         GetFireflyProportion() const = 0;
		virtual void                        SetFireflyProportion(float prop) = 0;
		[[nodiscard]] virtual bool          GetEnableLights() const = 0;
		virtual void                        SetEnableLights(bool enable) = 0;

		virtual void GetCutoffs(float& outBirdCutoff, float& outFishCutoff) const = 0;
	};

} // namespace brassica
