#pragma once

#include "particle/IParticleManager.hpp"

namespace brassica {

	class ParticleManager: public IParticleManager {
	public:
		ParticleManager() = default;
		~ParticleManager() override = default;

		void Initialize() override { m_initialized = true; }

		void Shutdown() override { m_initialized = false; }

		[[nodiscard]] State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

		[[nodiscard]] bool IsEnabled() const override { return m_state.enabled; }

		void SetEnabled(bool enabled) override { m_state.enabled = enabled; }

		[[nodiscard]] std::uint32_t GetActiveParticles() const override { return m_state.activeParticles; }

		void SetActiveParticles(std::uint32_t count) override { m_state.activeParticles = count; }

		[[nodiscard]] std::uint32_t GetMaxParticles() const override { return m_state.maxParticles; }

		void SetMaxParticles(std::uint32_t maxCount) override { m_state.maxParticles = maxCount; }

		[[nodiscard]] float GetBirdProportion() const override { return m_state.birdProportion; }

		void SetBirdProportion(float prop) override { m_state.birdProportion = prop; }

		[[nodiscard]] float GetFishProportion() const override { return m_state.fishProportion; }

		void SetFishProportion(float prop) override { m_state.fishProportion = prop; }

		[[nodiscard]] float GetFireflyProportion() const override { return m_state.fireflyProportion; }

		void SetFireflyProportion(float prop) override { m_state.fireflyProportion = prop; }

		[[nodiscard]] bool GetEnableLights() const override { return m_state.enableLights; }

		void SetEnableLights(bool enable) override { m_state.enableLights = enable; }

		void GetCutoffs(float& outBirdCutoff, float& outFishCutoff) const override;

	private:
		ParticleManagerState m_state;
	};

} // namespace brassica
