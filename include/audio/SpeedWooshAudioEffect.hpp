#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	/**
	 * @brief Procedural wind woosh for camera moving at high speeds.
	 *
	 * Driven by camera speed (m/s). Generates dynamic aerodynamic wind woosh
	 * with bandpass sweep and resonance as speed increases.
	 */
	class SpeedWooshAudioEffect: public ProceduralAudioSource {
	public:
		SpeedWooshAudioEffect();
		~SpeedWooshAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

	private:
		uint32_t m_xorshiftState{357912468};
		float    NextWhiteNoise();

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_freq{0.02f};
		std::atomic<float> m_res{0.2f};

		float m_currentSpeed{0.0f};

		// Filter state
		float m_low{0.0f};
		float m_band{0.0f};
	};

} // namespace brassica
