#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	class WindAudioEffect: public ProceduralAudioSource {
	public:
		WindAudioEffect();
		~WindAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

	private:
		uint32_t m_xorshiftState{123456789};
		float    NextWhiteNoise();

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_freq{0.01f};
		std::atomic<float> m_res{0.1f};

		float m_currentWindStrength{0.0f};

		// State Variable Filter state
		float m_low{0.0f};
		float m_band{0.0f};
	};

} // namespace brassica
