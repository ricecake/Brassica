#include "audio/WindAudioEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	WindAudioEffect::WindAudioEffect()
		: ProceduralAudioSource(2, 48000) {}

	float WindAudioEffect::NextWhiteNoise() {
		m_xorshiftState ^= m_xorshiftState << 13;
		m_xorshiftState ^= m_xorshiftState >> 17;
		m_xorshiftState ^= m_xorshiftState << 5;
		return (static_cast<float>(m_xorshiftState) / static_cast<float>(0xFFFFFFFF)) * 2.0f - 1.0f;
	}

	void WindAudioEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float f = m_freq.load(std::memory_order_relaxed);
		float res = m_res.load(std::memory_order_relaxed);
		float gain = m_gain.load(std::memory_order_relaxed);

		if (gain <= 0.0001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		float damping = 1.0f / (res + 0.001f);

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			float input = NextWhiteNoise();

			// SVF Step
			m_low = m_low + f * m_band;
			float high = input - m_low - damping * m_band;
			m_band = m_band + f * high;

			float sample = m_band * gain;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void WindAudioEffect::OnUpdate(float deltaTime, const AudioState& state) {
		float targetStrength = std::clamp(state.windStrength * 2.0f, 0.0f, 10.0f);
		m_currentWindStrength += (targetStrength - m_currentWindStrength) * deltaTime * 3.0f;

		float gain = std::clamp(m_currentWindStrength * 0.5f, 0.0f, 0.5f);
		float freq = 0.005f + std::clamp(m_currentWindStrength * 0.02f, 0.0f, 0.1f);
		float resonance = 0.5f + m_currentWindStrength * 0.2f;

		m_gain.store(gain, std::memory_order_relaxed);
		m_freq.store(freq, std::memory_order_relaxed);
		m_res.store(resonance, std::memory_order_relaxed);
	}

} // namespace brassica
