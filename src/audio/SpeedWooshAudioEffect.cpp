#include "audio/SpeedWooshAudioEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	SpeedWooshAudioEffect::SpeedWooshAudioEffect()
		: ProceduralAudioSource(2, 48000) {}

	float SpeedWooshAudioEffect::NextWhiteNoise() {
		m_xorshiftState ^= m_xorshiftState << 13;
		m_xorshiftState ^= m_xorshiftState >> 17;
		m_xorshiftState ^= m_xorshiftState << 5;
		return (static_cast<float>(m_xorshiftState) / static_cast<float>(0xFFFFFFFF)) * 2.0f - 1.0f;
	}

	void SpeedWooshAudioEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float gain = m_gain.load(std::memory_order_relaxed);
		float f = m_freq.load(std::memory_order_relaxed);
		float res = m_res.load(std::memory_order_relaxed);

		if (gain <= 0.0001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		float damping = 1.0f / (res + 0.001f);

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			float white = NextWhiteNoise();

			// SVF bandpass filter
			m_low = m_low + f * m_band;
			float high = white - m_low - damping * m_band;
			m_band = m_band + f * high;

			float sample = (m_band * 0.7f + m_low * 0.3f) * gain;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void SpeedWooshAudioEffect::OnUpdate(float deltaTime, const AudioState& state) {
		float minSpeed = 20.0f;
		float maxSpeed = 1000.0f;
		float speedFactor = std::clamp((state.speed - minSpeed) / (maxSpeed - minSpeed), 0.0f, 1.0f);

		m_currentSpeed += (state.speed - m_currentSpeed) * deltaTime * 4.0f;

		float gain = speedFactor * 0.5f;
		float freq = 0.01f + speedFactor * 0.14f;
		float resonance = 0.2f + speedFactor * 0.5f;

		m_gain.store(gain, std::memory_order_relaxed);
		m_freq.store(freq, std::memory_order_relaxed);
		m_res.store(resonance, std::memory_order_relaxed);
	}

} // namespace brassica
