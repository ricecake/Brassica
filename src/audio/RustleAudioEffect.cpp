#include "audio/RustleAudioEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	RustleAudioEffect::RustleAudioEffect()
		: ProceduralAudioSource(2, 48000) {}

	float RustleAudioEffect::NextWhiteNoise() {
		m_xorshiftState ^= m_xorshiftState << 13;
		m_xorshiftState ^= m_xorshiftState >> 17;
		m_xorshiftState ^= m_xorshiftState << 5;
		return (static_cast<float>(m_xorshiftState) / static_cast<float>(0xFFFFFFFF)) * 2.0f - 1.0f;
	}

	void RustleAudioEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float gain = m_gain.load(std::memory_order_relaxed);
		float alpha = m_lowPassAlpha.load(std::memory_order_relaxed);

		if (gain <= 0.0001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			float white = NextWhiteNoise();

			m_low = m_low + alpha * (white - m_low);
			float highPass = white - m_low;

			float sample = highPass * gain;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void RustleAudioEffect::OnUpdate(float deltaTime, const AudioState& state) {
		float strength = std::clamp(state.windStrength, 0.0f, 1.0f);
		float targetGain = strength * state.grassDensity * 0.25f;
		m_timeStep += deltaTime;

		float mod = std::sin(m_timeStep * 3.0f) * 0.25f + 0.75f;

		float currentGain = m_gain.load(std::memory_order_relaxed);
		float nextGain = currentGain + (targetGain - currentGain) * deltaTime * 2.0f * mod;
		m_gain.store(nextGain, std::memory_order_relaxed);

		float targetAlpha = 0.02f + strength * 0.1f;
		float currentAlpha = m_lowPassAlpha.load(std::memory_order_relaxed);
		float nextAlpha = currentAlpha + (targetAlpha - currentAlpha) * deltaTime * 1.0f;
		m_lowPassAlpha.store(nextAlpha, std::memory_order_relaxed);
	}

} // namespace brassica
