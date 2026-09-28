#include "audio/RainAudioEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	RainAudioEffect::RainAudioEffect()
		: ProceduralAudioSource(2, 48000) {}

	float RainAudioEffect::NextWhiteNoise() {
		m_xorshiftState ^= m_xorshiftState << 13;
		m_xorshiftState ^= m_xorshiftState >> 17;
		m_xorshiftState ^= m_xorshiftState << 5;
		return (static_cast<float>(m_xorshiftState) / static_cast<float>(0xFFFFFFFF)) * 2.0f - 1.0f;
	}

	void RainAudioEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float intensity = m_intensity.load(std::memory_order_relaxed);
		if (intensity <= 0.001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		float washAlpha = 0.01f;
		float patterAlpha = 0.07f;

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			float white = NextWhiteNoise();

			m_lowWash = m_lowWash + washAlpha * (white - m_lowWash);
			float wash = m_lowWash * 0.05f * intensity;

			float drop = 0.0f;
			float dropChance = 0.0005f + intensity * 0.01f;

			if ((static_cast<float>(m_xorshiftState & 0xFFFF) / 65535.0f) < dropChance) {
				drop = white * 0.2f * intensity;
			}

			m_lowPatter = m_lowPatter + patterAlpha * (drop - m_lowPatter);
			float sample = wash + m_lowPatter;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void RainAudioEffect::OnUpdate(float deltaTime, const AudioState& state) {
		float current = m_intensity.load(std::memory_order_relaxed);
		float target = std::clamp(state.rainIntensity, 0.0f, 1.0f);
		float next = current + (target - current) * deltaTime * 2.0f;
		m_intensity.store(next, std::memory_order_relaxed);
	}

} // namespace brassica
