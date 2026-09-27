#include "audio/SpaceAudioEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	SpaceAudioEffect::SpaceAudioEffect()
		: ProceduralAudioSource(2, 48000) {}

	float SpaceAudioEffect::NextWhiteNoise() {
		m_xorshiftState ^= m_xorshiftState << 13;
		m_xorshiftState ^= m_xorshiftState >> 17;
		m_xorshiftState ^= m_xorshiftState << 5;
		return (static_cast<float>(m_xorshiftState) / static_cast<float>(0xFFFFFFFF)) * 2.0f - 1.0f;
	}

	void SpaceAudioEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float gain = m_gain.load(std::memory_order_relaxed);
		float alpha = m_cutoffAlpha.load(std::memory_order_relaxed);

		if (gain <= 0.0001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		const float dt = 1.0f / static_cast<float>(m_sampleRate);

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			float white = NextWhiteNoise();

			// Muffled low-pass filtering (sub-bass / drone)
			m_lowPass = m_lowPass + alpha * (white - m_lowPass);

			// Add a subtle low-frequency sub-oscillation (~35Hz sub-bass)
			m_subPhase += 2.0f * 3.14159265f * 35.0f * dt;
			if (m_subPhase > 2.0f * 3.14159265f) {
				m_subPhase -= 2.0f * 3.14159265f;
			}
			float subSine = std::sin(m_subPhase) * 0.15f;

			float sample = (m_lowPass * 0.6f + subSine) * gain;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void SpaceAudioEffect::OnUpdate(float deltaTime, const AudioState& state) {
		float minAlt = 1000.0f;
		float maxAlt = 50000.0f;
		float targetSpaceFactor = std::clamp((state.altitude - minAlt) / (maxAlt - minAlt), 0.0f, 1.0f);

		m_currentAltitude += (state.altitude - m_currentAltitude) * deltaTime * 2.0f;

		float gain = targetSpaceFactor * 0.4f;
		float alpha = 0.001f + targetSpaceFactor * 0.007f;

		m_gain.store(gain, std::memory_order_relaxed);
		m_cutoffAlpha.store(alpha, std::memory_order_relaxed);
	}

} // namespace brassica
