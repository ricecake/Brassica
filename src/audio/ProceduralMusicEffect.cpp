#include "audio/ProceduralMusicEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	ProceduralMusicEffect::ProceduralMusicEffect(): ProceduralAudioSource(2, 48000) {}

	void ProceduralMusicEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float gain = m_gain.load(std::memory_order_relaxed);
		float speed = m_speed.load(std::memory_order_relaxed);
		float mood = m_moodScale.load(std::memory_order_relaxed);

		// Early exit for silence, saving processing time
		if (gain <= 0.0001f) {
			for (ma_uint64 i = 0; i < frameCount * m_channels; ++i) {
				pOutput[i] = 0.0f;
			}
			return;
		}

		// Calculate how many samples long each note should be based on speed
		float noteDurationSamples = static_cast<float>(m_sampleRate) / (speed * 4.0f + 0.1f);
		float twoPi = 2.0f * 3.14159265359f;

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			m_timeSinceLastNote += 1.0f;

			// Arpeggiator sequence logic triggered by sample count
			if (m_timeSinceLastNote >= noteDurationSamples) {
				m_timeSinceLastNote = 0.0f;
				m_envelope = 1.0f; // Strike the note

				// Advance a 5-note pattern
				m_noteIndex = (m_noteIndex + 1) % 5;

				// Pentatonic scale semitone offsets
				int majorPentatonic[] = {0, 2, 4, 7, 9};
				int minorPentatonic[] = {0, 3, 5, 7, 10};

				int semitoneOffset = (mood >= 0.5f) ? majorPentatonic[m_noteIndex] : minorPentatonic[m_noteIndex];

				// Calculate frequency from a root of A3 (220 Hz)
				m_currentFreq = 220.0f * std::pow(2.0f, static_cast<float>(semitoneOffset) / 12.0f);
			}

			// Apply an exponential decay to the envelope for a "plucked" sound
			m_envelope *= 0.9998f;

			// Accumulate oscillator phase
			float phaseInc = m_currentFreq * twoPi / m_sampleRate;
			m_phase += phaseInc;
			if (m_phase >= twoPi) {
				m_phase -= twoPi;
			}

			// Generate a simple FM synthesis pluck (modulating a sine with another sine)
			float sample = std::sin(m_phase + std::sin(m_phase * 2.0f) * m_envelope) * m_envelope * gain;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = sample;
			}
		}
	}

	void ProceduralMusicEffect::OnUpdate(float deltaTime, const AudioState& state) {
		// Map the state's movement speed to an internal tempo multiplier
		float minSpeed = 5.0f;
		float maxSpeed = 100.0f;
		float speedFactor = std::clamp((state.speed - minSpeed) / (maxSpeed - minSpeed), 0.0f, 1.0f);

		// Calculate tempo (from 0.5x to 5.0x speed)
		float tempo = 0.5f + speedFactor * 4.5f;

		// Smoothly fade volume in as speed increases
		float gain = speedFactor * 0.4f;

		m_speed.store(tempo, std::memory_order_relaxed);
		m_gain.store(gain, std::memory_order_relaxed);
	}

} // namespace brassica