#include "audio/ProceduralMusicEffect.hpp"

#include <algorithm>
#include <cmath>

namespace brassica {

	ProceduralMusicEffect::ProceduralMusicEffect(): ProceduralAudioSource(2, 48000) {}

	void ProceduralMusicEffect::OnRead(float* pOutput, ma_uint64 frameCount) {
		float tempoHz = m_tempoHz.load(std::memory_order_relaxed);
		float chordRelease = m_chordRelease.load(std::memory_order_relaxed);
		float drumVol = m_drumVolume.load(std::memory_order_relaxed);
		float drumBoom = m_drumBoom.load(std::memory_order_relaxed);
		bool  isMinor = m_isMinor.load(std::memory_order_relaxed);

		float samplesPerStep = static_cast<float>(m_sampleRate) / tempoHz;
		constexpr float twoPi = 2.0f * 3.14159265359f;

		for (ma_uint64 i = 0; i < frameCount; ++i) {
			m_timeSinceLastStep += 1.0f;

			// 16-Step Sequencer Logic
			if (m_timeSinceLastStep >= samplesPerStep) {
				m_timeSinceLastStep = 0.0f;

				// Four-on-the-floor kick drum pattern (steps 0, 4, 8, 12)
				if (m_stepIndex % 4 == 0) {
					m_kickEnv = 1.0f;
					m_kickPhase = 0.0f; // Reset phase for punchy transient
				}

				// Chords hit on step 0 and 8 (half notes)
				if (m_stepIndex % 8 == 0) {
					m_chordEnv = 1.0f;

					// Basic I-V-vi-IV progression logic
					int rootMidi = 57; // A3

					// Swap progressions based on rain mood
					int progressions[2][4] = {
						{0, 7, 9, 5}, // Major (A, E, F#m, D)
						{0, 5, 8, 7}  // Minor (Am, Dm, F, E)
					};

					int chordOffsets[2][3] = {
						{0, 4, 7}, // Major Triad
						{0, 3, 7}  // Minor Triad
					};

					int progRoot = progressions[isMinor ? 1 : 0][m_chordProgressionIndex];

					// Determine if the current chord in the progression is naturally minor or major
					// (Simplified logic: in the minor progression, 3rd chord (F) is major. In major, 3rd (F#m) is minor)
					bool chordIsMinor = isMinor;
					if (!isMinor && m_chordProgressionIndex == 2) chordIsMinor = true;
					if (isMinor && m_chordProgressionIndex == 2) chordIsMinor = false;
					if (isMinor && m_chordProgressionIndex == 3) chordIsMinor = false; // Harmonic minor V

					for (int v = 0; v < 3; ++v) {
						int semitone = rootMidi + progRoot + chordOffsets[chordIsMinor ? 1 : 0][v];
						m_chordFreqs[v] = 440.0f * std::pow(2.0f, (semitone - 69) / 12.0f);
					}

					m_chordProgressionIndex = (m_chordProgressionIndex + 1) % 4;
				}

				m_stepIndex = (m_stepIndex + 1) % 16;
			}

			// --- DSP Generation ---

			// 1. Kick Drum (Sine wave with rapid pitch and amplitude decay)
			// Higher drumBoom increases the starting frequency of the pitch envelope
			m_kickEnv *= (0.999f + (drumBoom * 0.0005f));
			float kickPitchEnv = m_kickEnv * m_kickEnv * m_kickEnv;
			float kickFreq = 50.0f + (kickPitchEnv * (100.0f + drumBoom * 200.0f));

			m_kickPhase += kickFreq * twoPi / m_sampleRate;
			if (m_kickPhase >= twoPi) m_kickPhase -= twoPi;
			float kickSample = std::sin(m_kickPhase) * m_kickEnv * drumVol;

			// 2. Chords (Additive sine waves)
			m_chordEnv *= chordRelease;
			float chordSample = 0.0f;

			for (int v = 0; v < 3; ++v) {
				m_chordPhases[v] += m_chordFreqs[v] * twoPi / m_sampleRate;
				if (m_chordPhases[v] >= twoPi) m_chordPhases[v] -= twoPi;

				// Gentle FM mod on chords to make them sound like electric pianos/bells
				float mod = std::sin(m_chordPhases[v] * 2.0f) * 0.5f * m_chordEnv;
				chordSample += std::sin(m_chordPhases[v] + mod) * m_chordEnv * 0.2f;
			}

			// 3. Mix
			float finalSample = kickSample + chordSample;

			for (ma_uint32 c = 0; c < m_channels; ++c) {
				pOutput[i * m_channels + c] = finalSample;
			}
		}
	}

	void ProceduralMusicEffect::OnUpdate(float deltaTime, const AudioState& state) {
		// Rain: 0.0 -> 1.0
		// Triggers minor scale at high rain, increases kick boominess
		bool isMinor = state.rainIntensity > 0.5f;
		float drumBoom = std::clamp(state.rainIntensity, 0.0f, 1.0f);

		// Altitude mappings
		// Fade out drums between 10k and 15k
		float drumVol = 1.0f - std::clamp((state.altitude - 10000.0f) / 5000.0f, 0.0f, 1.0f);

		// Altitude 0 -> 32k mapping
		float altFactor = std::clamp(state.altitude / 25000.0f, 0.0f, 1.0f);

		// Base tempo modified by camera speed, slowed down significantly at high altitudes
		float speedFactor = std::clamp((state.speed - 5.0f) / 95.0f, 0.0f, 1.0f);
		float baseTempoHz = 4.0f + (speedFactor * 4.0f); // 4 to 8 steps per second
		float tempoHz = baseTempoHz * (1.0f - (altFactor * 0.86f)); // Drops by 60% at 32k

		// Lengthen the release curve of the chords for spacey bleeding
		// 0.999 decays normally, 0.9999 is a massive ambient wash
		float chordRelease = 0.999f + (altFactor * 0.00099f);

		m_tempoHz.store(tempoHz, std::memory_order_relaxed);
		m_chordRelease.store(chordRelease, std::memory_order_relaxed);
		m_drumVolume.store(drumVol, std::memory_order_relaxed);
		m_drumBoom.store(drumBoom, std::memory_order_relaxed);
		m_isMinor.store(isMinor, std::memory_order_relaxed);
	}

} // namespace brassica
