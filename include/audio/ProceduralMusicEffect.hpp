#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	class ProceduralMusicEffect: public ProceduralAudioSource {
	public:
		ProceduralMusicEffect();
		~ProceduralMusicEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

	private:
		// Game thread -> Audio thread parameters
		std::atomic<float> m_tempoHz{4.0f};       // Steps per second
		std::atomic<float> m_chordRelease{0.999f}; // Higher = longer, spacey tails
		std::atomic<float> m_drumVolume{1.0f};
		std::atomic<float> m_drumBoom{0.0f};      // Pitch envelope intensity for kick
		std::atomic<bool>  m_isMinor{false};

		// Audio thread sequencing state
		float m_timeSinceLastStep{99999.0f};
		int   m_stepIndex{0};
		int   m_chordProgressionIndex{0};

		// Audio thread DSP state
		float m_kickPhase{0.0f};
		float m_kickEnv{0.0f};

		float m_chordPhases[3]{0.0f, 0.0f, 0.0f};
		float m_chordFreqs[3]{220.0f, 277.0f, 329.0f};
		float m_chordEnv{0.0f};
	};

} // namespace brassica
