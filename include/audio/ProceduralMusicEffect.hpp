#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	/**
	 * @brief Procedural music generator that reacts to speed and mood.
	 *
	 * Driven by state speed to control tempo and volume, and a mood parameter
	 * to switch between Major and Minor pentatonic scales.
	 */
	class ProceduralMusicEffect: public ProceduralAudioSource {
	public:
		ProceduralMusicEffect();
		~ProceduralMusicEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

		void  SetMoodScale(float mood) { m_moodScale.store(mood, std::memory_order_relaxed); }
		float GetMoodScale() const { return m_moodScale.load(std::memory_order_relaxed); }

	private:
		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_speed{1.0f};
		std::atomic<float> m_moodScale{1.0f};

		// Audio and sequencing state (only mutated inside the OnRead thread)
		float m_phase{0.0f};
		float m_currentFreq{220.0f};
		float m_envelope{0.0f};
		float m_timeSinceLastNote{99999.0f}; // Set high to trigger note immediately
		int   m_noteIndex{0};
	};

} // namespace brassica