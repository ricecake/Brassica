#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	/**
	 * @brief Procedural altitude-driven muffled space noise.
	 *
	 * As camera altitude increases into upper atmosphere and space (1,000m to 50,000m+),
	 * atmospheric wind fades into a low-frequency, heavily muffled sub-bass cosmic rumble.
	 */
	class SpaceAudioEffect: public ProceduralAudioSource {
	public:
		SpaceAudioEffect();
		~SpaceAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

		void  SetVolumeScale(float volume) { m_volumeScale.store(volume, std::memory_order_relaxed); }
		float GetVolumeScale() const { return m_volumeScale.load(std::memory_order_relaxed); }

		void  SetPitchScale(float pitch) { m_pitchScale.store(pitch, std::memory_order_relaxed); }
		float GetPitchScale() const { return m_pitchScale.load(std::memory_order_relaxed); }

		void  SetSubBassFreq(float freq) { m_subBassFreq.store(freq, std::memory_order_relaxed); }
		float GetSubBassFreq() const { return m_subBassFreq.load(std::memory_order_relaxed); }

	private:
		uint32_t m_xorshiftState{246801357};
		float    NextWhiteNoise();

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_cutoffAlpha{0.002f};

		std::atomic<float> m_volumeScale{1.0f};
		std::atomic<float> m_pitchScale{1.0f};
		std::atomic<float> m_subBassFreq{35.0f};

		// Filter and phase state
		float m_lowPass{0.0f};
		float m_subPhase{0.0f};
		float m_currentAltitude{0.0f};
	};

} // namespace brassica
