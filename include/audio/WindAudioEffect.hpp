#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	class WindAudioEffect: public ProceduralAudioSource {
	public:
		WindAudioEffect();
		~WindAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

		void  SetVolumeScale(float volume) { m_volumeScale.store(volume, std::memory_order_relaxed); }
		float GetVolumeScale() const { return m_volumeScale.load(std::memory_order_relaxed); }

		void  SetPitchScale(float pitch) { m_pitchScale.store(pitch, std::memory_order_relaxed); }
		float GetPitchScale() const { return m_pitchScale.load(std::memory_order_relaxed); }

		void  SetResonanceScale(float resonance) { m_resonanceScale.store(resonance, std::memory_order_relaxed); }
		float GetResonanceScale() const { return m_resonanceScale.load(std::memory_order_relaxed); }

	private:
		uint32_t m_xorshiftState{123456789};
		float    NextWhiteNoise();

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_freq{0.01f};
		std::atomic<float> m_res{0.1f};

		std::atomic<float> m_volumeScale{1.0f};
		std::atomic<float> m_pitchScale{1.0f};
		std::atomic<float> m_resonanceScale{1.0f};

		float m_currentWindStrength{0.0f};

		// State Variable Filter state
		float m_low{0.0f};
		float m_band{0.0f};
	};

} // namespace brassica
