#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	class RustleAudioEffect: public ProceduralAudioSource {
	public:
		RustleAudioEffect();
		~RustleAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

		void  SetVolumeScale(float volume) { m_volumeScale.store(volume, std::memory_order_relaxed); }
		float GetVolumeScale() const { return m_volumeScale.load(std::memory_order_relaxed); }

		void  SetPitchScale(float pitch) { m_pitchScale.store(pitch, std::memory_order_relaxed); }
		float GetPitchScale() const { return m_pitchScale.load(std::memory_order_relaxed); }

		void  SetModSpeedScale(float speed) { m_modSpeedScale.store(speed, std::memory_order_relaxed); }
		float GetModSpeedScale() const { return m_modSpeedScale.load(std::memory_order_relaxed); }

	private:
		uint32_t m_xorshiftState{135792468};
		float    NextWhiteNoise();
		float    m_timeStep{0.0f};

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_lowPassAlpha{0.1f};

		std::atomic<float> m_volumeScale{1.0f};
		std::atomic<float> m_pitchScale{1.0f};
		std::atomic<float> m_modSpeedScale{1.0f};

		// Filter state
		float m_low{0.0f};
	};

} // namespace brassica
