#pragma once

#include <atomic>
#include <cstdint>

#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	class RainAudioEffect: public ProceduralAudioSource {
	public:
		RainAudioEffect();
		~RainAudioEffect() override = default;

		void OnRead(float* pOutput, ma_uint64 frameCount) override;
		void OnUpdate(float deltaTime, const AudioState& state) override;

		void  SetVolumeScale(float volume) { m_volumeScale.store(volume, std::memory_order_relaxed); }
		float GetVolumeScale() const { return m_volumeScale.load(std::memory_order_relaxed); }

		void  SetPitchScale(float pitch) { m_pitchScale.store(pitch, std::memory_order_relaxed); }
		float GetPitchScale() const { return m_pitchScale.load(std::memory_order_relaxed); }

		void  SetDensityScale(float density) { m_densityScale.store(density, std::memory_order_relaxed); }
		float GetDensityScale() const { return m_densityScale.load(std::memory_order_relaxed); }

	private:
		uint32_t m_xorshiftState{987654321};
		float    NextWhiteNoise();

		std::atomic<float> m_intensity{0.0f};

		std::atomic<float> m_volumeScale{1.0f};
		std::atomic<float> m_pitchScale{1.0f};
		std::atomic<float> m_densityScale{1.0f};

		// Filter state
		float m_lowWash{0.0f};
		float m_lowPatter{0.0f};
	};

} // namespace brassica
