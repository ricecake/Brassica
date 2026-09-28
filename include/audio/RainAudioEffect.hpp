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

	private:
		uint32_t m_xorshiftState{987654321};
		float    NextWhiteNoise();

		std::atomic<float> m_intensity{0.0f};

		// Filter state
		float m_lowWash{0.0f};
		float m_lowPatter{0.0f};
	};

} // namespace brassica
