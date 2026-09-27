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

	private:
		uint32_t m_xorshiftState{135792468};
		float    NextWhiteNoise();
		float    m_timeStep{0.0f};

		std::atomic<float> m_gain{0.0f};
		std::atomic<float> m_lowPassAlpha{0.1f};

		// Filter state
		float m_low{0.0f};
	};

} // namespace brassica
