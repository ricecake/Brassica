#pragma once

#include "miniaudio.h"
#include "audio/AudioTypes.hpp"

namespace brassica {

	class ProceduralAudioSource {
	public:
		ProceduralAudioSource(ma_uint32 channels = 2, ma_uint32 sampleRate = 48000);
		virtual ~ProceduralAudioSource();

		// Non-copyable
		ProceduralAudioSource(const ProceduralAudioSource&) = delete;
		ProceduralAudioSource& operator=(const ProceduralAudioSource&) = delete;

		virtual void OnRead(float* pOutput, ma_uint64 frameCount) = 0;
		virtual void OnUpdate(float deltaTime, const AudioState& state) {}

		ma_data_source* GetDataSource() { return &m_base.ds; }

		ma_uint32 GetChannels() const { return m_channels; }

		ma_uint32 GetSampleRate() const { return m_sampleRate; }

		// Wrapper struct allowing retrieval of ProceduralAudioSource* from ma_data_source*
		struct MiniaudioSource {
			ma_data_source_base    ds;
			ProceduralAudioSource* pParent;
		};

	protected:
		MiniaudioSource m_base{};
		ma_uint32       m_channels;
		ma_uint32       m_sampleRate;
	};

} // namespace brassica
