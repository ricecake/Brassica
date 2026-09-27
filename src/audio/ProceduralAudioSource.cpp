#include "audio/ProceduralAudioSource.hpp"

#include <cstring>

namespace brassica {

	static ma_result ProceduralAudioSource_Read(
		ma_data_source* pDataSource,
		void*           pFramesOut,
		ma_uint64       frameCount,
		ma_uint64*      pFramesRead
	) {
		auto* pMiniaudioSource = reinterpret_cast<ProceduralAudioSource::MiniaudioSource*>(pDataSource);
		if (pMiniaudioSource && pMiniaudioSource->pParent) {
			pMiniaudioSource->pParent->OnRead(static_cast<float*>(pFramesOut), frameCount);
		}
		if (pFramesRead) {
			*pFramesRead = frameCount;
		}
		return MA_SUCCESS;
	}

	static ma_result ProceduralAudioSource_GetDataFormat(
		ma_data_source* pDataSource,
		ma_format*      pFormat,
		ma_uint32*      pChannels,
		ma_uint32*      pSampleRate,
		ma_channel*     pChannelMap,
		size_t          channelMapCap
	) {
		auto* pMiniaudioSource = reinterpret_cast<ProceduralAudioSource::MiniaudioSource*>(pDataSource);
		if (!pMiniaudioSource || !pMiniaudioSource->pParent) {
			return MA_INVALID_ARGS;
		}

		auto* pProcedural = pMiniaudioSource->pParent;
		if (pFormat) {
			*pFormat = ma_format_f32;
		}
		if (pChannels) {
			*pChannels = pProcedural->GetChannels();
		}
		if (pSampleRate) {
			*pSampleRate = pProcedural->GetSampleRate();
		}
		if (pChannelMap) {
			ma_channel_map_init_standard(
				ma_standard_channel_map_default,
				pChannelMap,
				channelMapCap,
				pProcedural->GetChannels()
			);
		}
		return MA_SUCCESS;
	}

	static ma_data_source_vtable g_ProceduralAudioSourceVTable = {
		ProceduralAudioSource_Read,
		nullptr, // onSeek
		ProceduralAudioSource_GetDataFormat,
		nullptr, // onGetCursor
		nullptr, // onGetLength
		nullptr, // onSetLooping
		0        // flags
	};

	ProceduralAudioSource::ProceduralAudioSource(ma_uint32 channels, ma_uint32 sampleRate)
		: m_channels(channels), m_sampleRate(sampleRate) {
		m_base.pParent = this;
		ma_data_source_config config = ma_data_source_config_init();
		config.vtable = &g_ProceduralAudioSourceVTable;
		ma_data_source_init(&config, &m_base.ds);
	}

	ProceduralAudioSource::~ProceduralAudioSource() {
		ma_data_source_uninit(&m_base.ds);
	}

} // namespace brassica
