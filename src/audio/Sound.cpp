#include "audio/Sound.hpp"

#include "spdlog/spdlog.h"
#include "audio/ProceduralAudioSource.hpp"

namespace brassica {

	Sound::Sound(
		ma_engine*         engine,
		const std::string& filepath,
		bool               loop,
		float              volume,
		bool               spatialized,
		const glm::vec3&   position,
		ma_sound_group*    group
	) {
		if (!engine) {
			spdlog::error("Sound created with null audio engine.");
			return;
		}

		ma_result result = ma_sound_init_from_file(engine, filepath.c_str(), 0, group, NULL, &_sound);
		if (result != MA_SUCCESS) {
			spdlog::error("Failed to load sound file: {}", filepath);
			return;
		}

		ma_sound_set_volume(&_sound, volume);
		ma_sound_set_looping(&_sound, loop ? MA_TRUE : MA_FALSE);
		ma_sound_set_spatialization_enabled(&_sound, spatialized ? MA_TRUE : MA_FALSE);
		if (spatialized) {
			ma_sound_set_position(&_sound, position.x, position.y, position.z);
		}

		ma_sound_start(&_sound);
		_initialized = true;
	}

	Sound::Sound(
		ma_engine*                             engine,
		std::shared_ptr<ProceduralAudioSource> source,
		bool                                   loop,
		float                                  volume,
		bool                                   spatialized,
		const glm::vec3&                       position,
		ma_sound_group*                        group
	)
		: _proceduralSource(source) {
		if (!engine) {
			spdlog::error("Procedural Sound created with null audio engine.");
			return;
		}
		if (!source) {
			spdlog::error("Procedural Sound created with null source.");
			return;
		}

		ma_result result = ma_sound_init_from_data_source(engine, source->GetDataSource(), 0, group, &_sound);
		if (result != MA_SUCCESS) {
			spdlog::error("Failed to initialize procedural sound from data source.");
			return;
		}

		ma_sound_set_volume(&_sound, volume);
		ma_sound_set_looping(&_sound, loop ? MA_TRUE : MA_FALSE);
		ma_sound_set_spatialization_enabled(&_sound, spatialized ? MA_TRUE : MA_FALSE);
		if (spatialized) {
			ma_sound_set_position(&_sound, position.x, position.y, position.z);
		}

		ma_sound_start(&_sound);
		_initialized = true;
	}

	Sound::~Sound() {
		if (_initialized) {
			ma_sound_uninit(&_sound);
		}
	}

	void Sound::SetPosition(const glm::vec3& position) {
		if (_initialized) {
			ma_sound_set_position(&_sound, position.x, position.y, position.z);
		}
	}

	void Sound::SetVolume(float volume) {
		if (_initialized) {
			ma_sound_set_volume(&_sound, volume);
		}
	}

	void Sound::SetPitch(float pitch) {
		if (_initialized) {
			ma_sound_set_pitch(&_sound, pitch);
		}
	}

	void Sound::SetLooping(bool loop) {
		if (_initialized) {
			ma_sound_set_looping(&_sound, loop ? MA_TRUE : MA_FALSE);
		}
	}

	bool Sound::IsDone() {
		if (!_initialized) {
			return true;
		}
		return ma_sound_at_end(&_sound) ? true : false;
	}

	void Sound::Stop() {
		if (_initialized) {
			ma_sound_stop(&_sound);
		}
	}

} // namespace brassica
