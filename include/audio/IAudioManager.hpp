#pragma once

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "audio/AudioTypes.hpp"

namespace brassica {

	class ProceduralAudioSource;
	class Sound;

	struct AudioManagerState {
		float masterVolume{1.0f};
		float musicVolume{1.0f};
		float sfxVolume{1.0f};
		float globalPitch{1.0f};

		bool enableWind{true};
		bool enableRain{false};
		bool enableRustle{false};
		bool enableSpaceNoise{true};
		bool enableSpeedWoosh{true};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("masterVolume", "Master Volume", &AudioManagerState::masterVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("musicVolume", "Music Volume", &AudioManagerState::musicVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("sfxVolume", "SFX Volume", &AudioManagerState::sfxVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("globalPitch", "Global Pitch", &AudioManagerState::globalPitch, 0.1f, 2.0f, UIHint::Slider),
				MakeField("enableWind", "Enable Wind Effect", &AudioManagerState::enableWind),
				MakeField("enableRain", "Enable Rain Effect", &AudioManagerState::enableRain),
				MakeField("enableRustle", "Enable Rustle Effect", &AudioManagerState::enableRustle),
				MakeField("enableSpaceNoise", "Enable Space Noise Effect", &AudioManagerState::enableSpaceNoise),
				MakeField("enableSpeedWoosh", "Enable Speed Woosh Effect", &AudioManagerState::enableSpeedWoosh)
			);
		}
	};

	class IAudioManager: public ManagerBase<IAudioManager, AudioManagerState> {
	public:
		using State = AudioManagerState;

		~IAudioManager() override = default;

		std::string GetManagerName() const override { return "AudioManager"; }

		virtual void UpdateState(const AudioState& state) = 0;
		virtual void Update(float deltaTime) = 0;

		virtual void PlayMusic(const std::string& filepath, bool loop = true, float volume = 1.0f) = 0;
		virtual void PlayAmbientSound(
			const std::string& name,
			const std::string& filepath,
			bool               loop = true,
			float              volume = 1.0f
		) = 0;
		virtual void StopAmbientSound(const std::string& name) = 0;
		virtual void SetAmbientSoundVolume(const std::string& name, float volume) = 0;

		virtual std::shared_ptr<Sound> CreateSound(
			const std::string& filepath,
			const glm::vec3&   position = glm::vec3(0.0f),
			float              volume = 1.0f,
			bool               loop = false
		) = 0;

		virtual std::shared_ptr<Sound> CreateProceduralSound(
			std::shared_ptr<ProceduralAudioSource> source,
			const glm::vec3&                       position = glm::vec3(0.0f),
			float                                  volume = 1.0f,
			bool                                   loop = true,
			bool                                   spatialized = false
		) = 0;

		virtual void SetGlobalPitch(float pitch) = 0;

		virtual float GetMasterVolume() const = 0;
		virtual void  SetMasterVolume(float volume) = 0;
		virtual float GetMusicVolume() const = 0;
		virtual void  SetMusicVolume(float volume) = 0;
		virtual float GetSfxVolume() const = 0;
		virtual void  SetSfxVolume(float volume) = 0;

		virtual AudioState GetCurrentState() const = 0;

		virtual void StopAllSounds() = 0;
	};

} // namespace brassica
