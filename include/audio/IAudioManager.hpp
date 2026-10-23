#pragma once

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "audio/AudioTypes.hpp"
#include "IManager.hpp"

namespace brassica {

	class ProceduralAudioSource;
	class Sound;

	struct AudioManagerState {
		// Total Output Levels
		float masterVolume{1.0f};
		float musicVolume{1.0f};
		float sfxVolume{1.0f};
		float globalPitch{1.0f};

		// Wind Generator Aspects
		bool  enableWind{true};
		float windVolume{1.0f};
		float windPitch{1.0f};
		float windResonance{1.0f};

		// Rain Generator Aspects
		bool  enableRain{false};
		float rainVolume{1.0f};
		float rainPitch{1.0f};
		float rainDensity{1.0f};

		// Rustle Generator Aspects
		bool  enableRustle{false};
		float rustleVolume{1.0f};
		float rustlePitch{1.0f};
		float rustleModSpeed{1.0f};

		// Space Generator Aspects
		bool  enableSpaceNoise{true};
		float spaceVolume{1.0f};
		float spacePitch{1.0f};
		float spaceSubFreq{35.0f};

		// Speed Woosh Generator Aspects
		bool  enableSpeedWoosh{true};
		float speedWooshVolume{1.0f};
		float speedWooshPitch{1.0f};
		float speedWooshResonance{1.0f};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("masterVolume", "Master Volume", &AudioManagerState::masterVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("musicVolume", "Music Volume", &AudioManagerState::musicVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("sfxVolume", "SFX Volume", &AudioManagerState::sfxVolume, 0.0f, 1.0f, UIHint::Slider),
				MakeField("globalPitch", "Global Pitch", &AudioManagerState::globalPitch, 0.1f, 2.0f, UIHint::Slider),

				MakeField("enableWind", "Enable Wind Effect", &AudioManagerState::enableWind),
				MakeField("windVolume", "Wind Volume", &AudioManagerState::windVolume, 0.0f, 2.0f, UIHint::Slider),
				MakeField("windPitch", "Wind Pitch / Frequency", &AudioManagerState::windPitch, 0.1f, 3.0f, UIHint::Slider),
				MakeField("windResonance", "Wind Resonance", &AudioManagerState::windResonance, 0.1f, 3.0f, UIHint::Slider),

				MakeField("enableRain", "Enable Rain Effect", &AudioManagerState::enableRain),
				MakeField("rainVolume", "Rain Volume", &AudioManagerState::rainVolume, 0.0f, 2.0f, UIHint::Slider),
				MakeField("rainPitch", "Rain Pitch / Filter", &AudioManagerState::rainPitch, 0.1f, 3.0f, UIHint::Slider),
				MakeField("rainDensity", "Rain Drop Density", &AudioManagerState::rainDensity, 0.1f, 3.0f, UIHint::Slider),

				MakeField("enableRustle", "Enable Rustle Effect", &AudioManagerState::enableRustle),
				MakeField("rustleVolume", "Rustle Volume", &AudioManagerState::rustleVolume, 0.0f, 2.0f, UIHint::Slider),
				MakeField("rustlePitch", "Rustle Pitch / Cutoff", &AudioManagerState::rustlePitch, 0.1f, 3.0f, UIHint::Slider),
				MakeField("rustleModSpeed", "Rustle Modulation Speed", &AudioManagerState::rustleModSpeed, 0.1f, 3.0f, UIHint::Slider),

				MakeField("enableSpaceNoise", "Enable Space Noise Effect", &AudioManagerState::enableSpaceNoise),
				MakeField("spaceVolume", "Space Noise Volume", &AudioManagerState::spaceVolume, 0.0f, 2.0f, UIHint::Slider),
				MakeField("spacePitch", "Space Pitch / Cutoff", &AudioManagerState::spacePitch, 0.1f, 3.0f, UIHint::Slider),
				MakeField("spaceSubFreq", "Space Sub-Bass Frequency (Hz)", &AudioManagerState::spaceSubFreq, 10.0f, 120.0f, UIHint::Slider),

				MakeField("enableSpeedWoosh", "Enable Speed Woosh Effect", &AudioManagerState::enableSpeedWoosh),
				MakeField("speedWooshVolume", "Speed Woosh Volume", &AudioManagerState::speedWooshVolume, 0.0f, 2.0f, UIHint::Slider),
				MakeField("speedWooshPitch", "Speed Woosh Pitch / Sweep", &AudioManagerState::speedWooshPitch, 0.1f, 3.0f, UIHint::Slider),
				MakeField("speedWooshResonance", "Speed Woosh Resonance", &AudioManagerState::speedWooshResonance, 0.1f, 3.0f, UIHint::Slider)
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
