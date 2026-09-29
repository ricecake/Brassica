#pragma once

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include "audio/IAudioManager.hpp"

namespace brassica {

	class Sound;
	class ProceduralAudioSource;
	class WindAudioEffect;
	class RainAudioEffect;
	class RustleAudioEffect;
	class SpaceAudioEffect;
	class SpeedWooshAudioEffect;

	class AudioManager: public IAudioManager {
	public:
		AudioManager();
		~AudioManager() override;

		// IManager methods
		void Initialize() override;
		void Shutdown() override;

		State GetState() const override;
		void  SetState(const State& state) override;

		// IAudioManager methods
		void UpdateState(const AudioState& state) override;
		void Update(float deltaTime) override;

		void PlayMusic(const std::string& filepath, bool loop = true, float volume = 1.0f) override;
		void PlayAmbientSound(
			const std::string& name,
			const std::string& filepath,
			bool               loop = true,
			float              volume = 1.0f
		) override;
		void StopAmbientSound(const std::string& name) override;
		void SetAmbientSoundVolume(const std::string& name, float volume) override;

		std::shared_ptr<Sound> CreateSound(
			const std::string& filepath,
			const glm::vec3&   position = glm::vec3(0.0f),
			float              volume = 1.0f,
			bool               loop = false
		) override;

		std::shared_ptr<Sound> CreateProceduralSound(
			std::shared_ptr<ProceduralAudioSource> source,
			const glm::vec3&                       position = glm::vec3(0.0f),
			float                                  volume = 1.0f,
			bool                                   loop = true,
			bool                                   spatialized = false
		) override;

		void SetGlobalPitch(float pitch) override;

		float GetMasterVolume() const override;
		void  SetMasterVolume(float volume) override;
		float GetMusicVolume() const override;
		void  SetMusicVolume(float volume) override;
		float GetSfxVolume() const override;
		void  SetSfxVolume(float volume) override;

		AudioState GetCurrentState() const override;

		void StopAllSounds() override;

	private:
		struct AudioManagerImpl;
		std::unique_ptr<AudioManagerImpl> m_pimpl;
	};

} // namespace brassica
