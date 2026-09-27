#include "audio/AudioManager.hpp"

#include <algorithm>
#include <chrono>
#include <list>
#include <map>
#include <mutex>
#include <thread>

#include "spdlog/spdlog.h"
#include "audio/ProceduralAudioSource.hpp"
#include "audio/RainAudioEffect.hpp"
#include "audio/RustleAudioEffect.hpp"
#include "audio/Sound.hpp"
#include "audio/SpaceAudioEffect.hpp"
#include "audio/SpeedWooshAudioEffect.hpp"
#include "audio/WindAudioEffect.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace brassica {

	struct AudioManager::AudioManagerImpl {
		ma_engine      engine;
		ma_sound_group masterGroup;
		ma_sound_group musicGroup;
		ma_sound_group sfxGroup;

		bool initialized{false};
		bool groupsInitialized{false};

		std::list<std::weak_ptr<Sound>>                 sounds;
		std::shared_ptr<Sound>                          music;
		std::map<std::string, std::shared_ptr<Sound>>   ambientSounds;
		std::list<std::weak_ptr<ProceduralAudioSource>> proceduralSources;

		std::shared_ptr<WindAudioEffect>       windEffect;
		std::shared_ptr<RainAudioEffect>       rainEffect;
		std::shared_ptr<RustleAudioEffect>     rustleEffect;
		std::shared_ptr<SpaceAudioEffect>      spaceEffect;
		std::shared_ptr<SpeedWooshAudioEffect> speedWooshEffect;

		std::shared_ptr<Sound> windSound;
		std::shared_ptr<Sound> rainSound;
		std::shared_ptr<Sound> rustleSound;
		std::shared_ptr<Sound> spaceSound;
		std::shared_ptr<Sound> speedWooshSound;

		std::mutex m_soundsMutex;
		std::mutex m_stateMutex;
		AudioState m_currentState;
		State      m_managerState;

		std::thread       m_audioThread;
		std::atomic<bool> m_running{false};

		void InitEngine() {
			ma_engine_config engineConfig = ma_engine_config_init();
			engineConfig.channels = 2;
			engineConfig.sampleRate = 48000;

			ma_result result = ma_engine_init(&engineConfig, &engine);
			if (result != MA_SUCCESS) {
				spdlog::error("Failed to initialize miniaudio engine.");
				initialized = false;
				return;
			}
			initialized = true;

			ma_sound_group_init(&engine, 0, NULL, &masterGroup);
			ma_sound_group_init(&engine, 0, &masterGroup, &musicGroup);
			ma_sound_group_init(&engine, 0, &masterGroup, &sfxGroup);
			groupsInitialized = true;

			m_running = true;
			m_audioThread = std::thread(&AudioManagerImpl::AudioLoop, this);
		}

		void ShutdownEngine() {
			m_running = false;
			if (m_audioThread.joinable()) {
				m_audioThread.join();
			}

			windSound.reset();
			rainSound.reset();
			rustleSound.reset();
			spaceSound.reset();
			speedWooshSound.reset();

			windEffect.reset();
			rainEffect.reset();
			rustleEffect.reset();
			spaceEffect.reset();
			speedWooshEffect.reset();

			music.reset();
			ambientSounds.clear();
			sounds.clear();
			proceduralSources.clear();

			if (groupsInitialized) {
				ma_sound_group_uninit(&sfxGroup);
				ma_sound_group_uninit(&musicGroup);
				ma_sound_group_uninit(&masterGroup);
				groupsInitialized = false;
			}

			if (initialized) {
				ma_engine_uninit(&engine);
				initialized = false;
			}
		}

		void AudioLoop() {
			auto lastTime = std::chrono::high_resolution_clock::now();

			while (m_running) {
				auto  currentTime = std::chrono::high_resolution_clock::now();
				float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
				lastTime = currentTime;

				UpdateLoop(deltaTime);

				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}

		void UpdateLoop(float deltaTime) {
			if (!initialized)
				return;

			AudioState state;
			State      mgrState;
			{
				std::lock_guard<std::mutex> lock(m_stateMutex);
				state = m_currentState;
				mgrState = m_managerState;
			}

			// Update Listener
			auto vel = state.listenerSpeed * state.listenerFront;
			ma_engine_listener_set_position(
				&engine,
				0,
				state.listenerPos.x,
				state.listenerPos.y,
				state.listenerPos.z
			);
			ma_engine_listener_set_direction(
				&engine,
				0,
				state.listenerFront.x,
				state.listenerFront.y,
				state.listenerFront.z
			);
			ma_engine_listener_set_world_up(&engine, 0, state.listenerUp.x, state.listenerUp.y, state.listenerUp.z);
			ma_engine_listener_set_velocity(&engine, 0, vel.x, vel.y, vel.z);
			ma_engine_listener_set_cone(
				&engine,
				0,
				glm::radians(state.listenerFov),
				glm::radians(state.listenerFov) * 2.0f,
				0.75f
			);

			// Update Volumes/Pitch
			if (groupsInitialized) {
				ma_sound_group_set_volume(&masterGroup, mgrState.masterVolume * state.masterVolume);
				ma_sound_group_set_volume(&musicGroup, mgrState.musicVolume * state.musicVolume);
				ma_sound_group_set_volume(&sfxGroup, mgrState.sfxVolume * state.sfxVolume);
				ma_sound_group_set_pitch(&masterGroup, mgrState.globalPitch * state.globalPitch);
			}

			// Enable / disable procedural sound effects based on manager state
			if (windSound) {
				windSound->SetVolume(mgrState.enableWind ? 1.0f : 0.0f);
			}
			if (rainSound) {
				rainSound->SetVolume(mgrState.enableRain ? 1.0f : 0.0f);
			}
			if (rustleSound) {
				rustleSound->SetVolume(mgrState.enableRustle ? 1.0f : 0.0f);
			}
			if (spaceSound) {
				spaceSound->SetVolume(mgrState.enableSpaceNoise ? 1.0f : 0.0f);
			}
			if (speedWooshSound) {
				speedWooshSound->SetVolume(mgrState.enableSpeedWoosh ? 1.0f : 0.0f);
			}

			// Update procedural sources
			{
				std::lock_guard<std::mutex> lock(m_soundsMutex);
				proceduralSources.remove_if(
					[deltaTime, &state](const std::weak_ptr<ProceduralAudioSource>& weakSource) {
						if (auto source = weakSource.lock()) {
							source->OnUpdate(deltaTime, state);
							return false;
						}
						return true;
					}
				);

				// Cleanup finished sounds
				sounds.remove_if([](const std::weak_ptr<Sound>& weakSound) {
					if (auto sound = weakSound.lock()) {
						return sound->IsDone();
					}
					return true;
				});
			}
		}
	};

	AudioManager::AudioManager()
		: m_pimpl(std::make_unique<AudioManagerImpl>()) {}

	AudioManager::~AudioManager() {
		Shutdown();
	}

	void AudioManager::Initialize() {
		if (m_initialized)
			return;

		m_pimpl->InitEngine();
		if (!m_pimpl->initialized)
			return;

		m_initialized = true;

		// Create procedural audio effects
		m_pimpl->windEffect = std::make_shared<WindAudioEffect>();
		m_pimpl->rainEffect = std::make_shared<RainAudioEffect>();
		m_pimpl->rustleEffect = std::make_shared<RustleAudioEffect>();
		m_pimpl->spaceEffect = std::make_shared<SpaceAudioEffect>();
		m_pimpl->speedWooshEffect = std::make_shared<SpeedWooshAudioEffect>();

		// Create non-spatialized background procedural sounds
		m_pimpl->windSound = CreateProceduralSound(m_pimpl->windEffect, glm::vec3(0.0f), 1.0f, true, false);
		m_pimpl->rainSound = CreateProceduralSound(m_pimpl->rainEffect, glm::vec3(0.0f), 1.0f, true, false);
		m_pimpl->rustleSound = CreateProceduralSound(m_pimpl->rustleEffect, glm::vec3(0.0f), 1.0f, true, false);
		m_pimpl->spaceSound = CreateProceduralSound(m_pimpl->spaceEffect, glm::vec3(0.0f), 1.0f, true, false);
		m_pimpl->speedWooshSound = CreateProceduralSound(m_pimpl->speedWooshEffect, glm::vec3(0.0f), 1.0f, true, false);

		spdlog::info("AudioManager initialized successfully with procedural audio effects.");
	}

	void AudioManager::Shutdown() {
		if (m_pimpl) {
			m_pimpl->ShutdownEngine();
		}
		m_initialized = false;
	}

	IAudioManager::State AudioManager::GetState() const {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		return m_pimpl->m_managerState;
	}

	void AudioManager::SetState(const State& state) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_managerState = state;
	}

	void AudioManager::UpdateState(const AudioState& state) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_currentState = state;
	}

	void AudioManager::Update(float deltaTime) {
		// Parameter interpolation and audio loop run on m_audioThread
	}

	void AudioManager::PlayMusic(const std::string& filepath, bool loop, float volume) {
		if (!m_pimpl->initialized)
			return;
		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		m_pimpl->music = std::make_shared<Sound>(
			&m_pimpl->engine,
			filepath,
			loop,
			volume,
			false,
			glm::vec3(0.0f),
			m_pimpl->groupsInitialized ? &m_pimpl->musicGroup : nullptr
		);
	}

	void AudioManager::PlayAmbientSound(
		const std::string& name,
		const std::string& filepath,
		bool               loop,
		float              volume
	) {
		if (!m_pimpl->initialized)
			return;
		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		m_pimpl->ambientSounds[name] = std::make_shared<Sound>(
			&m_pimpl->engine,
			filepath,
			loop,
			volume,
			false,
			glm::vec3(0.0f),
			m_pimpl->groupsInitialized ? &m_pimpl->sfxGroup : nullptr
		);
	}

	void AudioManager::StopAmbientSound(const std::string& name) {
		if (!m_pimpl->initialized)
			return;
		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		m_pimpl->ambientSounds.erase(name);
	}

	void AudioManager::SetAmbientSoundVolume(const std::string& name, float volume) {
		if (!m_pimpl->initialized)
			return;
		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		if (m_pimpl->ambientSounds.count(name)) {
			m_pimpl->ambientSounds[name]->SetVolume(volume);
		}
	}

	std::shared_ptr<Sound> AudioManager::CreateSound(
		const std::string& filepath,
		const glm::vec3&   position,
		float              volume,
		bool               loop
	) {
		if (!m_pimpl->initialized)
			return nullptr;

		auto sound = std::make_shared<Sound>(
			&m_pimpl->engine,
			filepath,
			loop,
			volume,
			true,
			position,
			m_pimpl->groupsInitialized ? &m_pimpl->sfxGroup : nullptr
		);

		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		m_pimpl->sounds.push_back(sound);
		return sound;
	}

	std::shared_ptr<Sound> AudioManager::CreateProceduralSound(
		std::shared_ptr<ProceduralAudioSource> source,
		const glm::vec3&                       position,
		float                                  volume,
		bool                                   loop,
		bool                                   spatialized
	) {
		if (!m_pimpl->initialized || !source)
			return nullptr;

		auto sound = std::make_shared<Sound>(
			&m_pimpl->engine,
			source,
			loop,
			volume,
			spatialized,
			position,
			m_pimpl->groupsInitialized ? &m_pimpl->sfxGroup : nullptr
		);

		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);
		m_pimpl->proceduralSources.push_back(source);
		m_pimpl->sounds.push_back(sound);
		return sound;
	}

	void AudioManager::SetGlobalPitch(float pitch) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_managerState.globalPitch = pitch;
	}

	float AudioManager::GetMasterVolume() const {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		return m_pimpl->m_managerState.masterVolume;
	}

	void AudioManager::SetMasterVolume(float volume) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_managerState.masterVolume = volume;
	}

	float AudioManager::GetMusicVolume() const {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		return m_pimpl->m_managerState.musicVolume;
	}

	void AudioManager::SetMusicVolume(float volume) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_managerState.musicVolume = volume;
	}

	float AudioManager::GetSfxVolume() const {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		return m_pimpl->m_managerState.sfxVolume;
	}

	void AudioManager::SetSfxVolume(float volume) {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		m_pimpl->m_managerState.sfxVolume = volume;
	}

	AudioState AudioManager::GetCurrentState() const {
		std::lock_guard<std::mutex> lock(m_pimpl->m_stateMutex);
		return m_pimpl->m_currentState;
	}

	void AudioManager::StopAllSounds() {
		if (!m_pimpl->initialized)
			return;
		std::lock_guard<std::mutex> lock(m_pimpl->m_soundsMutex);

		if (m_pimpl->music) {
			m_pimpl->music->Stop();
		}
		for (auto& pair : m_pimpl->ambientSounds) {
			if (pair.second) {
				pair.second->Stop();
			}
		}
		for (auto& weakSound : m_pimpl->sounds) {
			if (auto sound = weakSound.lock()) {
				sound->Stop();
			}
		}

		m_pimpl->music.reset();
		m_pimpl->ambientSounds.clear();
		m_pimpl->sounds.clear();
		m_pimpl->proceduralSources.clear();
	}

} // namespace brassica
