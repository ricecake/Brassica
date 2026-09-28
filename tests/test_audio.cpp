#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <vector>

#include "ConfigManager.hpp"
#include "ServiceLocator.hpp"
#include "audio/AudioManager.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/IAudioManager.hpp"
#include "audio/ProceduralAudioSource.hpp"
#include "audio/RainAudioEffect.hpp"
#include "audio/RustleAudioEffect.hpp"
#include "audio/SpaceAudioEffect.hpp"
#include "audio/SpeedWooshAudioEffect.hpp"
#include "audio/WindAudioEffect.hpp"

namespace brassica {

	TEST_CASE("AudioState Data Structure Defaults and Assignments") {
		AudioState state{};
		CHECK(state.globalPitch == 1.0f);
		CHECK(state.masterVolume == 1.0f);
		CHECK(state.altitude == 0.0f);
		CHECK(state.speed == 0.0f);

		state.altitude = 12000.0f;
		state.speed = 150.0f;
		state.windStrength = 0.8f;

		CHECK(state.altitude == 12000.0f);
		CHECK(state.speed == 150.0f);
		CHECK(state.windStrength == 0.8f);
	}

	TEST_CASE("AudioManagerState Reflection and Active State Serialization") {
		AudioManagerState mgrState{};
		CHECK(mgrState.masterVolume == 1.0f);
		CHECK(mgrState.enableWind == true);
		CHECK(mgrState.enableSpaceNoise == true);
		CHECK(mgrState.enableSpeedWoosh == true);

		auto reflection = mgrState.GetReflection();
		CHECK(std::tuple_size<decltype(reflection)>::value == 9);

		ConfigManager config("TestApp");
		config.Initialize();

		AudioManager audioMgr;
		audioMgr.Initialize();

		mgrState.masterVolume = 0.75f;
		mgrState.enableRain = true;
		audioMgr.SetState(mgrState);

		audioMgr.SaveState(config);
		CHECK(config.GetManagerSetting<float>("AudioManager", "masterVolume", 1.0f) == 0.75f);
		CHECK(config.GetManagerSetting<bool>("AudioManager", "enableRain", false) == true);

		AudioManager newAudioMgr;
		newAudioMgr.Initialize();
		newAudioMgr.LoadState(config);

		auto loadedState = newAudioMgr.GetState();
		CHECK(loadedState.masterVolume == 0.75f);
		CHECK(loadedState.enableRain == true);
	}

	TEST_CASE("Procedural Audio Effects PCM Buffer Generation") {
		SUBCASE("WindAudioEffect Buffer Output") {
			WindAudioEffect wind;
			AudioState      state{};
			state.windStrength = 0.5f;

			wind.OnUpdate(0.016f, state);

			std::vector<float> buffer(512 * 2, 0.0f);
			wind.OnRead(buffer.data(), 512);

			bool nonZero = false;
			for (float sample : buffer) {
				if (std::abs(sample) > 1e-6f) {
					nonZero = true;
					break;
				}
			}
			CHECK(nonZero);
		}

		SUBCASE("RainAudioEffect Buffer Output") {
			RainAudioEffect rain;
			AudioState      state{};
			state.rainIntensity = 0.8f;

			rain.OnUpdate(0.016f, state);

			std::vector<float> buffer(512 * 2, 0.0f);
			rain.OnRead(buffer.data(), 512);

			bool nonZero = false;
			for (float sample : buffer) {
				if (std::abs(sample) > 1e-6f) {
					nonZero = true;
					break;
				}
			}
			CHECK(nonZero);
		}

		SUBCASE("RustleAudioEffect Buffer Output") {
			RustleAudioEffect rustle;
			AudioState        state{};
			state.windStrength = 0.6f;
			state.grassDensity = 0.9f;

			rustle.OnUpdate(0.016f, state);

			std::vector<float> buffer(512 * 2, 0.0f);
			rustle.OnRead(buffer.data(), 512);

			bool nonZero = false;
			for (float sample : buffer) {
				if (std::abs(sample) > 1e-6f) {
					nonZero = true;
					break;
				}
			}
			CHECK(nonZero);
		}

		SUBCASE("SpaceAudioEffect Altitude Scaling and Buffer Output") {
			SpaceAudioEffect space;
			AudioState       lowAltState{};
			lowAltState.altitude = 100.0f;

			space.OnUpdate(0.1f, lowAltState);

			std::vector<float> lowBuffer(512 * 2, 0.0f);
			space.OnRead(lowBuffer.data(), 512);

			AudioState highAltState{};
			highAltState.altitude = 25000.0f;

			space.OnUpdate(0.1f, highAltState);

			std::vector<float> highBuffer(512 * 2, 0.0f);
			space.OnRead(highBuffer.data(), 512);

			float lowEnergy = 0.0f;
			float highEnergy = 0.0f;
			for (size_t i = 0; i < lowBuffer.size(); ++i) {
				lowEnergy += std::abs(lowBuffer[i]);
				highEnergy += std::abs(highBuffer[i]);
			}

			// High altitude should produce significantly higher space noise rumble energy
			CHECK(highEnergy > lowEnergy);
		}

		SUBCASE("SpeedWooshAudioEffect Speed Scaling and Buffer Output") {
			SpeedWooshAudioEffect woosh;
			AudioState            lowSpeedState{};
			lowSpeedState.speed = 5.0f;

			woosh.OnUpdate(0.1f, lowSpeedState);

			std::vector<float> lowBuffer(512 * 2, 0.0f);
			woosh.OnRead(lowBuffer.data(), 512);

			AudioState highSpeedState{};
			highSpeedState.speed = 300.0f;

			woosh.OnUpdate(0.1f, highSpeedState);

			std::vector<float> highBuffer(512 * 2, 0.0f);
			woosh.OnRead(highBuffer.data(), 512);

			float lowEnergy = 0.0f;
			float highEnergy = 0.0f;
			for (size_t i = 0; i < lowBuffer.size(); ++i) {
				lowEnergy += std::abs(lowBuffer[i]);
				highEnergy += std::abs(highBuffer[i]);
			}

			// High speed camera movement should produce higher wind woosh energy
			CHECK(highEnergy > lowEnergy);
		}
	}

	TEST_CASE("AudioManager Service Locator and Lifecycle Operations") {
		ServiceLocator locator;
		ServiceLocator::SetInstance(&locator);

		AudioManager audioMgr;
		audioMgr.Initialize();

		locator.Provide<IAudioManager>(std::shared_ptr<IAudioManager>(&audioMgr, [](IAudioManager*) {}));

		CHECK(locator.Has<IAudioManager>());
		auto retrieved = locator.Get<IAudioManager>();

		CHECK(retrieved->GetMasterVolume() == 1.0f);
		retrieved->SetMasterVolume(0.5f);
		CHECK(retrieved->GetMasterVolume() == 0.5f);

		AudioState state{};
		state.altitude = 5000.0f;
		state.speed = 80.0f;
		retrieved->UpdateState(state);

		AudioState current = retrieved->GetCurrentState();
		CHECK(current.altitude == 5000.0f);
		CHECK(current.speed == 80.0f);

		retrieved->StopAllSounds();

		ServiceLocator::SetInstance(nullptr);
	}

} // namespace brassica
