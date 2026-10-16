#include "lighting/LightManager.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>

#include "EngineConstants.hpp"
#include <glm/gtc/constants.hpp>

namespace brassica {

	static std::map<char, std::string> morseAlphabet = {
		{'a', ".-"},    {'b', "-..."},   {'c', "-.-."},   {'d', "-.."},    {'e', "."},     {'f', "..-."},
		{'g', "--."},   {'h', "...."},   {'i', ".."},     {'j', ".---"},   {'k', "-.-"},   {'l', ".-.."},
		{'m', "--"},    {'n', "-."},     {'o', "---"},    {'p', ".--."},   {'q', "--.-"},  {'r', ".-."},
		{'s', "..."},   {'t', "-"},      {'u', "..-"},    {'v', "...-"},   {'w', ".--"},   {'x', "-..-"},
		{'y', "-.--"},  {'z', "--.."},   {'1', ".----"},  {'2', "..---"},  {'3', "...--"}, {'4', "....-"},
		{'5', "....."}, {'6', "-...."},  {'7', "--..."},  {'8', "---.."},  {'9', "----."}, {'0', "-----"},
		{' ', "/"},     {',', "--..--"}, {'.', ".-.-.-"}, {'\'', ".----."}
	};

	LightManager::LightManager() {
		_lights[0].id = _nextLightId++;
		_lights[1].id = _nextLightId++;
	}

	static void GenerateMorseSequence(Light& light) {
		light.behavior.morseSequence.clear();
		std::string msg = light.behavior.message;
		std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char c) { return std::tolower(c); });

		for (char c : msg) {
			if (morseAlphabet.count(c)) {
				std::string code = morseAlphabet[c];
				if (code == "/") {
					for (int j = 0; j < 4; ++j) {
						light.behavior.morseSequence.push_back(false);
					}
				} else {
					for (char symbol : code) {
						if (symbol == '.') {
							light.behavior.morseSequence.push_back(true);  // dot
							light.behavior.morseSequence.push_back(false); // inter-symbol gap
						} else if (symbol == '-') {
							light.behavior.morseSequence.push_back(true);
							light.behavior.morseSequence.push_back(true);
							light.behavior.morseSequence.push_back(true);  // dash
							light.behavior.morseSequence.push_back(false); // inter-symbol gap
						}
					}
					light.behavior.morseSequence.push_back(false);
					light.behavior.morseSequence.push_back(false);
				}
			}
		}
	}

	int LightManager::AddLight(const Light& light) {
		Light l = light;
		if (l.id == -1) {
			l.id = _nextLightId++;
		}
		_lights.push_back(l);
		return l.id;
	}

	void LightManager::RemoveLight(int id) {
		if (id == -1)
			return;
		for (auto it = _lights.begin(); it != _lights.end(); ++it) {
			if (it->id == id) {
				_lights.erase(it);
				return;
			}
		}
	}

	Light* LightManager::GetLight(int id) {
		if (id == -1)
			return nullptr;
		for (auto& light : _lights) {
			if (light.id == id)
				return &light;
		}
		return nullptr;
	}

	const Light* LightManager::GetLight(int id) const {
		if (id == -1)
			return nullptr;
		for (const auto& light : _lights) {
			if (light.id == id)
				return &light;
		}
		return nullptr;
	}

	std::vector<Light>& LightManager::GetLights() {
		return _lights;
	}

	const std::vector<Light>& LightManager::GetLights() const {
		return _lights;
	}

	void LightManager::Update(float deltaTime) {
		Update(deltaTime, glm::vec3(0.0f));
	}

	void LightManager::Update(float deltaTime, const glm::vec3& cameraPosition) {
		if (_cycle.enabled) {
			if (!_cycle.paused) {
				_cycle.time += deltaTime * _cycle.speed;
				if (_cycle.time >= 24.0f)
					_cycle.time -= 24.0f;
				if (_cycle.time < 0.0f)
					_cycle.time += 24.0f;
			}

			if (_lights.size() >= 2 && _lights[0].type == DIRECTIONAL_LIGHT && _lights[1].type == DIRECTIONAL_LIGHT) {
				constexpr float R = FAKE_PLANET_RADIUS;
				float phi = -cameraPosition.z / R; // Latitude in radians
				phi = std::clamp(phi, -glm::half_pi<float>(), glm::half_pi<float>());
				float lambda = cameraPosition.x / R; // Longitude in radians
				while (lambda > glm::pi<float>()) lambda -= glm::two_pi<float>();
				while (lambda < -glm::pi<float>()) lambda += glm::two_pi<float>();

				// Summer Solstice for Northern Hemisphere (solar declination = +23.44 deg)
				constexpr float deltaSun = glm::radians(23.44f);

				// Camera local time in hours
				float localTime = _cycle.time + (lambda / glm::pi<float>()) * 12.0f;
				while (localTime >= 24.0f) localTime -= 24.0f;
				while (localTime < 0.0f) localTime += 24.0f;

				// Hour angle of sun (0 at local solar noon = 12h)
				float Hsun = (localTime - 12.0f) * (glm::pi<float>() / 12.0f);

				// Topocentric Sun direction vector (pointing TOWARDS sun)
				glm::vec3 sunDir;
				sunDir.x = -std::cos(deltaSun) * std::sin(Hsun);
				sunDir.y = std::sin(phi) * std::sin(deltaSun) + std::cos(phi) * std::cos(deltaSun) * std::cos(Hsun);
				sunDir.z = -std::cos(phi) * std::sin(deltaSun) + std::sin(phi) * std::cos(deltaSun) * std::cos(Hsun);
				if (glm::length(sunDir) > 1e-6f) {
					sunDir = glm::normalize(sunDir);
				} else {
					sunDir = glm::vec3(0.0f, 1.0f, 0.0f);
				}

				_lights[0].direction = -sunDir;
				_lights[0].elevation = glm::degrees(std::asin(std::clamp(sunDir.y, -1.0f, 1.0f)));
				_lights[0].azimuth = glm::degrees(std::atan2(sunDir.x, sunDir.z));

				// Moon ephemeris
				if (!_cycle.paused) {
					_cycle.moonPhaseDays += deltaTime * _cycle.speed / 24.0f;
				}

				float phaseRad = std::fmod(_cycle.moonPhaseDays, _cycle.lunarMonth) / _cycle.lunarMonth * glm::two_pi<float>();
				float deltaMoon = deltaSun * std::cos(phaseRad);
				float Hmoon = Hsun - phaseRad + glm::radians(_cycle.moonOffset * 15.0f);

				glm::vec3 moonDir;
				moonDir.x = -std::cos(deltaMoon) * std::sin(Hmoon);
				moonDir.y = std::sin(phi) * std::sin(deltaMoon) + std::cos(phi) * std::cos(deltaMoon) * std::cos(Hmoon);
				moonDir.z = -std::cos(phi) * std::sin(deltaMoon) + std::sin(phi) * std::cos(deltaMoon) * std::cos(Hmoon);
				if (glm::length(moonDir) > 1e-6f) {
					moonDir = glm::normalize(moonDir);
				} else {
					moonDir = glm::vec3(0.0f, -1.0f, 0.0f);
				}

				_lights[1].direction = -moonDir;
				_lights[1].elevation = glm::degrees(std::asin(std::clamp(moonDir.y, -1.0f, 1.0f)));
				_lights[1].azimuth = glm::degrees(std::atan2(moonDir.x, moonDir.z));

				// Geometric planet curvature occlusion from camera altitude
				glm::vec3 planetCenter(0.0f, -R, 0.0f);
				glm::vec3 roCam = cameraPosition - planetCenter;
				float     rCamLen = glm::length(roCam);
				glm::vec3 topocentricNormal = (rCamLen > 1e-4f) ? (roCam / rCamLen) : glm::vec3(0.0f, 1.0f, 0.0f);
				float     muHorizon = -std::sqrt(std::max(0.0f, 1.0f - (R * R) / (rCamLen * rCamLen)));

				float sunNdotL = glm::dot(topocentricNormal, sunDir);
				float moonNdotL = glm::dot(topocentricNormal, moonDir);

				float sunVis = sunNdotL - muHorizon;
				float moonVis = moonNdotL - muHorizon;

				float sunFade = glm::smoothstep(-0.02f, 0.01f, sunVis);
				_lights[0].color = glm::vec3(2.5f, 2.3f, 2.0f);
				_lights[0].baseIntensity = 10.0f * sunFade;

				float cosPhase = glm::dot(sunDir, moonDir);
				float phase = glm::clamp((-cosPhase + 1.0f) * 0.5f, 0.05f, 1.0f);

				const float     lunarAlbedo = _cycle.lunarAlbedo;
				const glm::vec3 lunarTint = _cycle.moonTint;

				glm::vec3 sunFullRadiance = _lights[0].color * 10.0f;
				_lights[1].color = sunFullRadiance * lunarAlbedo * phase * lunarTint;

				float moonFade = glm::smoothstep(-0.02f, 0.01f, moonVis);
				_lights[1].baseIntensity = 1.0f * moonFade;

				_cycle.nightFactor = glm::smoothstep(0.1f, -0.1f, sunVis);

				const glm::vec3 airglow{0.0008f, 0.0015f, 0.0022f};
				glm::vec3       dayAmbient{0.2f, 0.2f, 0.25f};
				glm::vec3       nightAmbient = airglow + dayAmbient * 0.15f +
					_lights[1].color * 0.3f * std::max(0.0f, moonVis);
				float ambientFactor = std::clamp(sunVis * 5.0f + 0.5f, 0.0f, 1.0f);
				_ambientLight = glm::mix(nightAmbient, dayAmbient, ambientFactor);
			}
		}

		for (auto it = _lights.begin(); it != _lights.end();) {
			auto& light = *it;
			if (light.behavior.type == LightBehaviorType::NONE) {
				light.intensity = light.baseIntensity;
				++it;
				continue;
			}

			light.behavior.timer += deltaTime;

			bool finished = false;
			if (!light.behavior.loop && light.behavior.timer >= light.behavior.period) {
				finished = true;
			}

			switch (light.behavior.type) {
			case LightBehaviorType::BLINK: {
				float phase = std::fmod(light.behavior.timer, light.behavior.period);
				light.intensity = (phase < light.behavior.period * light.behavior.dutyCycle) ? light.baseIntensity
																							 : 0.0f;
				break;
			}
			case LightBehaviorType::PULSE: {
				float sine = std::sin(2.0f * 3.14159265f * light.behavior.timer / light.behavior.period);
				light.intensity = light.baseIntensity * (1.0f + light.behavior.amplitude * sine) * 0.5f;
				break;
			}
			case LightBehaviorType::EASE_IN: {
				float t = std::min(light.behavior.timer / light.behavior.period, 1.0f);
				float easing = t * t * (3.0f - 2.0f * t);
				light.intensity = light.baseIntensity * easing;
				break;
			}
			case LightBehaviorType::EASE_OUT: {
				float t = std::min(light.behavior.timer / light.behavior.period, 1.0f);
				float easing = 1.0f - (t * t * (3.0f - 2.0f * t));
				light.intensity = light.baseIntensity * easing;
				break;
			}
			case LightBehaviorType::EASE_IN_OUT: {
				float t = std::fmod(light.behavior.timer, light.behavior.period) / light.behavior.period;
				float easing = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * t);
				light.intensity = light.baseIntensity * easing;
				break;
			}
			case LightBehaviorType::FLICKER: {
				float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				if (light.behavior.flickerIntensity <= 0.5f) {
					if (r < 0.05f) {
						light.intensity = light.baseIntensity * (0.2f + 0.3f * (rand() % 100) / 100.0f);
					} else {
						light.intensity = light.baseIntensity;
					}
				} else if (light.behavior.flickerIntensity <= 1.5f) {
					if (r < 0.05f) {
						light.intensity = light.baseIntensity * (0.5f + 0.5f * (rand() % 100) / 100.0f);
					} else {
						light.intensity = 0.0f;
					}
				} else {
					float chance = std::min(light.behavior.flickerIntensity / 5.0f, 1.0f);
					if (r < chance) {
						light.intensity = light.baseIntensity *
							(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
					}
				}
				break;
			}
			case LightBehaviorType::MORSE: {
				if (light.behavior.morseIndex == -1) {
					GenerateMorseSequence(light);
					light.behavior.morseIndex = 0;
					light.behavior.timer = 0.0f;
				}
				if (light.behavior.morseSequence.empty())
					break;

				size_t index = static_cast<size_t>(light.behavior.timer / light.behavior.period);
				if (light.behavior.loop) {
					index = index % light.behavior.morseSequence.size();
				} else if (index >= light.behavior.morseSequence.size()) {
					light.intensity = 0.0f;
					break;
				}
				light.intensity = light.behavior.morseSequence[index] ? light.baseIntensity : 0.0f;
				break;
			}
			default:
				break;
			}

			if (light.autoRemove && finished) {
				it = _lights.erase(it);
			} else {
				++it;
			}
		}
	}

	std::vector<Light*> LightManager::GetShadowCastingLights() {
		std::vector<Light*> shadowLights;
		for (auto& light : _lights) {
			if (light.castsShadow && light.intensity > 0.0f) {
				shadowLights.push_back(&light);
			}
		}
		return shadowLights;
	}

	int LightManager::GetShadowCastingLightCount() const {
		int count = 0;
		for (const auto& light : _lights) {
			if (light.castsShadow && light.intensity > 0.0f) {
				++count;
			}
		}
		return count;
	}

	LightingUBO LightManager::GetLightingUBO() const {
		LightingUBO ubo{};
		ubo.numLights = static_cast<uint32_t>(std::min(_lights.size(), static_cast<size_t>(MAX_LIGHTS)));
		ubo.dayTime = _cycle.time;
		ubo.nightFactor = _cycle.nightFactor;
		ubo.worldScale = 1.0f;

		ubo.ambientLight = glm::vec4(_ambientLight, 1.0f);
		ubo.lightningColor = glm::vec4(0.0f);

		ubo.skyExposure = _skyExposure;
		ubo.starExposure = _starExposure;
		ubo.terrainExposure = _terrainExposure;
		ubo._paddingExposure = 0.0f;

		return ubo;
	}

	LightsSSBOData LightManager::GetLightsSSBOData() const {
		LightsSSBOData ssbo{};
		uint32_t       count = static_cast<uint32_t>(std::min(_lights.size(), static_cast<size_t>(MAX_LIGHTS)));
		ssbo.count = count;
		for (uint32_t i = 0; i < count; ++i) {
			ssbo.lights[i] = _lights[i].ToGPU();
		}
		return ssbo;
	}

} // namespace brassica
