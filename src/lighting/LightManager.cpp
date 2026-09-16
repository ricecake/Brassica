#include "lighting/LightManager.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>

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
		if (_cycle.enabled) {
			if (!_cycle.paused) {
				_cycle.time += deltaTime * _cycle.speed;
				if (_cycle.time >= 24.0f)
					_cycle.time -= 24.0f;
				if (_cycle.time < 0.0f)
					_cycle.time += 24.0f;
			}

			if (_lights.size() >= 2 && _lights[0].type == DIRECTIONAL_LIGHT && _lights[1].type == DIRECTIONAL_LIGHT) {
				// Sun
				float sunAngleDeg = (_cycle.time / 24.0f) * 360.0f;
				_lights[0].elevation = sunAngleDeg - 90.0f;
				_lights[0].azimuth = 90.0f;
				_lights[0].UpdateDirectionFromAngles();

				// Moon
				if (!_cycle.paused) {
					_cycle.moonPhaseDays += deltaTime * _cycle.speed / 24.0f;
				}

				float phaseDrift = std::fmod(_cycle.moonPhaseDays, _cycle.lunarMonth) / _cycle.lunarMonth * 24.0f;
				float effectiveOffset = _cycle.moonOffset + phaseDrift;

				float moonTime = _cycle.time + effectiveOffset;
				if (moonTime >= 24.0f)
					moonTime -= 24.0f;
				if (moonTime < 0.0f)
					moonTime += 24.0f;
				float moonAngleDeg = (moonTime / 24.0f) * 360.0f;
				_lights[1].elevation = moonAngleDeg - 90.0f;

				float erraticAzimuth = 30.0f * std::sin(_cycle.moonPhaseDays * 0.7f) +
					15.0f * std::sin(_cycle.moonPhaseDays * 2.3f + moonTime * 0.1f);
				_lights[1].azimuth = _cycle.moonAzimuth + erraticAzimuth;
				_lights[1].UpdateDirectionFromAngles();

				float sunVis = glm::sin(glm::radians(_lights[0].elevation));
				float moonVis = glm::sin(glm::radians(_lights[1].elevation));

				float sunFade = 1.0f;
				if (sunVis > 0.05f) {
					sunFade = 1.0f;
				} else if (sunVis < -0.20f) {
					sunFade = 0.0f;
				} else {
					float t = (sunVis - (-0.20f)) / (0.05f - (-0.20f));
					sunFade = glm::smoothstep(0.0f, 1.0f, t);
				}
				_lights[0].color = glm::vec3(2.5f, 2.3f, 2.0f);
				_lights[0].baseIntensity = 1.0f * sunFade;

				glm::vec3 sunDir = glm::normalize(-_lights[0].direction);
				glm::vec3 moonDir = glm::normalize(-_lights[1].direction);

				float cosAlpha = glm::clamp(-glm::dot(sunDir, moonDir), -1.0f, 1.0f);
				float alphaDeg = glm::degrees(glm::acos(cosAlpha));

				float alpha2 = alphaDeg * alphaDeg;
				float alpha4 = alpha2 * alpha2;
				float deltaM = 0.026f * alphaDeg + 0.000000004f * alpha4;
				float phaseFactor = std::pow(10.0f, -0.4f * deltaM);
				phaseFactor = glm::max(phaseFactor, 0.00015f);

				_lights[1].color = _cycle.moonTint;

				float moonFade = 1.0f;
				if (moonVis > 0.05f) {
					moonFade = 1.0f;
				} else if (moonVis < -0.05f) {
					moonFade = 0.0f;
				} else {
					float t = (moonVis - (-0.05f)) / (0.05f - (-0.05f));
					moonFade = glm::smoothstep(0.0f, 1.0f, t);
				}
				_lights[1].baseIntensity = 0.34f * phaseFactor * moonFade;

				_cycle.nightFactor = glm::smoothstep(0.2f, -0.2f, sunVis);

				glm::vec3 dayAmbient{0.2f, 0.2f, 0.25f};
				glm::vec3 nightAmbient = dayAmbient * 0.15f + _lights[1].color * 0.3f * std::max(0.0f, moonVis);
				float     ambientFactor = glm::smoothstep(-0.20f, 0.05f, sunVis);
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
