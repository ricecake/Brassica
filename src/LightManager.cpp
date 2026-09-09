#include "LightManager.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>

#include "spdlog/spdlog.h"

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
		for (auto& l : _lights) {
			if (l.id == -1) {
				l.id = nextLightId++;
			}
		}
	}

	LightManager::~LightManager() {
		CleanupGpuResources();
	}

	static void GenerateMorseSequence(Light& light) {
		light.behavior.morse_sequence.clear();
		std::string msg = light.behavior.message;
		std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

		for (char c : msg) {
			if (morseAlphabet.count(c)) {
				std::string code = morseAlphabet[c];
				if (code == "/") {
					for (int j = 0; j < 4; ++j) {
						light.behavior.morse_sequence.push_back(false);
					}
				} else {
					for (char symbol : code) {
						if (symbol == '.') {
							light.behavior.morse_sequence.push_back(true);
							light.behavior.morse_sequence.push_back(false);
						} else if (symbol == '-') {
							light.behavior.morse_sequence.push_back(true);
							light.behavior.morse_sequence.push_back(true);
							light.behavior.morse_sequence.push_back(true);
							light.behavior.morse_sequence.push_back(false);
						}
					}
					light.behavior.morse_sequence.push_back(false);
					light.behavior.morse_sequence.push_back(false);
				}
			}
		}
	}

	int LightManager::AddLight(const Light& light) {
		Light l = light;
		if (l.id == -1) {
			l.id = nextLightId++;
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
			if (light.id == id) {
				return &light;
			}
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
		if (cycle.enabled) {
			if (!cycle.paused) {
				cycle.time += deltaTime * cycle.speed;
				if (cycle.time >= 24.0f) {
					cycle.time -= 24.0f;
				}
				if (cycle.time < 0.0f) {
					cycle.time += 24.0f;
				}
			}

			if (_lights.size() >= 2 && _lights[0].type == DIRECTIONAL_LIGHT && _lights[1].type == DIRECTIONAL_LIGHT) {
				float sun_angle_deg = (cycle.time / 24.0f) * 360.0f;
				_lights[0].elevation = sun_angle_deg - 90.0f;
				_lights[0].azimuth = 90.0f;
				_lights[0].UpdateDirectionFromAngles();

				if (!cycle.paused) {
					cycle.moon_phase_days += deltaTime * cycle.speed / 24.0f;
				}

				float phase_drift = std::fmod(cycle.moon_phase_days, cycle.lunar_month) / cycle.lunar_month * 24.0f;
				float effective_offset = cycle.moon_offset + phase_drift;

				float moon_time = cycle.time + effective_offset;
				if (moon_time >= 24.0f)
					moon_time -= 24.0f;
				if (moon_time < 0.0f)
					moon_time += 24.0f;
				float moon_angle_deg = (moon_time / 24.0f) * 360.0f;
				_lights[1].elevation = moon_angle_deg - 90.0f;

				float erratic_azimuth = 30.0f * std::sin(cycle.moon_phase_days * 0.7f) +
					15.0f * std::sin(cycle.moon_phase_days * 2.3f + moon_time * 0.1f);
				_lights[1].azimuth = cycle.moon_azimuth + erratic_azimuth;
				_lights[1].UpdateDirectionFromAngles();

				float sun_vis = glm::sin(glm::radians(_lights[0].elevation));
				float moon_vis = glm::sin(glm::radians(_lights[1].elevation));

				float sun_fade = 1.0f;
				if (sun_vis > 0.05f) {
					sun_fade = 1.0f;
				} else if (sun_vis < -0.20f) {
					sun_fade = 0.0f;
				} else {
					float t = (sun_vis - (-0.20f)) / (0.05f - (-0.20f));
					sun_fade = glm::smoothstep(0.0f, 1.0f, t);
				}
				_lights[0].base_intensity = 100000.0f * sun_fade;

				glm::vec3 sunDir = glm::normalize(-_lights[0].direction);
				glm::vec3 moonDir = glm::normalize(-_lights[1].direction);

				float cosAlpha = glm::clamp(-glm::dot(sunDir, moonDir), -1.0f, 1.0f);
				float alphaDeg = glm::degrees(glm::acos(cosAlpha));

				float alpha2 = alphaDeg * alphaDeg;
				float alpha4 = alpha2 * alpha2;
				float deltaM = 0.026f * alphaDeg + 0.000000004f * alpha4;

				float phaseFactor = std::pow(10.0f, -0.4f * deltaM);
				const float earthshineFloor = 0.00015f;
				phaseFactor = glm::max(phaseFactor, earthshineFloor);

				_lights[1].color = cycle.moon_tint;

				const float peakFullMoonLux = 0.34f;
				float moon_fade = 1.0f;
				if (moon_vis > 0.05f) {
					moon_fade = 1.0f;
				} else if (moon_vis < -0.05f) {
					moon_fade = 0.0f;
				} else {
					float t = (moon_vis - (-0.05f)) / (0.05f - (-0.05f));
					moon_fade = glm::smoothstep(0.0f, 1.0f, t);
				}
				_lights[1].base_intensity = peakFullMoonLux * phaseFactor * moon_fade;

				cycle.night_factor = glm::smoothstep(0.2f, -0.2f, sun_vis);

				glm::vec3 day_ambient{0.2f, 0.2f, 0.2f};
				glm::vec3 night_ambient = day_ambient * 0.15f;
				night_ambient += _lights[1].color * 0.3f * std::max(0.0f, moon_vis);

				float ambient_factor = glm::smoothstep(-0.20f, 0.05f, sun_vis);
				ambientLight = glm::mix(night_ambient, day_ambient, ambient_factor);
			}
		}

		for (auto it = _lights.begin(); it != _lights.end();) {
			auto& light = *it;
			if (light.behavior.type == LightBehaviorType::NONE) {
				light.intensity = light.base_intensity;
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
				light.intensity = (phase < light.behavior.period * light.behavior.duty_cycle) ? light.base_intensity : 0.0f;
				break;
			}
			case LightBehaviorType::PULSE: {
				float sine = std::sin(2.0f * 3.14159265f * light.behavior.timer / light.behavior.period);
				light.intensity = light.base_intensity * (1.0f + light.behavior.amplitude * sine) * 0.5f;
				break;
			}
			case LightBehaviorType::EASE_IN: {
				float t = std::min(light.behavior.timer / light.behavior.period, 1.0f);
				float easing = t * t * (3.0f - 2.0f * t);
				light.intensity = light.base_intensity * easing;
				break;
			}
			case LightBehaviorType::EASE_OUT: {
				float t = std::min(light.behavior.timer / light.behavior.period, 1.0f);
				float easing = 1.0f - (t * t * (3.0f - 2.0f * t));
				light.intensity = light.base_intensity * easing;
				break;
			}
			case LightBehaviorType::EASE_IN_OUT: {
				float t = std::fmod(light.behavior.timer, light.behavior.period) / light.behavior.period;
				float easing = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * t);
				light.intensity = light.base_intensity * easing;
				break;
			}
			case LightBehaviorType::FLICKER: {
				float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				if (light.behavior.flicker_intensity <= 0.5f) {
					if (r < 0.05f)
						light.intensity = light.base_intensity * (0.2f + 0.3f * (rand() % 100) / 100.0f);
					else
						light.intensity = light.base_intensity;
				} else if (light.behavior.flicker_intensity <= 1.5f) {
					if (r < 0.05f)
						light.intensity = light.base_intensity * (0.5f + 0.5f * (rand() % 100) / 100.0f);
					else
						light.intensity = 0.0f;
				} else {
					float chance = std::min(light.behavior.flicker_intensity / 5.0f, 1.0f);
					if (r < chance) {
						light.intensity = light.base_intensity * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
					}
				}
				break;
			}
			case LightBehaviorType::MORSE: {
				if (light.behavior.morse_index == -1) {
					GenerateMorseSequence(light);
					light.behavior.morse_index = 0;
					light.behavior.timer = 0.0f;
				}
				if (light.behavior.morse_sequence.empty())
					break;

				int index = static_cast<int>(light.behavior.timer / light.behavior.period);
				if (light.behavior.loop) {
					index = index % static_cast<int>(light.behavior.morse_sequence.size());
				} else if (index >= static_cast<int>(light.behavior.morse_sequence.size())) {
					light.intensity = 0.0f;
					break;
				}
				light.intensity = light.behavior.morse_sequence[index] ? light.base_intensity : 0.0f;
				break;
			}
			default:
				break;
			}

			if (light.auto_remove && finished) {
				it = _lights.erase(it);
			} else {
				++it;
			}
		}
	}

	std::vector<Light*> LightManager::GetShadowCastingLights() {
		std::vector<Light*> shadow_lights;
		for (auto& light : _lights) {
			if (light.casts_shadow && light.intensity > 0.0f) {
				shadow_lights.push_back(&light);
			}
		}
		return shadow_lights;
	}

	int LightManager::GetShadowCastingLightCount() const {
		int count = 0;
		for (const auto& light : _lights) {
			if (light.casts_shadow && light.intensity > 0.0f) {
				++count;
			}
		}
		return count;
	}

	bool LightManager::InitGpuResources(vk::Device dev, VmaAllocator alloc) {
		device = dev;
		allocator = alloc;

		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			// Lighting UBO Allocation
			VkBufferCreateInfo uboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			uboInfo.size = sizeof(LightingUbo);
			uboInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VmaAllocationInfo uboAllocResultInfo{};
			if (vmaCreateBuffer(
					allocator,
					&uboInfo,
					&allocCreateInfo,
					&lightingUboBuffers[i],
					&lightingUboAllocations[i],
					&uboAllocResultInfo
				) != VK_SUCCESS) {
				spdlog::error("Failed to create Lighting UBO buffer via VMA");
				return false;
			}
			lightingUboMapped[i] = uboAllocResultInfo.pMappedData;

			// Lights SSBO Allocation
			VkBufferCreateInfo ssboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			ssboInfo.size = sizeof(LightsSSBOData);
			ssboInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

			VmaAllocationInfo ssboAllocResultInfo{};
			if (vmaCreateBuffer(
					allocator,
					&ssboInfo,
					&allocCreateInfo,
					&lightsSsboBuffers[i],
					&lightsSsboAllocations[i],
					&ssboAllocResultInfo
				) != VK_SUCCESS) {
				spdlog::error("Failed to create Lights SSBO buffer via VMA");
				return false;
			}
			lightsSsboMapped[i] = ssboAllocResultInfo.pMappedData;
		}

		return true;
	}

	void LightManager::CleanupGpuResources() {
		if (allocator != VK_NULL_HANDLE) {
			for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
				if (lightingUboBuffers[i]) {
					vmaDestroyBuffer(allocator, lightingUboBuffers[i], lightingUboAllocations[i]);
					lightingUboBuffers[i] = VK_NULL_HANDLE;
					lightingUboAllocations[i] = VK_NULL_HANDLE;
					lightingUboMapped[i] = nullptr;
				}
				if (lightsSsboBuffers[i]) {
					vmaDestroyBuffer(allocator, lightsSsboBuffers[i], lightsSsboAllocations[i]);
					lightsSsboBuffers[i] = VK_NULL_HANDLE;
					lightsSsboAllocations[i] = VK_NULL_HANDLE;
					lightsSsboMapped[i] = nullptr;
				}
			}
		}
	}

	void LightManager::UpdateGpuBuffers(
		uint32_t         activeFrame,
		const glm::mat4& view,
		const glm::mat4& projection,
		const glm::vec3& cameraPos,
		const glm::vec3& cameraDir,
		float            zNear,
		float            zFar,
		float            time
	) {
		uint32_t frameIdx = activeFrame % FRAME_OVERLAP;

		// 1. Populate Lighting UBO CPU struct
		lightingUbo.num_lights = static_cast<int32_t>(std::min(_lights.size(), static_cast<size_t>(MAX_LIGHTS)));
		lightingUbo.day_time = cycle.time;
		lightingUbo.night_factor = cycle.night_factor;
		lightingUbo.view_pos = cameraPos;
		lightingUbo.ambient_light = ambientLight;
		lightingUbo.time = time;
		lightingUbo.view_dir = cameraDir;
		lightingUbo.zNear = zNear;
		lightingUbo.zFar = zFar;
		lightingUbo.view = view;
		lightingUbo.projection = projection;
		lightingUbo.skyExposure = skyExposure;
		lightingUbo.starExposure = starExposure;
		lightingUbo.terrainExposure = terrainExposure;

		if (lightingUboMapped[frameIdx]) {
			std::memcpy(lightingUboMapped[frameIdx], &lightingUbo, sizeof(LightingUbo));
		}

		// 2. Populate Lights SSBO Data CPU struct
		lightsSsboData.count = static_cast<uint32_t>(lightingUbo.num_lights);
		for (uint32_t i = 0; i < lightsSsboData.count; ++i) {
			lightsSsboData.lights[i] = _lights[i].ToGPU();
		}

		if (lightsSsboMapped[frameIdx]) {
			std::memcpy(lightsSsboMapped[frameIdx], &lightsSsboData, sizeof(LightsSSBOData));
		}
	}

} // namespace brassica
