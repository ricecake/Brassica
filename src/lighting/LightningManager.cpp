#include "lighting/LightningManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <glm/gtc/random.hpp>

namespace brassica {

	void
	LightningManager::GenerateBolt(LightningStrike& strike, const glm::vec3& start, const glm::vec3& end, int depth) {
		if (depth <= 0) {
			strike.segments.push_back({start, end});
			return;
		}

		glm::vec3 mid = (start + end) * 0.5f;
		float     dist = glm::length(end - start);

		// Add perpendicular offset
		glm::vec3 dir = glm::normalize(end - start);
		glm::vec3 perp = glm::linearRand(glm::vec3(-1.0f), glm::vec3(1.0f));
		perp = perp - dir * glm::dot(perp, dir);
		if (glm::length(perp) > 0.001f) {
			perp = glm::normalize(perp);
		} else {
			perp = glm::vec3(1.0f, 0.0f, 0.0f);
		}

		float offsetMagnitude = dist * 0.15f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		mid += perp * offsetMagnitude;

		GenerateBolt(strike, start, mid, depth - 1);
		GenerateBolt(strike, mid, end, depth - 1);

		// Branching logic
		float branchRand = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		if (branchRand < _branchProbability && depth > 1) {
			glm::vec3 branchDir = glm::normalize(dir + perp * 0.5f);
			glm::vec3 branchEnd = mid + branchDir * (dist * 0.4f);
			GenerateBolt(strike, mid, branchEnd, depth - 2);
		}
	}

	void LightningManager::TriggerStrike(
		LightningType    type,
		const glm::vec3& startPos,
		const glm::vec3& endPos,
		const glm::vec3& color,
		LightManager&    lightManager
	) {
		LightningStrike strike;
		strike.id = _nextStrikeId++;
		strike.type = type;
		strike.lifetime = 0.0f;
		strike.maxLifetime = (0.2f + 0.3f * (rand() % 100) / 100.0f) * _lifetimeMultiplier;
		strike.intensity = _intensityMultiplier;
		strike.color = color;
		strike.hasSpawnedFlash = false;
		strike.driftVelocity = glm::vec3(
			(rand() % 100 / 100.0f - 0.5f) * 10.0f,
			0.0f,
			(rand() % 100 / 100.0f - 0.5f) * 10.0f
		);

		int recursionDepth = 4;
		if (type == LightningType::CLOUD_TO_CLOUD) {
			recursionDepth = 3;
		}

		GenerateBolt(strike, startPos, endPos, recursionDepth);

		// Spawn flash light
		float     flashRadius = (type == LightningType::CLOUD_TO_CLOUD) ? 150.0f : 300.0f;
		float     flashIntensity = 50.0f * strike.intensity;
		glm::vec3 flashPos = (startPos + endPos) * 0.5f;

		Light flash = Light::CreateFlash(flashPos, flashIntensity, color, flashRadius, 2.0f);
		flash.autoRemove = true;
		flash.SetEaseOut(strike.maxLifetime);

		strike.flashLightId = lightManager.AddLight(flash);
		strike.hasSpawnedFlash = true;

		_activeStrikes.push_back(strike);
	}

	void LightningManager::Update(float deltaTime, float /*totalTime*/, LightManager& lightManager) {
		_globalPulse = 0.0f;
		_globalColor = glm::vec3(0.0f);

		for (auto it = _activeStrikes.begin(); it != _activeStrikes.end();) {
			auto& strike = *it;
			strike.lifetime += deltaTime;

			if (strike.lifetime >= strike.maxLifetime) {
				if (strike.flashLightId != -1) {
					lightManager.RemoveLight(strike.flashLightId);
				}
				it = _activeStrikes.erase(it);
				continue;
			}

			// Drift strike segments
			for (auto& seg : strike.segments) {
				seg.start += strike.driftVelocity * deltaTime;
				seg.end += strike.driftVelocity * deltaTime;
			}

			float normalizedAge = strike.lifetime / strike.maxLifetime;

			// Rapid flicker pulse
			float pulse = std::sin(normalizedAge * 3.14159f) * (0.8f + 0.2f * std::sin(strike.lifetime * 50.0f));
			strike.intensity = pulse * _intensityMultiplier;

			if (strike.intensity > _globalPulse) {
				_globalPulse = strike.intensity;
				_globalColor = strike.color;
			}

			// Update flash light intensity if present
			if (strike.flashLightId != -1) {
				Light* flash = lightManager.GetLight(strike.flashLightId);
				if (flash) {
					flash->intensity = flash->baseIntensity * strike.intensity;
				}
			}

			++it;
		}
	}

} // namespace brassica
