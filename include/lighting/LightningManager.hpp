#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "lighting/ILightManager.hpp"

namespace brassica {

	enum class LightningType {
		BOLT,          // Sky to ground
		FORK,          // Branching sky to ground
		CLOUD_TO_CLOUD // Internal cloud flashes
	};

	struct LightningBoltSegment {
		glm::vec3 start;
		glm::vec3 end;
	};

	struct LightningStrike {
		int                               id = 0;
		LightningType                     type = LightningType::BOLT;
		std::vector<LightningBoltSegment> segments;
		float                             lifetime = 0.0f;    // Current duration
		float                             maxLifetime = constants::Class::Lighting::DefaultLightningMaxLifetime; // Total flash duration
		float                             intensity = 1.0f;   // Brightness [0-1]
		glm::vec3                         color{constants::Class::Lighting::DefaultLightningColor()};
		bool                              hasSpawnedFlash = false;
		int                               flashLightId = -1;
		glm::vec3                         driftVelocity{0.0f};
	};

	class LightningManager {
	public:
		LightningManager() = default;

		void Update(float deltaTime, float totalTime, ILightManager& lightManager);

		void TriggerStrike(
			LightningType    type,
			const glm::vec3& startPos,
			const glm::vec3& endPos,
			const glm::vec3& color,
			ILightManager&   lightManager
		);

		const std::vector<LightningStrike>& GetActiveStrikes() const { return _activeStrikes; }

		float GetGlobalPulse() const { return _globalPulse; }

		glm::vec3 GetGlobalColor() const { return _globalColor; }

		float GetIntensityMultiplier() const { return _intensityMultiplier; }

		void SetIntensityMultiplier(float m) { _intensityMultiplier = m; }

		float GetFrequencyMultiplier() const { return _frequencyMultiplier; }

		void SetFrequencyMultiplier(float m) { _frequencyMultiplier = m; }

		float GetLifetimeMultiplier() const { return _lifetimeMultiplier; }

		void SetLifetimeMultiplier(float m) { _lifetimeMultiplier = m; }

		float GetBranchProbability() const { return _branchProbability; }

		void SetBranchProbability(float p) { _branchProbability = p; }

		float GetThickness() const { return _thickness; }

		void SetThickness(float t) { _thickness = t; }

	private:
		void GenerateBolt(LightningStrike& strike, const glm::vec3& start, const glm::vec3& end, int depth);

		std::vector<LightningStrike> _activeStrikes;
		int                          _nextStrikeId = 0;

		float     _globalPulse = 0.0f;
		glm::vec3 _globalColor{0.0f};

		float _intensityMultiplier = 1.0f;
		float _frequencyMultiplier = 1.0f;
		float _lifetimeMultiplier = 1.0f;
		float _branchProbability = constants::Class::Lighting::DefaultLightningBranchProbability;
		float _thickness = constants::Class::Lighting::DefaultLightningThickness;
	};

} // namespace brassica
