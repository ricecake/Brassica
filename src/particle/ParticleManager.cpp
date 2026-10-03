#include "particle/ParticleManager.hpp"

#include <algorithm>

namespace brassica {

	void ParticleManager::GetCutoffs(float& outBirdCutoff, float& outFishCutoff) const {
		float birdProp = std::max(0.0f, m_state.birdProportion);
		float fishProp = std::max(0.0f, m_state.fishProportion);
		float fireflyProp = std::max(0.0f, m_state.fireflyProportion);

		float total = birdProp + fishProp + fireflyProp;
		if (total <= 0.0001f) {
			outBirdCutoff = 0.3333f;
			outFishCutoff = 0.6666f;
			return;
		}

		float birdNorm = birdProp / total;
		float fishNorm = fishProp / total;

		outBirdCutoff = birdNorm;
		outFishCutoff = birdNorm + fishNorm;
	}

} // namespace brassica
