#include "foliage/FoliageManager.hpp"

#include <algorithm>

namespace brassica {

	FoliageManager::FoliageManager() {
		m_state = FoliageState{};
		PopulateDefaults();
	}

	bool FoliageManager::IsEnabled() const {
		return m_state.enabled;
	}

	void FoliageManager::SetEnabled(bool enabled) {
		m_state.enabled = enabled;
	}

	const GrassProperties& FoliageManager::GetBiomeProperties(uint32_t biomeIdx) const {
		uint32_t idx = std::min(biomeIdx, static_cast<uint32_t>(m_biomeProps.size() - 1));
		return m_biomeProps[idx];
	}

	void FoliageManager::SetBiomeProperties(uint32_t biomeIdx, const GrassProperties& props) {
		if (biomeIdx < m_biomeProps.size()) {
			m_biomeProps[biomeIdx] = props;
		}
	}

	GlobalGrassProperties FoliageManager::GetGlobalProperties() const {
		return GlobalGrassProperties{
			.lengthMultiplier = m_state.lengthMultiplier,
			.widthMultiplier = m_state.widthMultiplier,
			.densityMultiplier = m_state.densityMultiplier,
			.rigidityMultiplier = m_state.rigidityMultiplier,
			.windMultiplier = m_state.windMultiplier,
			.enabled = m_state.enabled ? 1u : 0u,
			.lodScaleFactor = m_state.lodScaleFactor,
			.lodBaseRange = m_state.lodBaseRange,
			.baseScale = m_state.baseScale,
			.flowerRatio = m_state.flowerRatio,
			.fernRatio = m_state.fernRatio,
			.rockRatio = m_state.rockRatio,
			.seaweedRatio = m_state.seaweedRatio,
			.bushRatio = m_state.bushRatio,
			.baseTileSize = m_state.baseTileSize,
			.maxLODs = m_state.maxLODs,
			.tilesPerRow = m_state.tilesPerRow,
		};
	}

	void FoliageManager::SetGlobalProperties(const GlobalGrassProperties& props) {
		m_state.lengthMultiplier = props.lengthMultiplier;
		m_state.widthMultiplier = props.widthMultiplier;
		m_state.densityMultiplier = props.densityMultiplier;
		m_state.rigidityMultiplier = props.rigidityMultiplier;
		m_state.windMultiplier = props.windMultiplier;
		m_state.enabled = (props.enabled != 0);
		m_state.lodScaleFactor = props.lodScaleFactor;
		m_state.lodBaseRange = props.lodBaseRange;
		m_state.baseScale = props.baseScale;
		m_state.flowerRatio = props.flowerRatio;
		m_state.fernRatio = props.fernRatio;
		m_state.rockRatio = props.rockRatio;
		m_state.seaweedRatio = props.seaweedRatio;
		m_state.bushRatio = props.bushRatio;
		m_state.baseTileSize = props.baseTileSize;
		m_state.maxLODs = props.maxLODs;
		m_state.tilesPerRow = props.tilesPerRow;
	}

	FoliageState FoliageManager::GetState() const {
		return m_state;
	}

	void FoliageManager::SetState(const FoliageState& state) {
		m_state = state;
	}

	void FoliageManager::PopulateDefaults() {
		// Default biome grass properties
		for (size_t i = 0; i < m_biomeProps.size(); ++i) {
			m_biomeProps[i] = GrassProperties{
				.colorTop = glm::vec4(0.2f + 0.05f * i, 0.7f - 0.02f * i, 0.15f, 1.0f),
				.colorBottom = glm::vec4(0.08f + 0.02f * i, 0.25f, 0.05f, 1.0f),
				.height = 1.0f + 0.1f * i,
				.width = 0.08f + 0.01f * i,
				.rigidity = 0.5f,
				.heightVariance = 0.3f,
				.widthVariance = 0.05f,
				.density = 1.0f,
				.colorVariability = 0.2f,
				.windInfluence = 1.0f,
				.enabled = 1u,
				.flowerRatio = (i % 2 == 0) ? 0.15f : 0.05f,
			};
		}
	}

} // namespace brassica
