#include "foliage/FoliageManager.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Graph.hpp"

namespace brassica {

	FoliageManager::FoliageManager() {
		m_state = FoliageState{};
		PopulateDefaults();
		SyncTableNode();
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
			SyncTableNode();
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
		// Indexed by shaders/helpers/whittaker.glsl's WhittakerBiome::biomeIndex (0-9), not an
		// arbitrary ramp -- each entry's character matches what that biome actually is.
		m_biomeProps[0] = GrassProperties{ // Ice / Glacier / Snow
			.colorTop = glm::vec4(0.80f, 0.85f, 0.80f, 1.0f),
			.colorBottom = glm::vec4(0.55f, 0.60f, 0.55f, 1.0f),
			.height = 0.3f, .width = 0.04f, .rigidity = 0.7f,
			.heightVariance = 0.2f, .widthVariance = 0.05f, .density = 0.15f,
			.colorVariability = 0.1f, .windInfluence = 0.6f, .enabled = 0u, .flowerRatio = 0.0f,
		};
		m_biomeProps[1] = GrassProperties{ // Tundra
			.colorTop = glm::vec4(0.45f, 0.48f, 0.30f, 1.0f),
			.colorBottom = glm::vec4(0.20f, 0.22f, 0.12f, 1.0f),
			.height = 0.35f, .width = 0.05f, .rigidity = 0.6f,
			.heightVariance = 0.3f, .widthVariance = 0.1f, .density = 0.5f,
			.colorVariability = 0.2f, .windInfluence = 0.9f, .enabled = 1u, .flowerRatio = 0.05f,
		};
		m_biomeProps[2] = GrassProperties{ // Taiga / Boreal Forest
			.colorTop = glm::vec4(0.18f, 0.40f, 0.18f, 1.0f),
			.colorBottom = glm::vec4(0.07f, 0.20f, 0.08f, 1.0f),
			.height = 0.8f, .width = 0.08f, .rigidity = 0.55f,
			.heightVariance = 0.3f, .widthVariance = 0.08f, .density = 0.9f,
			.colorVariability = 0.2f, .windInfluence = 0.8f, .enabled = 1u, .flowerRatio = 0.03f,
		};
		m_biomeProps[3] = GrassProperties{ // Cold Desert / Temperate Grassland
			.colorTop = glm::vec4(0.62f, 0.58f, 0.32f, 1.0f),
			.colorBottom = glm::vec4(0.30f, 0.26f, 0.12f, 1.0f),
			.height = 0.7f, .width = 0.07f, .rigidity = 0.5f,
			.heightVariance = 0.35f, .widthVariance = 0.1f, .density = 0.7f,
			.colorVariability = 0.25f, .windInfluence = 1.1f, .enabled = 1u, .flowerRatio = 0.08f,
		};
		m_biomeProps[4] = GrassProperties{ // Temperate Woodland / Shrubland
			.colorTop = glm::vec4(0.32f, 0.52f, 0.20f, 1.0f),
			.colorBottom = glm::vec4(0.12f, 0.28f, 0.08f, 1.0f),
			.height = 0.9f, .width = 0.09f, .rigidity = 0.5f,
			.heightVariance = 0.3f, .widthVariance = 0.1f, .density = 1.0f,
			.colorVariability = 0.2f, .windInfluence = 1.0f, .enabled = 1u, .flowerRatio = 0.12f,
		};
		m_biomeProps[5] = GrassProperties{ // Temperate Deciduous Forest
			.colorTop = glm::vec4(0.22f, 0.55f, 0.18f, 1.0f),
			.colorBottom = glm::vec4(0.08f, 0.30f, 0.06f, 1.0f),
			.height = 1.1f, .width = 0.10f, .rigidity = 0.45f,
			.heightVariance = 0.3f, .widthVariance = 0.1f, .density = 1.1f,
			.colorVariability = 0.2f, .windInfluence = 1.0f, .enabled = 1u, .flowerRatio = 0.18f,
		};
		m_biomeProps[6] = GrassProperties{ // Temperate Rainforest
			.colorTop = glm::vec4(0.10f, 0.48f, 0.14f, 1.0f),
			.colorBottom = glm::vec4(0.04f, 0.26f, 0.08f, 1.0f),
			.height = 1.3f, .width = 0.11f, .rigidity = 0.4f,
			.heightVariance = 0.35f, .widthVariance = 0.12f, .density = 1.3f,
			.colorVariability = 0.15f, .windInfluence = 0.7f, .enabled = 1u, .flowerRatio = 0.10f,
		};
		m_biomeProps[7] = GrassProperties{ // Subtropical Desert
			.colorTop = glm::vec4(0.72f, 0.60f, 0.35f, 1.0f),
			.colorBottom = glm::vec4(0.40f, 0.32f, 0.16f, 1.0f),
			.height = 0.4f, .width = 0.05f, .rigidity = 0.65f,
			.heightVariance = 0.4f, .widthVariance = 0.1f, .density = 0.2f,
			.colorVariability = 0.3f, .windInfluence = 1.2f, .enabled = 1u, .flowerRatio = 0.02f,
		};
		m_biomeProps[8] = GrassProperties{ // Savanna / Tropical Seasonal Forest
			.colorTop = glm::vec4(0.58f, 0.56f, 0.22f, 1.0f),
			.colorBottom = glm::vec4(0.28f, 0.28f, 0.10f, 1.0f),
			.height = 1.0f, .width = 0.09f, .rigidity = 0.45f,
			.heightVariance = 0.35f, .widthVariance = 0.1f, .density = 0.8f,
			.colorVariability = 0.25f, .windInfluence = 1.1f, .enabled = 1u, .flowerRatio = 0.10f,
		};
		m_biomeProps[9] = GrassProperties{ // Tropical Rainforest
			.colorTop = glm::vec4(0.08f, 0.46f, 0.12f, 1.0f),
			.colorBottom = glm::vec4(0.03f, 0.24f, 0.06f, 1.0f),
			.height = 1.4f, .width = 0.12f, .rigidity = 0.35f,
			.heightVariance = 0.4f, .widthVariance = 0.15f, .density = 1.4f,
			.colorVariability = 0.15f, .windInfluence = 0.6f, .enabled = 1u, .flowerRatio = 0.15f,
		};
	}

	void FoliageManager::RegisterTableNode(graph::Graph& frameGraph) {
		frameGraph.RegisterRef(m_biomeTableNode);
	}

	void FoliageManager::SyncTableNode() {
		// 5 texels wide x kFoliageBiomeCount rows, RGBA32F -- see FoliageBiomeTableTexture's
		// comment (ResourceKeys.hpp) for the texel layout.
		constexpr std::uint32_t kTexelsPerBiome = 5;
		std::array<glm::vec4, kTexelsPerBiome * kFoliageBiomeCount> texels{};

		for (std::uint32_t i = 0; i < kFoliageBiomeCount; ++i) {
			const GrassProperties& p = m_biomeProps[i];
			glm::vec4*             row = &texels[i * kTexelsPerBiome];
			row[0] = p.colorTop;
			row[1] = p.colorBottom;
			row[2] = glm::vec4(p.height, p.width, p.density, p.flowerRatio);
			row[3] = glm::vec4(p.rigidity, p.heightVariance, p.widthVariance, p.windInfluence);
			row[4] = glm::vec4(p.colorVariability, p.enabled != 0u ? 1.0f : 0.0f, 0.0f, 0.0f);
		}

		graph::ResourceDesc desc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = kTexelsPerBiome,
			.height = kFoliageBiomeCount,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(vk::ImageUsageFlagBits::eSampled),
			.persistent = true,
		};

		std::span<const std::uint8_t> bytes(
			reinterpret_cast<const std::uint8_t*>(texels.data()), texels.size() * sizeof(glm::vec4)
		);
		m_biomeTableNode.SetData(bytes, desc);
	}

} // namespace brassica
