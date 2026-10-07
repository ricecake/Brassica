#pragma once

#include <array>
#include <memory>

#include "foliage/IFoliageManager.hpp"
#include "graph/Node.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"

namespace brassica {

	class FoliageManager final: public IFoliageManager {
	public:
		FoliageManager();
		~FoliageManager() override = default;

		[[nodiscard]] bool IsEnabled() const override;
		void               SetEnabled(bool enabled) override;

		[[nodiscard]] const GrassProperties& GetBiomeProperties(uint32_t biomeIdx) const override;
		void SetBiomeProperties(uint32_t biomeIdx, const GrassProperties& props) override;

		[[nodiscard]] GlobalGrassProperties GetGlobalProperties() const override;
		void SetGlobalProperties(const GlobalGrassProperties& props) override;

		void Initialize() override {}
		[[nodiscard]] FoliageState GetState() const override;
		void SetState(const FoliageState& state) override;

		// Registers the host-written per-biome properties texture (FoliageBiomeTableTexture) into
		// this frame's graph -- same RegisterBufferNode-into-Engine::DrawFrame pattern
		// MaterialManager already uses for MaterialBuffer, just a texture instead of a buffer
		// since each biome needs more than one SSBO-unfriendly vec4's worth of fields.
		void RegisterTableNode(graph::Graph& frameGraph);

	private:
		std::array<GrassProperties, kFoliageBiomeCount>                     m_biomeProps{};
		FoliageState                                                        m_state{};
		graph::PredefinedTextureNode<FoliageBiomeTableTexture, SubPhase::Prepare> m_biomeTableNode;

		void PopulateDefaults();
		// Rebuilds m_biomeTableNode's pixel data from m_biomeProps -- called whenever a biome's
		// properties change, not just once at startup, so the UI's "Per-Biome Grass Properties"
		// panel actually reaches the GPU instead of dead-ending in m_biomeProps the way it used to.
		void SyncTableNode();
	};

} // namespace brassica
