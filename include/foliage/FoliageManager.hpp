#pragma once

#include <array>
#include <memory>

#include "foliage/IFoliageManager.hpp"

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

	private:
		std::array<GrassProperties, 8> m_biomeProps{};
		FoliageState                   m_state{};
		void PopulateDefaults();
	};

} // namespace brassica
