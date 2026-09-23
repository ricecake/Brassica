#pragma once

#include <string>

#include "ui/IWidget.hpp"

namespace brassica::ui {

	class QuickSettingsWidget: public IWidget {
	public:
		QuickSettingsWidget();
		~QuickSettingsWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Quick Controls"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Settings"; }

	private:
		float m_timeOfDay{12.0f};
		bool  m_timePaused{false};
		bool  m_enableMood{true};
		bool  m_renderTerrain{true};
		bool  m_enableVolumetricLighting{true};
		float m_volumetricIntensity{1.0f};
		bool  m_enableAtmosphere{true};
		float m_cloudCoverage{0.3f};

		float       m_mapExtent{2048.0f};
		int         m_mapResolutionIndex{1}; // 1024x1024
		char        m_mapFilename[128]{"terrain_map.png"};
		std::string m_mapExportStatus{""};
	};

} // namespace brassica::ui
