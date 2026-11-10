#pragma once

#include "types/AutoExposureData.hpp"
#include "ui/IWidget.hpp"

namespace brassica::ui {

	class LightingWidget: public IWidget {
	public:
		LightingWidget();

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Lighting"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Lighting"; }

	private:
		// EV100 is the user-facing auto-exposure limit control this was ported from (boidish's
		// LightingWidget/BloomSettingsComponent) -- LayerDataHost only stores the derived
		// minExposure/maxExposure the shader actually reads, so the EV100 values themselves live
		// here and get reconverted into those every frame. Sized for kMaxExposureLayers (not just
		// the original 2 Scene/Sky slots) now that exposure layers are a dynamic, addable list.
		float m_minEV100[kMaxExposureLayers]{-10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f};
		float m_maxEV100[kMaxExposureLayers]{20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f};
	};

} // namespace brassica::ui
