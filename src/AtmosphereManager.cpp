#include "AtmosphereManager.hpp"

#include "imgui.h"

namespace brassica {

	bool AtmosphereManager::DrawUI() {
		State state = GetState();
		bool  modified = DrawReflectedStateUI(GetManagerName(), state);

		// Tiny-magnitude physical coefficients (~1e-3): displayed/edited x1000 so ColorEdit3
		// shows a usable color instead of near-black -- same convention boidish's
		// EnvironmentWidget.cpp uses for ozone absorption specifically; this extends it to the
		// sibling rayleigh/mie coefficients for consistency.
		glm::vec3 rayleighDisplay = state.rayleighScattering * 1000.0f;
		if (ImGui::ColorEdit3(
				"Rayleigh Scattering (x1e-3)##atmosphere", &rayleighDisplay.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float
			)) {
			state.rayleighScattering = rayleighDisplay * 0.001f;
			modified = true;
		}

		float mieScatterDisplay = state.mieScattering * 1000.0f;
		if (ImGui::SliderFloat("Mie Scattering (x1e-3)##atmosphere", &mieScatterDisplay, 0.0f, 50.0f)) {
			state.mieScattering = mieScatterDisplay * 0.001f;
			modified = true;
		}

		float mieExtinctionDisplay = state.mieExtinction * 1000.0f;
		if (ImGui::SliderFloat("Mie Extinction (x1e-3)##atmosphere", &mieExtinctionDisplay, 0.0f, 50.0f)) {
			state.mieExtinction = mieExtinctionDisplay * 0.001f;
			modified = true;
		}

		glm::vec3 ozoneDisplay = state.ozoneAbsorption * 1000.0f;
		if (ImGui::ColorEdit3(
				"Ozone Absorption (x1e-3)##atmosphere", &ozoneDisplay.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float
			)) {
			state.ozoneAbsorption = ozoneDisplay * 0.001f;
			modified = true;
		}

		if (modified) {
			SetState(state);
		}
		return modified;
	}

} // namespace brassica
