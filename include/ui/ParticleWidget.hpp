#pragma once

#include "ui/IWidget.hpp"

namespace brassica::ui {

	class ParticleWidget: public IWidget {
	public:
		ParticleWidget();
		~ParticleWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Particle System"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Simulation"; }
	};

} // namespace brassica::ui
