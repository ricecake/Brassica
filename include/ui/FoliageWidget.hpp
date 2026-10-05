#pragma once

#include "ui/IWidget.hpp"

namespace brassica::ui {

	class FoliageWidget: public IWidget {
	public:
		FoliageWidget();
		~FoliageWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Foliage & Grass"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Environment"; }
	};

} // namespace brassica::ui
