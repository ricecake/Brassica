#pragma once

#include "ui/IWidget.hpp"

namespace brassica::ui {

	class ManagerSettingsWidget: public IWidget {
	public:
		ManagerSettingsWidget();
		~ManagerSettingsWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Manager Settings"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Settings"; }
	};

} // namespace brassica::ui
