#pragma once

#include "ui/IWidget.hpp"

namespace brassica::ui {

	class AudioWidget: public IWidget {
	public:
		AudioWidget();
		~AudioWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Audio Settings"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Audio"; }
	};

} // namespace brassica::ui
