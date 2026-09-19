#pragma once

#include <string_view>

namespace brassica::ui {

	class IWidget {
	public:
		virtual ~IWidget() = default;

		virtual void Draw() = 0;

		[[nodiscard]] virtual std::string_view GetTitle() const = 0;

		[[nodiscard]] virtual std::string_view GetCategory() const { return "Windows"; }

		[[nodiscard]] virtual bool IsHud() const { return false; }

		virtual void SetVisible(bool visible) { m_visible = visible; }

		[[nodiscard]] virtual bool IsVisible() const { return m_visible; }

	protected:
		bool m_visible{true};
	};

} // namespace brassica::ui
