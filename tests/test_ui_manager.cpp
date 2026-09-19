#include "doctest/doctest.h"

#include "ConfigManager.hpp"
#include "ServiceLocator.hpp"
#include "ui/IWidget.hpp"
#include "ui/QuickSettingsWidget.hpp"

namespace brassica {

	class TestWidget: public ui::IWidget {
	public:
		TestWidget(std::string title, std::string category = "TestCategory", bool isHud = false)
			: m_title(std::move(title)), m_category(std::move(category)), m_isHud(isHud) {}

		void Draw() override { m_drawCount++; }

		[[nodiscard]] std::string_view GetTitle() const override { return m_title; }
		[[nodiscard]] std::string_view GetCategory() const override { return m_category; }
		[[nodiscard]] bool             IsHud() const override { return m_isHud; }

		int m_drawCount{0};

	private:
		std::string m_title;
		std::string m_category;
		bool        m_isHud;
	};

	TEST_CASE("IWidget Visibility and Virtual Base Interface") {
		TestWidget widget("Test Window", "Windows", false);
		CHECK(widget.GetTitle() == "Test Window");
		CHECK(widget.GetCategory() == "Windows");
		CHECK(widget.IsHud() == false);
		CHECK(widget.IsVisible() == true);

		widget.SetVisible(false);
		CHECK(widget.IsVisible() == false);

		widget.Draw();
		CHECK(widget.m_drawCount == 1);
	}

	TEST_CASE("QuickSettingsWidget Configuration Initialization") {
		ServiceLocator locator;
		ServiceLocator::SetInstance(&locator);
		auto cfg = std::make_shared<ConfigManager>();
		locator.Provide<ConfigManager>(cfg);

		ui::QuickSettingsWidget quickWidget;
		CHECK(quickWidget.GetTitle() == "Quick Controls");
		CHECK(quickWidget.GetCategory() == "Settings");
		CHECK(quickWidget.IsHud() == false);
		CHECK(quickWidget.IsVisible() == true);

		ServiceLocator::SetInstance(nullptr);
	}

} // namespace brassica
