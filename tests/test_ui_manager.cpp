#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ConfigManager.hpp"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainClipmap.hpp"
#include "types/TonemapPushConstants.hpp"
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

	class MockTerrainClipmap: public ITerrainClipmap {
	public:
		void Initialize() override { m_initialized = true; }
		void Shutdown() override { m_initialized = false; }
		void Regenerate() override { m_regenerated = true; }

		bool m_regenerated{false};
	};

	TEST_CASE("ITerrainClipmap Service Locator and Regeneration") {
		ServiceLocator locator;
		ServiceLocator::SetInstance(&locator);

		auto terrainMock = std::make_shared<MockTerrainClipmap>();
		terrainMock->Initialize();
		locator.Provide<ITerrainClipmap>(terrainMock);

		CHECK(locator.Has<ITerrainClipmap>());
		auto retrieved = locator.Get<ITerrainClipmap>();
		CHECK_FALSE(terrainMock->m_regenerated);

		retrieved->Regenerate();
		CHECK(terrainMock->m_regenerated);

		ServiceLocator::SetInstance(nullptr);
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

	TEST_CASE("TonemapPushConstants Post-Processing and CDL Parameters") {
		TonemapPushConstants push{};
		CHECK(push.toneMapMode == 5); // Default Uchimura
		CHECK(push.exposure == 1.0f);
		CHECK(push.cdlSlope == glm::vec4(1.0f));
		CHECK(push.cdlOffset == glm::vec4(0.0f));
		CHECK(push.cdlPower == glm::vec4(1.0f));
		CHECK(push.cdlSaturation == 1.0f);
	}

	TEST_CASE("LightManager Physical Values and Day/Night Cycle") {
		LightManager lightMgr;
		lightMgr.Update(0.016f);

		const auto& lights = lightMgr.GetLights();
		REQUIRE(lights.size() >= 2);

		// Directional Sun physical radiance
		CHECK(lights[0].type == DIRECTIONAL_LIGHT);
		CHECK(lights[0].baseIntensity == 10.0f);

		// Directional Moon reflected radiance
		CHECK(lights[1].type == DIRECTIONAL_LIGHT);
		CHECK(lights[1].baseIntensity >= 0.0f);
	}

} // namespace brassica
