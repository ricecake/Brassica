#include "doctest/doctest.h"

#include "ConfigManager.hpp"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainClipmap.hpp"
#include "EngineConstants.hpp"
#include "types/CameraData.hpp"
#include "types/TonemapPushConstants.hpp"
#include "ui/IWidget.hpp"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>
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

	TEST_CASE("CameraData Speed and Orientation in Degrees") {
		CameraData cam{};
		cam.speed = 42.5f;
		cam.pitch = -0.5f;
		cam.yaw = 1.0f;
		cam.roll = 0.0f;

		CHECK(cam.speed == 42.5f);
		CHECK(doctest::Approx(glm::degrees(cam.pitch)).epsilon(0.01) == -28.6479);
		CHECK(doctest::Approx(glm::degrees(cam.yaw)).epsilon(0.01) == 57.2958);
		CHECK(doctest::Approx(glm::degrees(cam.roll)).epsilon(0.01) == 0.0);
	}

	TEST_CASE("Local Camera Sky Frame Direction Transformation") {
		glm::vec3 sunDirGlobal(0.0f, 1.0f, 0.0f); // Zenith at origin

		// Position on top of planet: (0, 100, 0)
		glm::vec3 planetCenter(0.0f, -FAKE_PLANET_RADIUS, 0.0f);
		glm::vec3 camPosTop(0.0f, 100.0f, 0.0f);
		glm::vec3 camNormalTop = glm::normalize(camPosTop - planetCenter);

		glm::vec3 upRef(0.0f, 1.0f, 0.0f);
		float cosThetaTop = glm::dot(upRef, camNormalTop);
		CHECK(doctest::Approx(cosThetaTop).epsilon(0.0001) == 1.0f);

		// Position moved around planet curvature (chasing sun / moving over horizon)
		// e.g., 90 deg around X axis
		glm::vec3 camPos90(0.0f, -FAKE_PLANET_RADIUS, FAKE_PLANET_RADIUS + 100.0f);
		glm::vec3 camNormal90 = glm::normalize(camPos90 - planetCenter); // Should be (0, 0, 1)

		float cosTheta90 = glm::dot(upRef, camNormal90);
		glm::vec3 rotAxis = glm::cross(upRef, camNormal90);
		float s = std::sqrt((1.0f + cosTheta90) * 2.0f);
		float invs = 1.0f / s;
		glm::quat rotToCam(s * 0.5f, rotAxis.x * invs, rotAxis.y * invs, rotAxis.z * invs);

		glm::vec3 sunDirLocal = glm::inverse(rotToCam) * sunDirGlobal;
		// Since camera moved 90 deg, sun in local sky frame should now be pointing along -Z (horizon)
		CHECK(doctest::Approx(sunDirLocal.x).epsilon(0.001) == 0.0f);
		CHECK(doctest::Approx(sunDirLocal.y).epsilon(0.001) == 0.0f);
		CHECK(doctest::Approx(sunDirLocal.z).epsilon(0.001) == -1.0f);
	}

} // namespace brassica
