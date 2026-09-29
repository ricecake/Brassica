#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ConfigManager.hpp"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainClipmap.hpp"
#include "EngineConstants.hpp"
#include "types/CameraData.hpp"
#include "types/TonemapPushConstants.hpp"
#include "ui/IWidget.hpp"
#include "ui/ManagerSettingsWidget.hpp"
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

	TEST_CASE("Manager Active State Return, Accept, and Reflection") {
		LightManager lightMgr;
		lightMgr.Initialize();

		// Check manager name
		CHECK(lightMgr.GetManagerName() == "LightManager");

		// Get active state struct
		LightManager::State state = lightMgr.GetState();
		CHECK(state.cycleTime == constants::Class::Lighting::DefaultCycleTime);
		CHECK(state.skyExposure == constants::Class::Lighting::DefaultSkyExposure);

		// Modify state struct
		state.cycleTime = 18.5f;
		state.skyExposure = 3.2f;
		state.ambientLight = glm::vec3(0.5f, 0.4f, 0.3f);
		state.cyclePaused = true;

		// Accept state struct
		lightMgr.SetState(state);

		// Verify manager updated
		LightManager::State newState = lightMgr.GetState();
		CHECK(newState.cycleTime == 18.5f);
		CHECK(newState.skyExposure == 3.2f);
		CHECK(newState.ambientLight == glm::vec3(0.5f, 0.4f, 0.3f));
		CHECK(newState.cyclePaused == true);
	}

	TEST_CASE("Automatic Manager State Serialization with ConfigManager") {
		ConfigManager config("TestApp");
		config.Initialize();

		LightManager lightMgr;
		lightMgr.Initialize();

		auto state = lightMgr.GetState();
		state.cycleTime = 14.0f;
		state.skyExposure = 2.5f;
		lightMgr.SetState(state);

		// Save state automatically
		lightMgr.SaveState(config);

		// Verify settings exist under Application.TestApp.LightManager or Manager.LightManager
		CHECK(config.HasValue("Application.TestApp.LightManager", "cycleTime"));
		CHECK(config.GetManagerSetting<float>("LightManager", "cycleTime", 0.0f) == 14.0f);
		CHECK(config.GetManagerSetting<float>("LightManager", "skyExposure", 0.0f) == 2.5f);

		// Instantiate new manager and load state automatically
		LightManager newLightMgr;
		newLightMgr.Initialize();
		newLightMgr.LoadState(config);

		auto loadedState = newLightMgr.GetState();
		CHECK(loadedState.cycleTime == 14.0f);
		CHECK(loadedState.skyExposure == 2.5f);
	}

	TEST_CASE("ManagerSettingsWidget Creation") {
		ui::ManagerSettingsWidget widget;
		CHECK(widget.GetTitle() == "Manager Settings");
		CHECK(widget.GetCategory() == "Settings");
		CHECK_FALSE(widget.IsVisible());
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

	TEST_CASE("Shortest Distance Wrap Helpers") {
		float R = FAKE_PLANET_RADIUS;
		float halfL = FAKE_PLANET_HALF_PERIMETER; // ~1,884,955.59m
		float L = FAKE_PLANET_PERIMETER;           // ~3,769,911.18m

		// Small delta should remain unchanged
		glm::vec2 smallDelta(50.0f, -20.0f);
		glm::vec2 wrappedSmall = WrapShortestDistance(smallDelta);
		CHECK(doctest::Approx(wrappedSmall.x).epsilon(0.001) == 50.0f);
		CHECK(doctest::Approx(wrappedSmall.y).epsilon(0.001) == -20.0f);

		// Delta spanning across wrap line (> halfL)
		// e.g. Object at -1.8e6, Camera at +1.8e6 => diff = -3.6e6
		glm::vec2 largeNegDelta(-3600000.0f, 0.0f);
		glm::vec2 wrappedNeg = WrapShortestDistance(largeNegDelta);
		// Should pick shortest vector (+169,911.18m)
		CHECK(wrappedNeg.x > 0.0f);
		CHECK(doctest::Approx(wrappedNeg.x).epsilon(0.01) == (-3600000.0f + L));

		// Opposite direction (> +halfL)
		glm::vec2 largePosDelta(3600000.0f, 0.0f);
		glm::vec2 wrappedPos = WrapShortestDistance(largePosDelta);
		CHECK(wrappedPos.x < 0.0f);
		CHECK(doctest::Approx(wrappedPos.x).epsilon(0.01) == (3600000.0f - L));

		// 3D vector variant
		glm::vec3 delta3D(-3600000.0f, 15.0f, 0.0f);
		glm::vec3 wrapped3D = WrapShortestDistance(delta3D);
		CHECK(doctest::Approx(wrapped3D.y).epsilon(0.001) == 15.0f);
		CHECK(doctest::Approx(wrapped3D.x).epsilon(0.01) == (-3600000.0f + L));
	}

	TEST_CASE("Camera Position Wrap and Offset Tracking") {
		float halfL = FAKE_PLANET_HALF_PERIMETER;
		float L = FAKE_PLANET_PERIMETER;

		// Position just below wrap boundary
		glm::vec3 posBefore(halfL - 5.0f, 10.0f, 0.0f);
		glm::vec3 step(10.0f, 0.0f, 0.0f); // moves across wrap boundary

		glm::vec3 posTemp = posBefore + step; // halfL + 5.0f
		glm::vec3 posWrapped = posTemp;
		glm::vec3 wrapOffset{0.0f};

		if (posWrapped.x > halfL) {
			posWrapped.x -= L;
			wrapOffset.x = -L;
		}

		// Verify new camera position wrapped to negative domain
		CHECK(posWrapped.x < 0.0f);
		CHECK(doctest::Approx(posWrapped.x).epsilon(0.01) == (-halfL + 5.0f));
		CHECK(wrapOffset.x == -L);

		// Verify delta calculation using wrapOffset
		glm::vec3 prevAdjusted = posBefore + wrapOffset;
		glm::vec3 frameDelta = posWrapped - prevAdjusted; // Should equal true step (10, 0, 0)
		CHECK(doctest::Approx(frameDelta.x).epsilon(0.001) == 10.0f);
		CHECK(doctest::Approx(frameDelta.y).epsilon(0.001) == 0.0f);
		CHECK(doctest::Approx(frameDelta.z).epsilon(0.001) == 0.0f);
	}

	TEST_CASE("Pseudo Sphere Surface Normal and Celestial Continuity") {
		float halfL = FAKE_PLANET_HALF_PERIMETER;
		float R = FAKE_PLANET_RADIUS;

		auto getCamNormal = [R](float x, float z) {
			float theta = x / R;
			float phi = z / R;
			return glm::normalize(glm::vec3(
				std::sin(theta) * std::cos(phi),
				std::cos(theta) * std::cos(phi),
				std::sin(phi)
			));
		};

		// At origin (0,0), normal is (0,1,0)
		glm::vec3 normOrigin = getCamNormal(0.0f, 0.0f);
		CHECK(doctest::Approx(normOrigin.x).epsilon(0.001) == 0.0f);
		CHECK(doctest::Approx(normOrigin.y).epsilon(0.001) == 1.0f);
		CHECK(doctest::Approx(normOrigin.z).epsilon(0.001) == 0.0f);

		// At boundary +halfL (+pi * R) vs -halfL (-pi * R)
		glm::vec3 normPlusBoundary = getCamNormal(halfL, 0.0f);
		glm::vec3 normMinusBoundary = getCamNormal(-halfL, 0.0f);

		// Normals at +pi*R and -pi*R must be identical (0, -1, 0) -> continuous across warp
		CHECK(doctest::Approx(normPlusBoundary.x).epsilon(0.001) == normMinusBoundary.x);
		CHECK(doctest::Approx(normPlusBoundary.y).epsilon(0.001) == normMinusBoundary.y);
		CHECK(doctest::Approx(normPlusBoundary.z).epsilon(0.001) == normMinusBoundary.z);
		CHECK(doctest::Approx(normPlusBoundary.y).epsilon(0.001) == -1.0f);
	}

} // namespace brassica
