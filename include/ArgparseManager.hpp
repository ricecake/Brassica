#pragma once

#include <memory>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>

#include "IManager.hpp"

namespace brassica {

	struct ArgparseManagerState {
		bool        headless{false};
		uint32_t    maxFrames{0};
		std::string configFile{"config.ini"};
		std::string appName{"Sandbox"};
		bool        renderTerrainMap{false};
		std::string terrainMapPath{"terrain_map.png"};
		bool        aggressiveValidation{false};
		bool        printRenderGraph{false};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("headless", "Headless Mode", &ArgparseManagerState::headless),
				MakeField(
					"maxFrames",
					"Max Frames (0 = unlimited)",
					&ArgparseManagerState::maxFrames,
					0u,
					100000u,
					UIHint::Slider
				),
				MakeField("configFile", "Config File", &ArgparseManagerState::configFile),
				MakeField("appName", "Application Name", &ArgparseManagerState::appName),
				MakeField("renderTerrainMap", "Render Terrain Map", &ArgparseManagerState::renderTerrainMap),
				MakeField("terrainMapPath", "Terrain Map Output Path", &ArgparseManagerState::terrainMapPath),
				MakeField(
					"aggressiveValidation",
					"Aggressive Vulkan Validation (GPU-assisted + sync, slow)",
					&ArgparseManagerState::aggressiveValidation
				),
				MakeField("printRenderGraph", "Print Render Graph", &ArgparseManagerState::printRenderGraph)
			);
		}
	};

	class ArgparseManager: public ManagerBase<ArgparseManager, ArgparseManagerState> {
	public:
		using State = ArgparseManagerState;

		ArgparseManager();
		explicit ArgparseManager(const std::string& programName, const std::string& version = "1.0.0");
		~ArgparseManager() override = default;

		void Initialize() override;
		void Shutdown() override;

		std::string GetManagerName() const override { return "ArgparseManager"; }

		State GetState() const override {
			return State{
				.headless = GetHeadless(),
				.maxFrames = GetMaxFrames(),
				.configFile = GetConfigFile(),
				.appName = GetAppName(),
				.renderTerrainMap = GetRenderTerrainMap(),
				.terrainMapPath = GetTerrainMapPath(),
				.aggressiveValidation = GetAggressiveValidation(),
				.printRenderGraph = GetPrintRenderGraph()
			};
		}

		void SetState(const State& state) override {
			(void)state; // CLI parameters are parsed at startup
		}

		bool Parse(int argc, char** argv);
		bool Parse(const std::vector<std::string>& args);

		[[nodiscard]] bool        GetHeadless() const;
		[[nodiscard]] uint32_t    GetMaxFrames() const;
		[[nodiscard]] std::string GetConfigFile() const;
		[[nodiscard]] std::string GetAppName() const;
		[[nodiscard]] bool        GetRenderTerrainMap() const;
		[[nodiscard]] std::string GetTerrainMapPath() const;
		[[nodiscard]] bool        GetAggressiveValidation() const;
		[[nodiscard]] bool        GetPrintRenderGraph() const;

		argparse::ArgumentParser& GetParser() { return m_parser; }

		const argparse::ArgumentParser& GetParser() const { return m_parser; }

	private:
		void SetupArguments();

		argparse::ArgumentParser m_parser;
		bool                     m_argsParsed{false};
	};

} // namespace brassica
