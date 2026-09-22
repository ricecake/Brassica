#pragma once

#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "efsw/efsw.hpp"
#include "vulkan/vulkan.hpp"

#include "IManager.hpp"
#include "Shader.hpp"

namespace brassica {

	struct ShaderWatcherState {
		bool        enabled{true};
		std::string watchedDir{"shaders"};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("enabled", "Enable Shader Hot Reloading", &ShaderWatcherState::enabled),
				MakeField("watchedDir", "Watched Directory", &ShaderWatcherState::watchedDir)
			);
		}
	};

	class ShaderWatcher: public ManagerBase<ShaderWatcher, ShaderWatcherState>, public efsw::FileWatchListener {
	public:
		using ReloadCallback = std::function<void()>;
		using State = ShaderWatcherState;

		ShaderWatcher();
		~ShaderWatcher() override;

		void Initialize() override;
		void Shutdown() override;

		std::string GetManagerName() const override { return "ShaderWatcher"; }

		State GetState() const override { return State{watchID != -1, watchedDir}; }

		void SetState(const State& state) override {
			if (state.enabled && watchID == -1 && !state.watchedDir.empty()) {
				WatchDirectory(state.watchedDir);
			} else if (!state.enabled && watchID != -1) {
				StopWatching();
			}
		}

		bool WatchDirectory(const std::string& directory, bool recursive = true);
		void StopWatching();

		void RegisterShader(Shader* shader, ReloadCallback onReload = nullptr);
		void RegisterFile(const std::string& filepath, ReloadCallback onReload);

		bool ProcessPendingReloads(vk::Device device);

		void handleFileAction(
			efsw::WatchID      watchid,
			const std::string& dir,
			const std::string& filename,
			efsw::Action       action,
			const std::string& oldFilename = ""
		) override;

		static std::string NormalizePath(const std::string& path);

	private:
		efsw::FileWatcher fileWatcher;
		efsw::WatchID     watchID{-1};
		std::string       watchedDir;

		std::mutex            mutex;
		std::set<std::string> pendingModifiedFiles;

		std::unordered_map<std::string, std::vector<Shader*>>        shaderRegistry;
		std::unordered_map<Shader*, std::vector<ReloadCallback>>     shaderCallbacks;
		std::unordered_map<std::string, std::vector<ReloadCallback>> fileCallbacks;
	};

} // namespace brassica
