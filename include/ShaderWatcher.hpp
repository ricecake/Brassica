#pragma once

#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "efsw/efsw.hpp"
#include "vulkan/vulkan.hpp"

#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher: public efsw::FileWatchListener {
	public:
		using ReloadCallback = std::function<void()>;

		ShaderWatcher();
		~ShaderWatcher() override;

		bool WatchDirectory(const std::string& directory, bool recursive = true);
		void StopWatching();

		void RegisterShader(Shader* shader, ReloadCallback onReload = nullptr);
		void RegisterFile(const std::string& filepath, ReloadCallback onReload);

		void ProcessPendingReloads(vk::Device device);

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
