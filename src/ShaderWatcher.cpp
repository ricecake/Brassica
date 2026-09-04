#include "ShaderWatcher.hpp"

#include <algorithm>
#include <filesystem>

#include "spdlog/spdlog.h"

namespace brassica {

	ShaderWatcher::ShaderWatcher() {}

	ShaderWatcher::~ShaderWatcher() {
		StopWatching();
	}

	std::string ShaderWatcher::NormalizePath(const std::string& path) {
		std::filesystem::path p(path);
		std::string           normalized = std::filesystem::weakly_canonical(p).string();
		std::replace(normalized.begin(), normalized.end(), '\\', '/');
		return normalized;
	}

	bool ShaderWatcher::WatchDirectory(const std::string& directory, bool recursive) {
		std::filesystem::path p(directory);
		if (!std::filesystem::exists(p)) {
			spdlog::warn("ShaderWatcher: Directory does not exist: {}", directory);
			return false;
		}

		watchedDir = NormalizePath(directory);

		watchID = fileWatcher.addWatch(directory, this, recursive);
		if (watchID < 0) {
			spdlog::error("ShaderWatcher: Failed to watch directory: {}", directory);
			return false;
		}

		fileWatcher.watch();
		spdlog::info("ShaderWatcher: Watching directory '{}' for shader changes.", directory);
		return true;
	}

	void ShaderWatcher::StopWatching() {
		if (watchID >= 0) {
			fileWatcher.removeWatch(watchID);
			watchID = -1;
		}
	}

	void ShaderWatcher::RegisterShader(Shader* shader, ReloadCallback onReload) {
		if (!shader)
			return;
		std::string filepath = shader->GetFilePath();
		if (filepath.empty()) {
			spdlog::warn("ShaderWatcher: Cannot register shader with empty filepath.");
			return;
		}

		std::lock_guard<std::mutex> lock(mutex);

		auto addPathToRegistry = [this, shader](const std::string& path) {
			if (path.empty())
				return;
			std::string normalized = NormalizePath(path);
			auto&       vec = shaderRegistry[normalized];
			if (std::find(vec.begin(), vec.end(), shader) == vec.end()) {
				vec.push_back(shader);
			}
		};

		addPathToRegistry(filepath);
		for (const auto& incFile : shader->GetIncludedFiles()) {
			addPathToRegistry(incFile);
		}

		if (onReload) {
			shaderCallbacks[shader].push_back(onReload);
		}
	}

	void ShaderWatcher::RegisterFile(const std::string& filepath, ReloadCallback onReload) {
		if (filepath.empty() || !onReload)
			return;

		std::string normalized = NormalizePath(filepath);

		std::lock_guard<std::mutex> lock(mutex);
		fileCallbacks[normalized].push_back(onReload);
	}

	void ShaderWatcher::handleFileAction(
		efsw::WatchID      watchid,
		const std::string& dir,
		const std::string& filename,
		efsw::Action       action,
		const std::string& oldFilename
	) {
		(void)watchid;
		(void)oldFilename;

		if (action == efsw::Actions::Modified || action == efsw::Actions::Add) {
			std::string fullPath = dir + filename;
			std::string normalized = NormalizePath(fullPath);

			std::lock_guard<std::mutex> lock(mutex);
			pendingModifiedFiles.insert(normalized);
		}
	}

	void ShaderWatcher::ProcessPendingReloads(vk::Device device) {
		std::set<std::string> filesToProcess;
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (pendingModifiedFiles.empty()) {
				return;
			}
			filesToProcess.swap(pendingModifiedFiles);
		}

		for (const auto& filepath : filesToProcess) {
			spdlog::info("ShaderWatcher: File change detected: {}", filepath);

			std::vector<Shader*>        shadersToReload;
			std::vector<ReloadCallback> callbacksToInvoke;

			{
				std::lock_guard<std::mutex> lock(mutex);
				auto                        shaderIt = shaderRegistry.find(filepath);
				if (shaderIt != shaderRegistry.end()) {
					shadersToReload = shaderIt->second;
				}

				auto fileCbIt = fileCallbacks.find(filepath);
				if (fileCbIt != fileCallbacks.end()) {
					callbacksToInvoke.insert(callbacksToInvoke.end(), fileCbIt->second.begin(), fileCbIt->second.end());
				}
			}

			bool recompiledAny = false;
			std::set<Shader*> processedShaders;
			for (Shader* shader : shadersToReload) {
				if (!shader || processedShaders.count(shader))
					continue;
				processedShaders.insert(shader);

				spdlog::info("ShaderWatcher: Recompiling shader for file: {}", filepath);
				if (shader->Recompile(device)) {
					recompiledAny = true;
					std::lock_guard<std::mutex> lock(mutex);

					for (const auto& incFile : shader->GetIncludedFiles()) {
						std::string normalized = NormalizePath(incFile);
						auto&       vec = shaderRegistry[normalized];
						if (std::find(vec.begin(), vec.end(), shader) == vec.end()) {
							vec.push_back(shader);
						}
					}

					auto cbIt = shaderCallbacks.find(shader);
					if (cbIt != shaderCallbacks.end()) {
						callbacksToInvoke.insert(callbacksToInvoke.end(), cbIt->second.begin(), cbIt->second.end());
					}
				} else {
					spdlog::error("ShaderWatcher: Failed to recompile shader: {}", filepath);
				}
			}

			for (const auto& callback : callbacksToInvoke) {
				if (callback) {
					callback();
				}
			}
		}
	}

} // namespace brassica
