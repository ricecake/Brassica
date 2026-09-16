#include "ConfigManager.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>

#include "spdlog/spdlog.h"

namespace brassica {

	static std::string Trim(const std::string& str) {
		size_t first = str.find_first_not_of(" \t\r\n");
		if (first == std::string::npos)
			return "";
		size_t last = str.find_last_not_of(" \t\r\n");
		return str.substr(first, (last - first + 1));
	}

	ConfigManager::ConfigManager(): m_appName("Sandbox") {}

	ConfigManager::ConfigManager(std::string appName): m_appName(std::move(appName)) {}

	void ConfigManager::Initialize() {
		if (m_initialized)
			return;

		m_initialized = true;
	}

	void ConfigManager::Shutdown() {
		m_initialized = false;
	}

	bool ConfigManager::HasValue(const std::string& section, const std::string& key) const {
		auto secIt = m_sections.find(section);
		if (secIt == m_sections.end()) {
			return false;
		}
		return secIt->second.count(key) > 0;
	}

	std::string ConfigManager::GetRawString(
		const std::string& section,
		const std::string& key,
		const std::string& defaultVal
	) const {
		auto secIt = m_sections.find(section);
		if (secIt == m_sections.end()) {
			return defaultVal;
		}
		auto keyIt = secIt->second.find(key);
		if (keyIt == secIt->second.end()) {
			return defaultVal;
		}
		return keyIt->second;
	}

	void ConfigManager::SetRawString(const std::string& section, const std::string& key, const std::string& val) {
		m_sections[section][key] = val;
	}

	bool ConfigManager::LoadFromFile(const std::string& filepath) {
		m_filepath = filepath;
		std::ifstream file(filepath);
		if (!file.is_open()) {
			spdlog::info("ConfigManager: Config file '{}' not found, using defaults.", filepath);
			return false;
		}

		m_sections.clear();
		std::string currentSection = "General";
		std::string line;

		while (std::getline(file, line)) {
			line = Trim(line);
			if (line.empty() || line[0] == ';' || line[0] == '#') {
				continue;
			}

			if (line.front() == '[' && line.back() == ']') {
				currentSection = Trim(line.substr(1, line.size() - 2));
				continue;
			}

			auto eqPos = line.find('=');
			if (eqPos != std::string::npos) {
				std::string key = Trim(line.substr(0, eqPos));
				std::string val = Trim(line.substr(eqPos + 1));
				if (!key.empty()) {
					m_sections[currentSection][key] = val;
				}
			}
		}

		spdlog::info("ConfigManager: Loaded config from '{}'.", filepath);
		return true;
	}

	bool ConfigManager::SaveToFile(const std::string& filepath) const {
		std::ofstream file(filepath);
		if (!file.is_open()) {
			spdlog::error("ConfigManager: Failed to open '{}' for writing.", filepath);
			return false;
		}

		for (const auto& [section, keys] : m_sections) {
			file << "[" << section << "]\n";
			for (const auto& [key, val] : keys) {
				file << key << " = " << val << "\n";
			}
			file << "\n";
		}

		spdlog::info("ConfigManager: Saved config to '{}'.", filepath);
		return true;
	}

} // namespace brassica
