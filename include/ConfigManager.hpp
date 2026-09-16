#pragma once

#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <glm/glm.hpp>

#include "IManager.hpp"

namespace brassica {

	class ConfigManager: public IManager {
	public:
		ConfigManager();
		explicit ConfigManager(std::string appName);
		~ConfigManager() override = default;

		void Initialize() override;
		void Shutdown() override;

		void SetApplicationName(const std::string& appName) { m_appName = appName; }

		[[nodiscard]] const std::string& GetApplicationName() const { return m_appName; }

		bool LoadFromFile(const std::string& filepath);
		bool SaveToFile(const std::string& filepath) const;

		// Application-level configuration
		template <typename T>
		T GetAppSetting(const std::string& key, const T& defaultValue) const {
			std::string appSection = "Application." + m_appName;
			if (HasValue(appSection, key)) {
				return GetValue<T>(appSection, key, defaultValue);
			}
			if (HasValue("Application", key)) {
				return GetValue<T>("Application", key, defaultValue);
			}
			return defaultValue;
		}

		template <typename T>
		void SetAppSetting(const std::string& key, const T& value) {
			std::string appSection = "Application." + m_appName;
			SetValue<T>(appSection, key, value);
		}

		// Manager-level configuration
		template <typename T>
		T GetManagerSetting(const std::string& managerName, const std::string& key, const T& defaultValue) const {
			std::string scopedSection = "Application." + m_appName + "." + managerName;
			if (HasValue(scopedSection, key)) {
				return GetValue<T>(scopedSection, key, defaultValue);
			}
			std::string mgrSection = "Manager." + managerName;
			if (HasValue(mgrSection, key)) {
				return GetValue<T>(mgrSection, key, defaultValue);
			}
			if (HasValue(managerName, key)) {
				return GetValue<T>(managerName, key, defaultValue);
			}
			return defaultValue;
		}

		template <typename T>
		void SetManagerSetting(const std::string& managerName, const std::string& key, const T& value) {
			std::string scopedSection = "Application." + m_appName + "." + managerName;
			SetValue<T>(scopedSection, key, value);
		}

		[[nodiscard]] bool HasValue(const std::string& section, const std::string& key) const;

		std::string GetRawString(const std::string& section, const std::string& key, const std::string& defaultVal) const;
		void        SetRawString(const std::string& section, const std::string& key, const std::string& val);

	private:
		template <typename T>
		T GetValue(const std::string& section, const std::string& key, const T& defaultValue) const {
			std::string raw = GetRawString(section, key, "");
			if (raw.empty()) {
				return defaultValue;
			}
			if constexpr (std::is_same_v<T, std::string>) {
				return raw;
			} else if constexpr (std::is_same_v<T, bool>) {
				return raw == "true" || raw == "1" || raw == "yes" || raw == "ON";
			} else if constexpr (std::is_integral_v<T>) {
				try {
					return static_cast<T>(std::stoll(raw));
				} catch (...) {
					return defaultValue;
				}
			} else if constexpr (std::is_floating_point_v<T>) {
				try {
					return static_cast<T>(std::stod(raw));
				} catch (...) {
					return defaultValue;
				}
			} else if constexpr (std::is_same_v<T, glm::vec3>) {
				std::stringstream ss(raw);
				glm::vec3         v(0.0f);
				if (ss >> v.x >> v.y >> v.z) {
					return v;
				}
				return defaultValue;
			} else if constexpr (std::is_same_v<T, glm::vec4>) {
				std::stringstream ss(raw);
				glm::vec4         v(0.0f);
				if (ss >> v.x >> v.y >> v.z >> v.w) {
					return v;
				}
				return defaultValue;
			} else {
				return defaultValue;
			}
		}

		template <typename T>
		void SetValue(const std::string& section, const std::string& key, const T& value) {
			if constexpr (std::is_same_v<T, std::string>) {
				SetRawString(section, key, value);
			} else if constexpr (std::is_same_v<T, bool>) {
				SetRawString(section, key, value ? "true" : "false");
			} else if constexpr (std::is_same_v<T, glm::vec3>) {
				std::stringstream ss;
				ss << value.x << " " << value.y << " " << value.z;
				SetRawString(section, key, ss.str());
			} else if constexpr (std::is_same_v<T, glm::vec4>) {
				std::stringstream ss;
				ss << value.x << " " << value.y << " " << value.z << " " << value.w;
				SetRawString(section, key, ss.str());
			} else {
				SetRawString(section, key, std::to_string(value));
			}
		}

		std::string                                            m_appName{"Sandbox"};
		std::string                                            m_filepath{"config.ini"};
		std::map<std::string, std::map<std::string, std::string>> m_sections;
	};

} // namespace brassica
