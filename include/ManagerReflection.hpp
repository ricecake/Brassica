#pragma once

#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <glm/glm.hpp>

#include "imgui.h"

namespace brassica {

	enum class UIHint { Default, Slider, Drag, Color, Checkbox, InputText };

	template <typename Class, typename T>
	struct FieldReflect {
		std::string name;  // Config section key
		std::string label; // Friendly display label
		T Class::* member; // Member pointer
		T          minVal{};
		T          maxVal{};
		UIHint     hint{UIHint::Default};
	};

	template <typename Class, typename T>
	auto MakeField(
		std::string name,
		std::string label,
		T Class::* member,
		T          minVal = T{},
		T          maxVal = T{},
		UIHint     hint = UIHint::Default
	) {
		return FieldReflect<Class, T>{std::move(name), std::move(label), member, minVal, maxVal, hint};
	}

	template <typename Class, typename T>
	auto MakeColorField(std::string name, std::string label, T Class::* member) {
		return FieldReflect<Class, T>{std::move(name), std::move(label), member, T{}, T{}, UIHint::Color};
	}

	template <typename ConfigType, typename StateStruct, typename Class, typename T>
	void SaveReflectedField(
		ConfigType&                   config,
		const std::string&            section,
		const StateStruct&            state,
		const FieldReflect<Class, T>& field
	) {
		config.SetManagerSetting(section, field.name, state.*field.member);
	}

	template <typename ConfigType, typename StateStruct, typename Class, typename T>
	bool LoadReflectedField(
		const ConfigType&             config,
		const std::string&            section,
		StateStruct&                  state,
		const FieldReflect<Class, T>& field
	) {
		state.*field.member = config.template GetManagerSetting<T>(section, field.name, state.*field.member);
		return true;
	}

	template <typename StateStruct, typename Class, typename T>
	bool DrawReflectedFieldUI(StateStruct& state, const FieldReflect<Class, T>& field) {
		T&          value = state.*field.member;
		std::string idLabel = field.label + "##" + field.name;
		bool        modified = false;

		if constexpr (std::is_same_v<T, bool>) {
			modified = ImGui::Checkbox(idLabel.c_str(), &value);
		} else if constexpr (std::is_same_v<T, float>) {
			if (field.hint == UIHint::Slider) {
				modified = ImGui::SliderFloat(idLabel.c_str(), &value, field.minVal, field.maxVal);
			} else {
				modified = ImGui::DragFloat(idLabel.c_str(), &value, 0.1f, field.minVal, field.maxVal);
			}
		} else if constexpr (std::is_integral_v<T>) {
			int tempVal = static_cast<int>(value);
			if (field.hint == UIHint::Slider) {
				modified = ImGui::SliderInt(
					idLabel.c_str(),
					&tempVal,
					static_cast<int>(field.minVal),
					static_cast<int>(field.maxVal)
				);
			} else {
				modified = ImGui::DragInt(
					idLabel.c_str(),
					&tempVal,
					1.0f,
					static_cast<int>(field.minVal),
					static_cast<int>(field.maxVal)
				);
			}
			if (modified) {
				value = static_cast<T>(tempVal);
			}
		} else if constexpr (std::is_same_v<T, glm::vec3>) {
			if (field.hint == UIHint::Color) {
				modified = ImGui::ColorEdit3(idLabel.c_str(), &value.x);
			} else {
				modified = ImGui::DragFloat3(idLabel.c_str(), &value.x, 0.1f);
			}
		} else if constexpr (std::is_same_v<T, glm::vec4>) {
			if (field.hint == UIHint::Color) {
				modified = ImGui::ColorEdit4(idLabel.c_str(), &value.x);
			} else {
				modified = ImGui::DragFloat4(idLabel.c_str(), &value.x, 0.1f);
			}
		} else if constexpr (std::is_same_v<T, std::string>) {
			char buffer[256];
			std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
			buffer[sizeof(buffer) - 1] = '\0';
			if (ImGui::InputText(idLabel.c_str(), buffer, sizeof(buffer))) {
				value = buffer;
				modified = true;
			}
		}

		return modified;
	}

	template <typename ConfigType, typename StateStruct>
	void SaveStateToConfig(ConfigType& config, const std::string& section, const StateStruct& state) {
		auto fields = state.GetReflection();
		std::apply(
			[&config, &section, &state](const auto&... field) {
				(SaveReflectedField(config, section, state, field), ...);
			},
			fields
		);
	}

	template <typename ConfigType, typename StateStruct>
	bool LoadStateToConfig(const ConfigType& config, const std::string& section, StateStruct& state) {
		auto fields = state.GetReflection();
		bool loadedAny = false;
		std::apply(
			[&config, &section, &state, &loadedAny](const auto&... field) {
				((loadedAny |= LoadReflectedField(config, section, state, field)), ...);
			},
			fields
		);
		return loadedAny;
	}

	template <typename StateStruct>
	bool DrawReflectedStateUI(const std::string& title, StateStruct& state) {
		(void)title;
		auto fields = state.GetReflection();
		bool modified = false;
		std::apply(
			[&state, &modified](const auto&... field) { ((modified |= DrawReflectedFieldUI(state, field)), ...); },
			fields
		);
		return modified;
	}

} // namespace brassica
