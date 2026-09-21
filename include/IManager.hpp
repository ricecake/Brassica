#pragma once

#include <string>

#include "ManagerReflection.hpp"

namespace brassica {

	class ConfigManager;

	/**
	 * @brief Base interface for all manager classes in the engine with RAII semantics.
	 *
	 * Managers own system lifecycles and persistent resources.
	 * Non-copyable to prevent resource duplication.
	 */
	class IManager {
	public:
		IManager() = default;
		virtual ~IManager() = default;

		IManager(const IManager&) = delete;
		IManager& operator=(const IManager&) = delete;

		IManager(IManager&&) noexcept = default;
		IManager& operator=(IManager&&) noexcept = default;

		/**
		 * @brief Initialize the manager. Called once after construction.
		 */
		virtual void Initialize() = 0;

		/**
		 * @brief Clean up resources. Called automatically on destruction or manually.
		 */
		virtual void Shutdown() { m_initialized = false; }

		/**
		 * @brief Query if the manager is currently initialized.
		 */
		[[nodiscard]] bool IsInitialized() const { return m_initialized; }

		/**
		 * @brief Get the unique name of the manager for config and UI display.
		 */
		[[nodiscard]] virtual std::string GetManagerName() const = 0;

		/**
		 * @brief Save the active state of the manager to configuration.
		 */
		virtual void SaveState(ConfigManager& config) const = 0;

		/**
		 * @brief Load the active state of the manager from configuration.
		 */
		virtual void LoadState(const ConfigManager& config) = 0;

		/**
		 * @brief Render UI controls for the manager active state.
		 * @return True if state was modified in UI.
		 */
		virtual bool DrawUI() = 0;

	protected:
		bool m_initialized{false};
	};

	/**
	 * @brief CRTP/Base class template enforcing serializable state methods on managers.
	 */
	template <typename Derived, typename StateType>
	class ManagerBase: public IManager {
	public:
		using State = StateType;

		/**
		 * @brief Return a serializable struct representing the active state of the manager.
		 */
		[[nodiscard]] virtual State GetState() const = 0;

		/**
		 * @brief Accept a serializable struct to update the active state of the manager.
		 */
		virtual void SetState(const State& state) = 0;

		void SaveState(ConfigManager& config) const override;
		void LoadState(const ConfigManager& config) override;
		bool DrawUI() override;
	};

} // namespace brassica

#include "ConfigManager.hpp"
