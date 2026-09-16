#pragma once

namespace brassica {

	/**
	 * @brief Base interface for all manager classes in the engine with RAII semantics.
	 *
	 * Managers own system lifecycles and persistent resources.
	 * Non-copyable to prevent resource duplication.
	 */
	class IManager {
	public:
		IManager() = default;

		virtual ~IManager() {
			if (m_initialized) {
				Shutdown();
			}
		}

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
		virtual void Shutdown() {
			m_initialized = false;
		}

		/**
		 * @brief Query if the manager is currently initialized.
		 */
		[[nodiscard]] bool IsInitialized() const { return m_initialized; }

	protected:
		bool m_initialized{false};
	};

} // namespace brassica
