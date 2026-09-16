#pragma once

namespace brassica {

	/**
	 * @brief Base interface for all manager classes in the engine.
	 *
	 * Managers are responsible for handling specific resources or systems
	 * and typically require an explicit initialization step.
	 */
	class IManager {
	public:
		virtual ~IManager() = default;

		/**
		 * @brief Initialize the manager. Called once after construction.
		 */
		virtual void Initialize() = 0;

		/**
		 * @brief Clean up resources. Called before destruction.
		 * Default no-op — override if the manager needs explicit teardown.
		 */
		virtual void Shutdown() {}
	};

} // namespace brassica
