#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "IManager.hpp"
#include "graph/Node.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "types/MaterialData.hpp"

namespace brassica {

	/**
	 * @brief Reflection state for MaterialManager.
	 */
	struct MaterialManagerState {
		std::uint32_t materialCount{0};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("materialCount", "Material Count", &MaterialManagerState::materialCount, 0u, 10000u, UIHint::Drag)
			);
		}
	};

	/**
	 * @brief Abstract interface for tracking and managing materials across frames.
	 */
	class IMaterialManager: public ManagerBase<IMaterialManager, MaterialManagerState> {
	public:
		using State = MaterialManagerState;

		~IMaterialManager() override = default;

		[[nodiscard]] std::string GetManagerName() const override { return "MaterialManager"; }

		/**
		 * @brief Register a new material and return its material ID.
		 */
		virtual std::uint32_t RegisterMaterial(const MaterialData& material) = 0;

		/**
		 * @brief Update an existing registered material by ID.
		 * @return True if update succeeded, false if ID was out of bounds.
		 */
		virtual bool UpdateMaterial(std::uint32_t id, const MaterialData& material) = 0;

		/**
		 * @brief Retrieve a pointer to material data by ID, or nullptr if invalid.
		 */
		virtual const MaterialData* GetMaterial(std::uint32_t id) const = 0;

		/**
		 * @brief Get the current number of registered materials.
		 */
		virtual std::size_t GetMaterialCount() const = 0;

		/**
		 * @brief Retrieve all registered materials.
		 */
		virtual const std::vector<MaterialData>& GetMaterials() const = 0;

		/**
		 * @brief Reset materials list to default state.
		 */
		virtual void ClearMaterials() = 0;

		/**
		 * @brief Access the render graph buffer node managing GPU MaterialBuffer updates.
		 */
		virtual graph::PredefinedBufferNode<MaterialBuffer, MaterialData, SubPhase::Prepare>& GetBufferNode() = 0;

		/**
		 * @brief Register the material buffer node into the given frame graph.
		 */
		virtual void RegisterBufferNode(graph::Graph& frameGraph) = 0;
	};

} // namespace brassica
