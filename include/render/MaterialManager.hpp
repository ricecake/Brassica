#pragma once

#include <vector>

#include "render/IMaterialManager.hpp"

namespace brassica {

	/**
	 * @brief Concrete MaterialManager tracking materials CPU-side and updating GPU MaterialBuffer across frames.
	 */
	class MaterialManager: public IMaterialManager {
	public:
		MaterialManager();
		~MaterialManager() override = default;

		void Initialize() override;
		void Shutdown() override;

		State GetState() const override;
		void  SetState(const State& state) override;

		std::uint32_t RegisterMaterial(const MaterialData& material) override;
		bool          UpdateMaterial(std::uint32_t id, const MaterialData& material) override;
		const MaterialData* GetMaterial(std::uint32_t id) const override;
		std::size_t   GetMaterialCount() const override;
		const std::vector<MaterialData>& GetMaterials() const override;
		void          ClearMaterials() override;

		graph::PredefinedBufferNode<MaterialBuffer, MaterialData, SubPhase::Prepare>& GetBufferNode() override;
		void RegisterBufferNode(graph::Graph& frameGraph) override;

	private:
		void SyncBufferNode();

		std::vector<MaterialData> m_materials;
		graph::PredefinedBufferNode<MaterialBuffer, MaterialData, SubPhase::Prepare> m_bufferNode;
	};

} // namespace brassica
