#include "render/MaterialManager.hpp"

#include "graph/Graph.hpp"

namespace brassica {

	MaterialManager::MaterialManager() {
		// Seed with default material at index 0
		m_materials.push_back(MaterialData{});
		SyncBufferNode();
	}

	void MaterialManager::Initialize() {
		m_initialized = true;
		SyncBufferNode();
	}

	void MaterialManager::Shutdown() {
		m_initialized = false;
	}

	MaterialManagerState MaterialManager::GetState() const {
		MaterialManagerState state;
		state.materialCount = static_cast<std::uint32_t>(m_materials.size());
		return state;
	}

	void MaterialManager::SetState(const MaterialManagerState& state) {
		(void)state;
	}

	std::uint32_t MaterialManager::RegisterMaterial(const MaterialData& material) {
		m_materials.push_back(material);
		const auto id = static_cast<std::uint32_t>(m_materials.size() - 1);
		SyncBufferNode();
		return id;
	}

	bool MaterialManager::UpdateMaterial(std::uint32_t id, const MaterialData& material) {
		if (id >= m_materials.size()) {
			return false;
		}
		m_materials[id] = material;
		SyncBufferNode();
		return true;
	}

	const MaterialData* MaterialManager::GetMaterial(std::uint32_t id) const {
		if (id >= m_materials.size()) {
			return nullptr;
		}
		return &m_materials[id];
	}

	std::size_t MaterialManager::GetMaterialCount() const {
		return m_materials.size();
	}

	const std::vector<MaterialData>& MaterialManager::GetMaterials() const {
		return m_materials;
	}

	void MaterialManager::ClearMaterials() {
		m_materials.clear();
		m_materials.push_back(MaterialData{}); // Always retain default material at index 0
		SyncBufferNode();
	}

	graph::PredefinedBufferNode<MaterialBuffer, MaterialData, SubPhase::Prepare>& MaterialManager::GetBufferNode() {
		return m_bufferNode;
	}

	void MaterialManager::RegisterBufferNode(graph::Graph& frameGraph) {
		SyncBufferNode();
		frameGraph.RegisterRef(m_bufferNode);
	}

	void MaterialManager::SyncBufferNode() {
		m_bufferNode.SetData(m_materials);
	}

} // namespace brassica
