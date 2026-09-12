#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "vulkan/vulkan.hpp"

#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/TypeList.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/GradientNode.hpp"
#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica::graph {

	struct NodeInitParams {
		vk::Device                    device;
		render::PipelineLibrary*      pipelineLibrary = nullptr;
		ShaderWatcher*                shaderWatcher = nullptr;
		TerrainAccelerationStructure* terrainAS = nullptr;
		vk::Format                    swapchainFormat = vk::Format::eUndefined;
	};

	struct NodeFrameParams {
		TerrainPushConstants    terrainPush{};
		DeferredPushConstants   deferredPush{};
		WaterPushConstants      waterPush{};
		AtmospherePushConstants atmospherePush{};
	};

	template <typename T>
	void InitNode(T& node, const NodeInitParams& params) {
		if constexpr (requires {
						  node.Init(params.device, params.pipelineLibrary, params.terrainAS, params.shaderWatcher);
					  }) {
			node.Init(params.device, params.pipelineLibrary, params.terrainAS, params.shaderWatcher);
		} else if constexpr (requires {
								 node.Init(
									 params.device,
									 params.pipelineLibrary,
									 params.terrainAS ? &params.terrainAS->GetDls() : nullptr,
									 params.swapchainFormat,
									 params.shaderWatcher
								 );
							 }) {
			node.Init(
				params.device,
				params.pipelineLibrary,
				params.terrainAS ? &params.terrainAS->GetDls() : nullptr,
				params.swapchainFormat,
				params.shaderWatcher
			);
		} else if constexpr (requires {
								 node.Init(
									 params.device,
									 params.pipelineLibrary,
									 params.swapchainFormat,
									 params.shaderWatcher
								 );
							 }) {
			node.Init(params.device, params.pipelineLibrary, params.swapchainFormat, params.shaderWatcher);
		} else if constexpr (requires {
								 node.Init(params.device, params.pipelineLibrary, params.shaderWatcher);
							 }) {
			node.Init(params.device, params.pipelineLibrary, params.shaderWatcher);
		} else if constexpr (requires { node.Init(params); }) {
			node.Init(params);
		}
	}

	template <typename T>
	void DestroyNode(T& node, vk::Device device) {
		if constexpr (requires { node.Destroy(device); }) {
			node.Destroy(device);
		}
	}

	template <typename T>
	void UpdateNodeFrameParams(T& node, const NodeFrameParams& params) {
		if constexpr (requires { node.SetFrameParams(params.terrainPush); }) {
			node.SetFrameParams(params.terrainPush);
		}
		if constexpr (requires { node.SetFrameParams(params.deferredPush); }) {
			node.SetFrameParams(params.deferredPush);
		}
		if constexpr (requires { node.SetFrameParams(params.waterPush); }) {
			node.SetFrameParams(params.waterPush);
		}
		if constexpr (requires { node.atmosphere = params.atmospherePush; }) {
			node.atmosphere = params.atmospherePush;
		}
		if constexpr (requires { node.SetFrameParams(params); }) {
			node.SetFrameParams(params);
		}
	}

	class NodeFactory {
	public:
		template <NodeLike T>
		static T CreateConfiguredNode(const NodeInitParams& params) {
			T node{};
			InitNode(node, params);
			return node;
		}
	};

	template <NodeLike... Nodes>
	class NodeRegistry {
	public:
		using NodeTypes = TypeList<Nodes...>;

		std::tuple<Nodes...> nodes;

		void Init(const NodeInitParams& params) {
			std::apply([&](auto&... n) { (InitNode(n, params), ...); }, nodes);
		}

		void Destroy(vk::Device device) {
			std::apply([&](auto&... n) { (DestroyNode(n, device), ...); }, nodes);
		}

		void UpdateFrameParams(const NodeFrameParams& params) {
			std::apply([&](auto&... n) { (UpdateNodeFrameParams(n, params), ...); }, nodes);
		}

		template <typename T>
		[[nodiscard]] T& GetNode() {
			return std::get<T>(nodes);
		}

		template <typename T>
		[[nodiscard]] const T& GetNode() const {
			return std::get<T>(nodes);
		}

		template <std::size_t I>
		[[nodiscard]] auto& GetNode() {
			return std::get<I>(nodes);
		}

		template <std::size_t I>
		[[nodiscard]] const auto& GetNode() const {
			return std::get<I>(nodes);
		}

		template <typename... SelectedNodes>
		void PopulateGraph(Graph& outGraph) {
			if constexpr (sizeof...(SelectedNodes) == 0) {
				std::apply([&](auto&... n) { (outGraph.RegisterRef(n), ...); }, nodes);
			} else {
				(outGraph.RegisterRef(std::get<SelectedNodes>(nodes)), ...);
			}
		}

		template <typename... SelectedNodes>
		[[nodiscard]] Graph CreateGraph() {
			Graph g;
			PopulateGraph<SelectedNodes...>(g);
			return g;
		}
	};

	template <NodeLike... Nodes>
	using GraphRegistry = NodeRegistry<Nodes...>;

	using DefaultNodeRegistry = NodeRegistry<
		GradientNode,
		TerrainNode,
		DeferredNode,
		WaterNode,
		TransmittanceLUTNode,
		MultiScatteringLUTNode>;

} // namespace brassica::graph
