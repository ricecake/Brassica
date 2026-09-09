#pragma once
#include <algorithm>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"

namespace brassica::graph {

	struct ValidationError {
		std::string             message;
		std::vector<ResourceId> missing;
	};

	// Runtime counterpart of IsRenderable/MissingOf (Validation.hpp), for graphs assembled
	// dynamically -- e.g. a subgraph registered at runtime with no compile-time FrameSpec
	// proof available. Same vocabulary, same diagnosability, different enforcement point.
	inline std::expected<void, ValidationError>
	ValidateRuntime(std::span<const NodeDescriptor> nodes, std::span<const ResourceId> imported) {
		std::vector<ResourceId> produced(imported.begin(), imported.end());
		for (const auto& node : nodes) {
			produced.insert(produced.end(), node.produces.begin(), node.produces.end());
		}

		std::vector<ResourceId> missing;
		for (const auto& node : nodes) {
			for (ResourceId key : node.consumes) {
				if (std::find(produced.begin(), produced.end(), key) == produced.end() &&
				    std::find(missing.begin(), missing.end(), key) == missing.end()) {
					missing.push_back(key);
				}
			}
		}

		if (missing.empty()) {
			return {};
		}

		std::string message = "frame is not renderable; no node or import produces these resource keys:";
		for (ResourceId key : missing) {
			message += ' ';
			message += key->name;
		}

		return std::unexpected(ValidationError{std::move(message), std::move(missing)});
	}

	// Runtime container of type-erased nodes. Unlike the declaration layer, Graph has no idea
	// type lists exist -- it only ever sees NodeDescriptor spans, which is what lets a
	// Subgraph (Frame.hpp) hold a Graph internally without leaking template machinery into
	// this class.
	class Graph {
	public:
		template <NodeLike T, typename... Args>
		void Register(Args&&... args) {
			Add(NodeHandle::Make<T>(std::forward<Args>(args)...));
		}

		void Add(NodeHandle handle) { m_nodes.push_back(std::move(handle)); }

		void Setup(const FrameContext& ctx) {
			m_recipes.clear();
			m_recipes.reserve(m_nodes.size());
			for (auto& node : m_nodes) {
				m_recipes.push_back(node.Setup(ctx));
			}
		}

		// Validates producer coverage, then orders nodes by dependency (Kahn's algorithm on
		// producer -> consumer edges) and culls inactive/unreachable ones. This is the
		// single biggest correctness gain over external/FrameGraph, which culls by refcount
		// but executes in registration order rather than dependency order.
		std::expected<void, ValidationError> Compile() {
			std::vector<NodeDescriptor> descriptors;
			descriptors.reserve(m_nodes.size());
			for (const auto& node : m_nodes) {
				descriptors.push_back(node.Descriptor());
			}

			if (auto result = ValidateRuntime(descriptors, {}); !result) {
				return result;
			}

			m_schedule = TopoSort(descriptors);

			std::vector<std::size_t> active;
			active.reserve(m_schedule.size());
			for (std::size_t index : m_schedule) {
				if (index < m_recipes.size() && m_recipes[index].isActive) {
					active.push_back(index);
				}
			}
			m_schedule = std::move(active);

			m_batches.assign(m_schedule.size(), BarrierBatch{});
			return {};
		}

		void Execute(CommandBuffer& cmd) {
			for (std::size_t index : m_schedule) {
				m_nodes[index].Execute(cmd);
			}
		}

		[[nodiscard]] std::span<const std::size_t> Schedule() const { return m_schedule; }

		[[nodiscard]] std::span<const Recipe> Recipes() const { return m_recipes; }

	private:
		// Node i depends on node j (must run after j) if i consumes a key that j produces.
		// Self-edges from Modify<K> (a node both consuming and producing K) are skipped --
		// otherwise Kahn's algorithm would report a false cycle on every in-place modify.
		static std::vector<std::size_t> TopoSort(const std::vector<NodeDescriptor>& nodes) {
			const std::size_t                     n = nodes.size();
			std::vector<std::vector<std::size_t>> dependents(n);
			std::vector<std::size_t>              indegree(n, 0);

			for (std::size_t consumer = 0; consumer < n; ++consumer) {
				for (ResourceId key : nodes[consumer].consumes) {
					for (std::size_t producer = 0; producer < n; ++producer) {
						if (producer == consumer) {
							continue;
						}
						const auto& produces = nodes[producer].produces;
						if (std::find(produces.begin(), produces.end(), key) != produces.end()) {
							dependents[producer].push_back(consumer);
							++indegree[consumer];
						}
					}
				}
			}

			std::vector<std::size_t> ready;
			for (std::size_t i = 0; i < n; ++i) {
				if (indegree[i] == 0) {
					ready.push_back(i);
				}
			}

			std::vector<std::size_t> order;
			order.reserve(n);
			while (!ready.empty()) {
				std::size_t current = ready.back();
				ready.pop_back();
				order.push_back(current);
				for (std::size_t next : dependents[current]) {
					if (--indegree[next] == 0) {
						ready.push_back(next);
					}
				}
			}

			return order;
		}

		std::vector<NodeHandle>   m_nodes;
		std::vector<Recipe>       m_recipes;
		std::vector<std::size_t>  m_schedule;
		std::vector<BarrierBatch> m_batches;
	};

} // namespace brassica::graph
