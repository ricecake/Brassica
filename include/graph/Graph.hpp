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

	// One dependency-respecting level: every node here is provably independent of every other
	// node here (no edge connects any pair within a stage), so a backend is free to dispatch
	// them concurrently -- to different queues, different threads, whatever it has. preBarriers
	// must be flushed before this stage's nodes run; postBarriers after, before the next stage
	// begins (this is where a cross-domain transfer's release half lives, recorded on the
	// producer's side rather than deferred all the way to the consumer's stage).
	struct ScheduleStage {
		std::vector<std::size_t> nodes;
		BarrierBatch             preBarriers;
		BarrierBatch             postBarriers;
	};

	// The whole point of Compile(): a structure a renderer walks without making any further
	// scheduling decisions of its own -- flush a stage's preBarriers, dispatch its nodes to
	// whatever queue matches their declared domain (in any order; that's the parallelism),
	// flush postBarriers, advance.
	struct Schedule {
		std::vector<ScheduleStage> stages;
	};

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

		// Validates producer coverage, levels nodes into dependency-respecting stages (a node's
		// stage is one past the latest stage of anything it consumes; two nodes share a stage
		// iff neither depends on the other -- that structural fact is the parallelism proof),
		// and synthesizes the barriers -- including cross-domain release/acquire pairs -- that
		// sit at each stage boundary. This is the single biggest correctness gain over
		// external/FrameGraph, which culls by refcount but executes in registration order and
		// never batches a barrier at all.
		std::expected<void, ValidationError> Compile() {
			std::vector<NodeDescriptor> descriptors;
			descriptors.reserve(m_nodes.size());
			for (const auto& node : m_nodes) {
				descriptors.push_back(node.Descriptor());
			}

			if (auto result = ValidateRuntime(descriptors, {}); !result) {
				return result;
			}

			const std::vector<Edge> edges = CollectEdges(descriptors);
			const std::size_t       n = descriptors.size();

			std::vector<std::size_t> nodeStage(n, 0);
			m_schedule.stages.clear();
			for (auto& level : LevelNodes(n, edges)) {
				const std::size_t stageIndex = m_schedule.stages.size();
				for (std::size_t index : level) {
					nodeStage[index] = stageIndex;
				}
				m_schedule.stages.push_back(ScheduleStage{std::move(level), {}, {}});
			}

			for (const Edge& edge : edges) {
				SynthesizeBarrier(descriptors, nodeStage, edge);
			}

			for (auto& stage : m_schedule.stages) {
				std::erase_if(stage.nodes, [this](std::size_t index) { return !m_recipes[index].isActive; });
			}

			return {};
		}

		void Execute(CommandBuffer& cmd) {
			for (const auto& stage : m_schedule.stages) {
				for (std::size_t index : stage.nodes) {
					m_nodes[index].Execute(cmd);
				}
			}
		}

		// Executes a single scheduled node by index, bracketed by nothing itself -- callers
		// that need to interleave per-node work (barrier flushes, dynamic-rendering begin/end)
		// use this instead of the batch Execute() above.
		void ExecuteNode(std::size_t index, CommandBuffer& cmd) {
			if (index < m_nodes.size()) {
				m_nodes[index].Execute(cmd);
			}
		}

		[[nodiscard]] const Schedule&             GetSchedule() const { return m_schedule; }
		[[nodiscard]] std::span<const Recipe>     Recipes() const { return m_recipes; }
		[[nodiscard]] std::span<const NodeHandle> Nodes() const { return m_nodes; }

	private:
		struct Edge {
			std::size_t producer;
			std::size_t consumer;
			ResourceId  key;
		};

		// One edge per (producer, consumer, key) triple where consumer.consumes contains key
		// and producer.produces contains it too. Self-edges (Modify<K>, a node both consuming
		// and producing K) are skipped -- otherwise leveling would treat a node as depending on
		// itself.
		static std::vector<Edge> CollectEdges(const std::vector<NodeDescriptor>& nodes) {
			std::vector<Edge> edges;
			for (std::size_t consumer = 0; consumer < nodes.size(); ++consumer) {
				for (ResourceId key : nodes[consumer].consumes) {
					for (std::size_t producer = 0; producer < nodes.size(); ++producer) {
						if (producer == consumer) {
							continue;
						}
						const auto& produces = nodes[producer].produces;
						if (std::find(produces.begin(), produces.end(), key) != produces.end()) {
							edges.push_back(Edge{producer, consumer, key});
						}
					}
				}
			}
			return edges;
		}

		// BFS layering (longest-path-from-source levels): repeatedly peel off every node whose
		// dependencies are already fully satisfied by prior levels. A node only leaves the
		// frontier once every one of its producers has been placed, so level(v) == 1 +
		// max(level(u)) over all direct producers u -- and no two nodes in the same level can
		// have an edge between them, which is exactly the "provably independent" property
		// Schedule promises.
		static std::vector<std::vector<std::size_t>> LevelNodes(std::size_t n, const std::vector<Edge>& edges) {
			std::vector<std::vector<std::size_t>> dependents(n);
			std::vector<std::size_t>              indegree(n, 0);

			for (std::size_t consumer = 0; consumer < n; ++consumer) {
				std::vector<std::size_t> producers;
				for (const Edge& edge : edges) {
					if (edge.consumer != consumer) {
						continue;
					}
					if (std::find(producers.begin(), producers.end(), edge.producer) != producers.end()) {
						continue;
					}
					producers.push_back(edge.producer);
					dependents[edge.producer].push_back(consumer);
				}
				indegree[consumer] = producers.size();
			}

			std::vector<std::size_t> frontier;
			for (std::size_t i = 0; i < n; ++i) {
				if (indegree[i] == 0) {
					frontier.push_back(i);
				}
			}

			std::vector<std::vector<std::size_t>> levels;
			while (!frontier.empty()) {
				std::sort(frontier.begin(), frontier.end());
				std::vector<std::size_t> next;
				for (std::size_t node : frontier) {
					for (std::size_t dependent : dependents[node]) {
						if (--indegree[dependent] == 0) {
							next.push_back(dependent);
						}
					}
				}
				levels.push_back(frontier);
				frontier = std::move(next);
			}

			return levels;
		}

		// A node's access to a key it touches: ReadWrite if the key appears in both its
		// consumes and produces (Modify<K>), Write if produces-only, Read otherwise.
		static AccessKind AccessOf(const NodeDescriptor& node, ResourceId key) {
			const bool reads = std::find(node.consumes.begin(), node.consumes.end(), key) != node.consumes.end();
			const bool writes = std::find(node.produces.begin(), node.produces.end(), key) != node.produces.end();
			if (reads && writes) {
				return AccessKind::ReadWrite;
			}
			return writes ? AccessKind::Write : AccessKind::Read;
		}

		// Every edge CollectEdges finds has, by construction, a producer that writes the key
		// (that's what made it a producer) -- so producerAccess is never pure Read, and there
		// is no read-after-read case to special-case away here. Two readers of an
		// already-settled resource simply never edge each other: each gets its own
		// producer->reader edge instead, and BarrierBatch::Add already merges those into one
		// entry when they land in the same stage.
		void SynthesizeBarrier(
			const std::vector<NodeDescriptor>& descriptors,
			const std::vector<std::size_t>&    nodeStage,
			const Edge&                        edge
		) {
			const AccessKind producerAccess = AccessOf(descriptors[edge.producer], edge.key);
			const AccessKind consumerAccess = AccessOf(descriptors[edge.consumer], edge.key);

			const ExecutionDomain producerDomain = m_recipes[edge.producer].domain;
			const ExecutionDomain consumerDomain = m_recipes[edge.consumer].domain;
			const std::size_t     producerStage = nodeStage[edge.producer];
			const std::size_t     consumerStage = nodeStage[edge.consumer];

			if (producerDomain == consumerDomain) {
				m_schedule.stages[consumerStage].preBarriers.Add(
					MemoryBarrier{
						.resource = edge.key,
						.access = consumerAccess,
						.srcDomain = producerDomain,
						.dstDomain = consumerDomain,
					}
				);
				return;
			}

			m_schedule.stages[producerStage].postBarriers.Add(
				MemoryBarrier{
					.resource = edge.key,
					.access = producerAccess,
					.srcDomain = producerDomain,
					.dstDomain = consumerDomain,
				}
			);
			m_schedule.stages[consumerStage].preBarriers.Add(
				MemoryBarrier{
					.resource = edge.key,
					.access = consumerAccess,
					.srcDomain = producerDomain,
					.dstDomain = consumerDomain,
				}
			);
		}

		std::vector<NodeHandle> m_nodes;
		std::vector<Recipe>     m_recipes;
		Schedule                m_schedule;
	};

} // namespace brassica::graph
