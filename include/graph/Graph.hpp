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

		// Non-null only when node `index` is a Subgraph -- lets a backend recurse into its
		// inner graph with the same treatment (Provision/barrier-synthesis/DynamicRendering) it
		// gives this graph, instead of falling through to ExecuteNode/Subgraph::Execute's naive
		// per-node loop. See PhysicalExecutionBackend::RunSchedule.
		[[nodiscard]] Graph* InnerGraphOfNode(std::size_t index) {
			return index < m_nodes.size() ? m_nodes[index].InnerGraphIfAny() : nullptr;
		}

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
		//
		// externallyProvided is for a Subgraph's own inner graph (Frame.hpp): its Resources are
		// its *net* interface (Spec::Unsatisfied), so a key in that set has, by construction, no
		// producer inside this graph at all -- the parent supplies it. Without naming that here,
		// ValidateRuntime would report it missing every time, which is exactly backwards: an
		// unsatisfied input is a Subgraph's interface, not a bug. A root Graph/Frame has none of
		// these, hence the empty default -- Frame already asserts full renderability itself
		// (Validation.hpp's AssertRenderable), so nothing is lost by requiring every input to be
		// a real producer or Import there.
		std::expected<void, ValidationError> Compile(std::span<const ResourceId> externallyProvided = {}) {
			std::vector<NodeDescriptor> descriptors;
			descriptors.reserve(m_nodes.size());
			for (const auto& node : m_nodes) {
				descriptors.push_back(node.Descriptor());
			}

			if (auto result = ValidateRuntime(descriptors, externallyProvided); !result) {
				return result;
			}

			const std::size_t n = descriptors.size();

			// Version lifting: a consumer that declares key K (base B, version V) binds to the
			// highest version of B any strictly-lower-phase node produces, if one exists -- so a
			// downstream reader of "whatever the latest state is" (plain Read<Swapchain>, version
			// 0) doesn't need to know an earlier phase used Modify<Swapchain, 2> internally.
			// Materialized per-node rather than mutated in place: NodeDescriptor::consumes is a
			// span into per-type static storage (ResourceKey.hpp's kIdArray) that every other
			// consumer of the type relies on staying untouched.
			std::vector<std::vector<ResourceId>> liftedConsumes(n);
			for (std::size_t consumer = 0; consumer < n; ++consumer) {
				liftedConsumes[consumer].assign(
					descriptors[consumer].consumes.begin(),
					descriptors[consumer].consumes.end()
				);
				for (ResourceId& key : liftedConsumes[consumer]) {
					const ResourceId base = key->versionBase ? key->versionBase : key;
					std::uint32_t    bestVersion = key->version;
					ResourceId       bestKey = key;
					for (std::size_t producer = 0; producer < n; ++producer) {
						if (producer == consumer || descriptors[producer].phase >= descriptors[consumer].phase) {
							continue;
						}
						for (ResourceId produced : descriptors[producer].produces) {
							const ResourceId producedBase = produced->versionBase ? produced->versionBase : produced;
							if (producedBase == base && produced->version > bestVersion) {
								bestVersion = produced->version;
								bestKey = produced;
							}
						}
					}
					key = bestKey;
				}
			}

			std::vector<NodeDescriptor> effective = descriptors;
			for (std::size_t i = 0; i < n; ++i) {
				effective[i].consumes = liftedConsumes[i];
			}

			const std::vector<Edge> allEdges = CollectEdges(effective);

			// Classify by phase relationship, using the *original* declared phases (lifting never
			// changes a node's own phase). A same-phase edge levels within its phase partition
			// below. A cross-phase edge is either already satisfied by construction (forward --
			// keep, so SynthesizeBarrier still emits the real memory dependency), the expected
			// shape of two same-key writers a phase apart (backward, consumer also produces the
			// key -- drop silently, ordering comes from phase, not resource flow: this is exactly
			// what lets DeferredNode and a later WaterNode both plain Modify<Swapchain> without a
			// version number between them), or a genuine phase violation (backward, consumer is a
			// pure reader -- error).
			std::vector<Edge> intraPhaseEdges;
			std::vector<Edge> crossPhaseEdges;
			for (const Edge& edge : allEdges) {
				const Phase producerPhase = descriptors[edge.producer].phase;
				const Phase consumerPhase = descriptors[edge.consumer].phase;
				if (producerPhase == consumerPhase) {
					intraPhaseEdges.push_back(edge);
					continue;
				}
				if (producerPhase < consumerPhase) {
					crossPhaseEdges.push_back(edge);
					continue;
				}

				const auto& consumerProduces = descriptors[edge.consumer].produces;
				const bool  consumerAlsoProduces = std::find(
													   consumerProduces.begin(),
													   consumerProduces.end(),
													   edge.key
												   ) != consumerProduces.end();
				if (consumerAlsoProduces) {
					continue;
				}

				return std::unexpected(
					ValidationError{
						.message = "phase violation: '" + std::string(descriptors[edge.consumer].name) + "' (phase " +
							std::to_string(static_cast<std::int32_t>(consumerPhase)) + ") requires '" +
							std::string(edge.key->name) + "', but its only producer, '" +
							std::string(descriptors[edge.producer].name) + "', runs in a later phase (" +
							std::to_string(static_cast<std::int32_t>(producerPhase)) +
							") -- a later phase cannot satisfy an earlier phase's input",
						.missing = {edge.key},
					}
				);
			}

			// Partition into phase groups in ascending phase order and level within each group
			// using only that group's own edges. A node in a later-phase group therefore lands
			// after every earlier-phase node's stage even with zero resource dependency between
			// them (Phase's whole purpose), and grouping this way means a phase can never
			// manufacture a false cross-phase cycle -- only genuinely same-phase edges ever feed
			// one LevelNodes call together.
			std::vector<Phase> phaseOrder;
			for (const auto& d : descriptors) {
				if (std::find(phaseOrder.begin(), phaseOrder.end(), d.phase) == phaseOrder.end()) {
					phaseOrder.push_back(d.phase);
				}
			}
			std::sort(phaseOrder.begin(), phaseOrder.end());

			std::vector<std::size_t> nodeStage(n, 0);
			m_schedule.stages.clear();
			for (Phase phase : phaseOrder) {
				std::vector<std::size_t> group;
				for (std::size_t i = 0; i < n; ++i) {
					if (descriptors[i].phase == phase) {
						group.push_back(i);
					}
				}

				std::vector<Edge> groupEdges;
				for (const Edge& edge : intraPhaseEdges) {
					if (descriptors[edge.consumer].phase == phase) {
						groupEdges.push_back(edge);
					}
				}

				std::vector<std::vector<std::size_t>> levels = LevelNodes(n, group, groupEdges);

				std::size_t scheduled = 0;
				for (const auto& level : levels) {
					scheduled += level.size();
				}
				if (scheduled < group.size()) {
					std::vector<std::size_t> placed;
					for (const auto& level : levels) {
						placed.insert(placed.end(), level.begin(), level.end());
					}
					std::string message = "graph contains a circular dependency; these nodes never "
										  "reach zero remaining dependencies:";
					for (std::size_t index : group) {
						if (std::find(placed.begin(), placed.end(), index) == placed.end()) {
							message += ' ';
							message += descriptors[index].name;
						}
					}
					return std::unexpected(ValidationError{.message = std::move(message), .missing = {}});
				}

				for (auto& level : levels) {
					const std::size_t stageIndex = m_schedule.stages.size();
					for (std::size_t index : level) {
						nodeStage[index] = stageIndex;
					}
					m_schedule.stages.push_back(ScheduleStage{std::move(level), {}, {}});
				}
			}

			for (const Edge& edge : intraPhaseEdges) {
				SynthesizeBarrier(descriptors, nodeStage, edge);
			}
			for (const Edge& edge : crossPhaseEdges) {
				SynthesizeBarrier(descriptors, nodeStage, edge);
			}

			for (auto& stage : m_schedule.stages) {
				std::erase_if(stage.nodes, [this](std::size_t index) { return !m_recipes[index].isActive; });
			}

			return {};
		}

		void Execute(NodeContext& ctx) {
			for (const auto& stage : m_schedule.stages) {
				for (std::size_t index : stage.nodes) {
					m_nodes[index].Execute(ctx);
				}
			}
		}

		// Legacy entry point for callers that only have a bare CommandBuffer (no pipeline/
		// bindless state to offer) -- wraps it in a default-initialized NodeContext and defers
		// to the overload above. PhysicalExecutionBackend::Execute builds a real NodeContext
		// per node instead of going through this.
		void Execute(CommandBuffer& cmd) {
			NodeContext ctx{};
			ctx.cmd = cmd;
			Execute(ctx);
		}

		// Executes a single scheduled node by index, bracketed by nothing itself -- callers
		// that need to interleave per-node work (barrier flushes, dynamic-rendering begin/end)
		// use this instead of the batch Execute() above.
		void ExecuteNode(std::size_t index, NodeContext& ctx) {
			if (index < m_nodes.size()) {
				m_nodes[index].Execute(ctx);
			}
		}

		void ExecuteNode(std::size_t index, CommandBuffer& cmd) {
			NodeContext ctx{};
			ctx.cmd = cmd;
			ExecuteNode(index, ctx);
		}

		[[nodiscard]] const Schedule& GetSchedule() const { return m_schedule; }

		[[nodiscard]] std::span<const Recipe> Recipes() const { return m_recipes; }

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

		// BFS layering (longest-path-from-source levels) restricted to one phase group: repeatedly
		// peel off every group member whose dependencies (within this group's own edges) are
		// already fully satisfied by prior levels. A node only leaves the frontier once every one
		// of its producers has been placed, so level(v) == 1 + max(level(u)) over all direct
		// producers u -- and no two nodes in the same level can have an edge between them, which
		// is exactly the "provably independent" property Schedule promises. `n` sizes the
		// dependents/indegree vectors to the full node count so `group`'s indices (global, not
		// contiguous) can index directly into them; only entries named by `group` are ever read.
		static std::vector<std::vector<std::size_t>>
		LevelNodes(std::size_t n, std::span<const std::size_t> group, const std::vector<Edge>& edges) {
			std::vector<std::vector<std::size_t>> dependents(n);
			std::vector<std::size_t>              indegree(n, 0);

			for (std::size_t consumer : group) {
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
			for (std::size_t index : group) {
				if (indegree[index] == 0) {
					frontier.push_back(index);
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

		// A node's access to a key it touches: ReadWrite if the key's base appears in both its
		// consumes and produces, Write if produces-only, Read otherwise. Matches by resolved base
		// rather than exact id so a node consuming VersionOf<K, N-1> and producing VersionOf<K, N>
		// (Modify<K, N>) reports ReadWrite for either version's edge -- both versions name the one
		// physical resource PhysicalResourceRegistry::ResolveId resolves them to, so a plain
		// exact-id match would wrongly report Read for the consumed (lower) version. For any
		// unversioned key this collapses to exact-id matching, unchanged from before.
		static AccessKind AccessOf(const NodeDescriptor& node, ResourceId key) {
			const ResourceId base = key->versionBase ? key->versionBase : key;
			auto             matchesBase = [base](ResourceId id) { return id == base || id->versionBase == base; };
			const bool       reads = std::any_of(node.consumes.begin(), node.consumes.end(), matchesBase);
			const bool       writes = std::any_of(node.produces.begin(), node.produces.end(), matchesBase);
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
