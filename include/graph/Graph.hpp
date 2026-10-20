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
#include "graph/RelationalCompiler.hpp"
#include "graph/ResourceKey.hpp"
#include "Scheduler.hpp"

namespace brassica::graph {

	struct ValidationError {
		std::string             message;
		std::vector<ResourceId> missing;
	};

	// Checks that every consumed key is produced by some node in the set (an Import<K> node
	// counts -- it produces keys and consumes nothing, Node.hpp) or is a History<K>, which is
	// self-satisfying regardless (see the isHistory check below).
	inline std::expected<void, ValidationError> ValidateRuntime(std::span<const NodeDescriptor> nodes) {
		std::vector<ResourceId> produced;
		for (const auto& node : nodes) {
			produced.insert(produced.end(), node.produces.begin(), node.produces.end());
		}

		std::vector<ResourceId> missing;
		for (const auto& node : nodes) {
			for (ResourceId key : node.consumes) {
				// A History<K> consumption is self-satisfying: its physical pairing comes from
				// PhysicalResourceRegistry::ProvisionTemporalPairs, which pairs it with K's own
				// catalog desc (or a sibling realization's), never from a graph-level producer
				// edge -- requiring a formal Import<History<K>>/PreviousFrame<K> producer here was
				// always a fiction this check imposed on top of a mechanism that doesn't need one.
				if (key->isHistory) {
					continue;
				}
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

		// Which GPU queue (Graphics/Compute/Transfer) a multi-queue backend should record this
		// stage's nodes into, or Host meaning "no GPU queue -- run inline, no segment, no
		// barriers" -- design plan's S4. Every node in one stage shares this queue by
		// construction (PlaceNodes/BuildBatches, Scheduler.hpp, only ever group same-queue
		// nodes together); it says nothing about any node's own *declared* Recipe::domain, which
		// still drives barrier derivation (DeriveImageState/DeriveBufferState) completely
		// unchanged -- a Compute-domain node placed onto Graphics still dispatches a real
		// compute shader, which is a property of the operation, not of which queue recorded it.
		// Always Graphics under QueuePlacementMode::GraphicsOnly (Compile()'s default and the
		// only mode PhysicalExecutionBackend ever compiles with), so the single-queue backend
		// never needs to read this field at all.
		ExecutionDomain queue = ExecutionDomain::Graphics;
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
		void SetQueueSet(const QueueSet& queues) { m_queues = queues; }

		[[nodiscard]] const QueueSet& GetQueueSet() const { return m_queues; }

		[[nodiscard]] std::uint32_t QueueFamilyForDomain(ExecutionDomain domain) const {
			switch (domain) {
			case ExecutionDomain::Graphics:
				return m_queues.graphics.familyIndex;
			case ExecutionDomain::Compute:
				return m_queues.compute.familyIndex;
			case ExecutionDomain::Transfer:
				return m_queues.transfer.familyIndex;
			case ExecutionDomain::Host:
				return kQueueFamilyIgnored;
			}
			return kQueueFamilyIgnored;
		}

		template <BuilderNode T, typename... Args>
		void Register(Args&&... args) {
			Add(NodeHandle::Make<T>(std::forward<Args>(args)...));
		}

		template <BuilderNode T>
		void RegisterRef(T& node) {
			Add(NodeHandle::MakeRef<T>(node));
		}

		void Add(NodeHandle handle) { m_nodes.push_back(std::move(handle)); }

		void Setup(const FrameContext& ctx) {
			m_recipes.clear();
			m_recipes.reserve(m_nodes.size());
			for (auto& node : m_nodes) {
				m_recipes.push_back(node.Setup(ctx));
			}
		}

		// Test/CI utility (design plan's §E/S5 exit criterion: "the force-all-active compile
		// runs in CI") -- overrides every node's own Setup()-declared activity to true, so
		// Compile() exercises the full graph regardless of what any node's real runtime state
		// (ServiceLocator managers, etc.) decided this frame. Call after Setup(), before
		// Compile() -- Setup() is what populates m_recipes in the first place. Compile()'s own
		// Requires<A>/.Required() fixed point can still turn a node back off after this (e.g. a
		// .Required() read with no registered-and-reachable writer at all), so "force all
		// active" means "don't let any node opt itself out," not "guarantee every node runs."
		void ForceAllActive() {
			for (auto& recipe : m_recipes) {
				recipe.isActive = true;
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
		// S3: swapped to the relational compiler (CompileRelational, graph/RelationalCompiler.hpp)
		// -- see the design plan's §C. Replaces three things that used to live inline here:
		// version lifting (a VersionedKey<K,N> consumer no longer needs rewriting to the highest
		// version an earlier phase produced -- the relational RAW/WAR derivation already finds
		// "the last writer before this reader" directly from the closure, version number or
		// not), the self-modify-chaining half of CollectEdges (replaced by the same RAW/WAR
		// derivation plus AnyOrder()'s deterministic tie-break), and the phase-violation error
		// (phase is now just a compat marker chain feeding into the same closure every other
		// relation does, not a hard partition -- a "later phase" producer for an "earlier phase"
		// reader is either a real ambiguity, caught by CompileRelational's own ambiguity check,
		// or legitimately fine, caught by nothing because it no longer needs special-casing).
		std::expected<void, ValidationError> Compile(QueuePlacementMode mode = QueuePlacementMode::GraphicsOnly) {
			std::vector<NodeDescriptor> descriptors;
			descriptors.reserve(m_nodes.size());
			for (const auto& node : m_nodes) {
				descriptors.push_back(node.Descriptor());
			}

			if (auto result = ValidateRuntime(descriptors); !result) {
				return result;
			}

			const std::size_t n = descriptors.size();

			RelationalResult rel = CompileRelational(m_nodes, m_recipes);
			if (!rel.ok) {
				return std::unexpected(ValidationError{.message = rel.error, .missing = {}});
			}
			if (!rel.ambiguities.empty()) {
				std::string message =
					"ambiguous write order: these resource/node pairs have two writers with nothing "
					"ordering them -- add a relation (After/Before) or mark both AnyOrder() if they "
					"genuinely commute:";
				std::vector<ResourceId> missing;
				for (const Ambiguity& a : rel.ambiguities) {
					message += "\n  '" + std::string(a.key->name) + "': '" + std::string(descriptors[a.a].name) +
						"' vs '" + std::string(descriptors[a.b].name) + "'";
					missing.push_back(a.key);
				}
				return std::unexpected(ValidationError{.message = std::move(message), .missing = std::move(missing)});
			}

			// Requires<A>/.Required() (§E, S5) can turn a node off beyond what its own Setup()
			// declared -- write the resolved activity back so Provision/the backend/
			// ZeroInitializeUndefinedReads all see it too, not just the schedule/barrier loop
			// below.
			for (std::size_t i = 0; i < n; ++i) {
				m_recipes[i].isActive = rel.activity[i];
			}
			m_zeroInitReads = std::move(rel.zeroInitReads);

			std::vector<std::size_t> nodeStage(n, 0);
			m_schedule.stages.clear();

			if (mode == QueuePlacementMode::GraphicsOnly) {
				// Verbatim today's path -- not just output-equivalent, the literal same
				// computation (rel.stages' own Kahn leveling), so this mode can never regress by
				// construction. Every stage happens to run on Graphics (the only mode
				// PhysicalExecutionBackend, the single-queue backend, ever compiles with -- it
				// never reads ScheduleStage::queue at all).
				for (auto& level : rel.stages) {
					const std::size_t stageIndex = m_schedule.stages.size();
					for (std::size_t index : level) {
						nodeStage[index] = stageIndex;
					}
					m_schedule.stages.push_back(ScheduleStage{std::move(level), {}, {}, ExecutionDomain::Graphics});
				}
			} else {
				// ForceDomain/Heft: real queue placement (design plan's §Compiler, S3's
				// PlaceNodes/BuildBatches). PlaceNodes/BuildBatches are activity-agnostic by
				// design (Vulkan-free, decoupled from CompileRelational entirely) -- they place
				// and batch every index 0..n-1 unconditionally, the same way rel.stages' own
				// Kahn leveling restricts itself to "activeReal" first. Remap to a dense
				// active-only index space here (and back again below) so an inactive node never
				// reaches PlaceNodes/BuildBatches at all, matching that same restriction -- the
				// first version of this branch skipped this and inactive nodes started
				// executing again, found by test_multiqueue_sync.cpp's existing inactive-node
				// cases.
				std::vector<std::size_t> activeIndices;
				activeIndices.reserve(n);
				std::vector<std::size_t> toActive(n, static_cast<std::size_t>(-1));
				for (std::size_t i = 0; i < n; ++i) {
					if (m_recipes[i].isActive) {
						toActive[i] = activeIndices.size();
						activeIndices.push_back(i);
					}
				}
				const std::size_t activeN = activeIndices.size();

				std::vector<ExecutionDomain> nodeDomains(activeN);
				for (std::size_t i = 0; i < activeN; ++i) {
					nodeDomains[i] = m_recipes[activeIndices[i]].domain;
				}
				// SchedulingEdge is built straight from rel.edges -- e.key != nullptr is exactly
				// "real" (a RAW/WAR/WAW resource edge or an explicit relation) vs "order-only" (a
				// phase marker or virtual relation-target edge), same filter the barrier-
				// synthesis loop below already applies for the same reason (never reason about an
				// edge CompileRelational already contracted out).
				std::vector<SchedulingEdge> edges;
				edges.reserve(rel.edges.size());
				for (const RelEdge& e : rel.edges) {
					if (e.from >= n || e.to >= n || !m_recipes[e.from].isActive || !m_recipes[e.to].isActive) {
						continue;
					}
					edges.push_back(SchedulingEdge{toActive[e.from], toActive[e.to], e.key != nullptr});
				}

				// Default weights only -- no measured-timestamp infrastructure exists yet
				// (design plan's S5). Placement under Heft mode is therefore structural, not
				// cost-based, until that lands.
				PlacementResult placement = PlaceNodes(activeN, nodeDomains, edges, DynamicWeights{}, mode);
				std::vector<ScheduleBatchPlan> batches = BuildBatches(activeN, nodeDomains, edges, placement);

				for (auto& batch : batches) {
					for (std::size_t& index : batch.nodes) {
						index = activeIndices[index]; // map back to the real node index
					}
					const std::size_t stageIndex = m_schedule.stages.size();
					for (std::size_t index : batch.nodes) {
						nodeStage[index] = stageIndex;
					}
					m_schedule.stages.push_back(ScheduleStage{std::move(batch.nodes), {}, {}, batch.queue});
				}
			}

			// Only resource edges (RAW/WAR/WAW, key != nullptr) need a barrier -- a pure Order
			// edge (phase marker, explicit relation) is scheduling constraint only. Skip any edge
			// touching a node CompileRelational already contracted out (a marker, or an inactive
			// real node, which never appears in nodeStage/m_schedule at all) -- per §C.4, "never
			// synthesize barriers for edges with an inactive endpoint".
			for (const RelEdge& e : rel.edges) {
				if (!e.key || e.from >= n || e.to >= n || !m_recipes[e.from].isActive || !m_recipes[e.to].isActive) {
					continue;
				}
				SynthesizeBarrier(descriptors, nodeStage, Edge{e.from, e.to, e.key});
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

		// Every (node index, key) this frame's Compile() found with no active writer preceding
		// it (§E's zero-init list, S5) -- the backend reads this to clear exactly these before
		// that node runs, instead of ZeroInitializeUndefinedReads' current History-image-only,
		// HasDefinedContents()-based check. Not yet consumed anywhere; this is the compiler-side
		// half, built and unit-tested in isolation same as every other S-stage mechanism here.
		[[nodiscard]] std::span<const std::pair<std::size_t, ResourceId>> ZeroInitReads() const {
			return m_zeroInitReads;
		}

		[[nodiscard]] std::span<const Recipe> Recipes() const { return m_recipes; }

		[[nodiscard]] std::span<const NodeHandle> Nodes() const { return m_nodes; }

	private:
		struct Edge {
			std::size_t producer;
			std::size_t consumer;
			ResourceId  key;
		};

		// A node's access to a key it touches: ReadWrite if the key appears in both its consumes
		// and produces, Write if produces-only, Read otherwise.
		static AccessKind AccessOf(const NodeDescriptor& node, ResourceId key) {
			auto       matchesBase = [key](ResourceId id) { return id == key; };
			const bool reads = std::any_of(node.consumes.begin(), node.consumes.end(), matchesBase);
			const bool writes = std::any_of(node.produces.begin(), node.produces.end(), matchesBase);
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

		std::vector<NodeHandle>                         m_nodes;
		std::vector<Recipe>                             m_recipes;
		Schedule                                        m_schedule;
		QueueSet                                        m_queues{};
		std::vector<std::pair<std::size_t, ResourceId>> m_zeroInitReads;
	};

} // namespace brassica::graph
