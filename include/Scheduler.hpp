#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"

namespace brassica::graph {

	inline constexpr std::size_t kNumExecutionDomains = 4;

	inline std::size_t DomainToLane(ExecutionDomain domain) {
		switch (domain) {
		case ExecutionDomain::Graphics:
			return 0;
		case ExecutionDomain::Compute:
			return 1;
		case ExecutionDomain::Transfer:
			return 2;
		case ExecutionDomain::Host:
			return 3;
		}
		return 0;
	}

	inline ExecutionDomain LaneToDomain(std::size_t lane) {
		switch (lane) {
		case 0:
			return ExecutionDomain::Graphics;
		case 1:
			return ExecutionDomain::Compute;
		case 2:
			return ExecutionDomain::Transfer;
		case 3:
			return ExecutionDomain::Host;
		}
		return ExecutionDomain::Graphics;
	}

	// Keyed by stable node name (NodeDescriptor::name -- a compile-time per-type string, one
	// instance per node type by this engine's own convention), not node index: a node's dense
	// active-only index (Graph::Compile's activeIndices remap) shifts frame to frame as other
	// nodes' Active() flags change, but an EMA accumulated across many frames (design plan's S5)
	// needs a key that means the same node every frame. std::string, not string_view: this
	// struct is meant to persist across frames (Engine owns one, long-lived), so its keys must
	// own their own storage rather than view into a transient per-frame NodeDescriptor span.
	struct DynamicWeights {
		std::unordered_map<std::string, int>                                  node_execution_weights;
		std::unordered_map<std::string, std::unordered_map<std::string, int>> edge_latencies;

		[[nodiscard]] int GetNodeWeight(std::string_view name, ExecutionDomain domain) const {
			auto it = node_execution_weights.find(std::string(name));
			if (it != node_execution_weights.end()) {
				return it->second;
			}
			switch (domain) {
			case ExecutionDomain::Graphics:
				return 10;
			case ExecutionDomain::Compute:
				return 8;
			case ExecutionDomain::Transfer:
				return 4;
			case ExecutionDomain::Host:
				return 2;
			}
			return 10;
		}

		[[nodiscard]] int GetEdgeLatency(
			std::string_view fromName,
			std::string_view toName,
			ExecutionDomain  srcDomain,
			ExecutionDomain  dstDomain
		) const {
			if (srcDomain == dstDomain) {
				return 0;
			}
			auto itFrom = edge_latencies.find(std::string(fromName));
			if (itFrom != edge_latencies.end()) {
				auto itTo = itFrom->second.find(std::string(toName));
				if (itTo != itFrom->second.end()) {
					return itTo->second;
				}
			}
			return 5;
		}

		// Design plan's S5: feeds a real GPU-measured duration (QueueRecorder's own per-batch
		// vkCmdWriteTimestamp2 readback, converted to microseconds and treated numerically as a
		// weight unit -- an explicit, uncalibrated guess, not a validated conversion; these
		// default weights were always dimensionless placeholders, so "microseconds" is simply
		// the least-arbitrary scale to adopt until the plan's own capture-comparison step
		// justifies something better) into this node's weight. The first measurement for a name
		// is adopted directly, not blended against the meaningless zero a brand-new map entry
		// would otherwise start from; every measurement after that is a standard EMA.
		void UpdateNodeWeight(std::string_view name, int measured, float emaAlpha = 0.2f) {
			const std::string key(name);
			auto              it = node_execution_weights.find(key);
			if (it == node_execution_weights.end()) {
				node_execution_weights[key] = measured;
				return;
			}
			it->second = static_cast<int>(emaAlpha * static_cast<float>(measured) + (1.0f - emaAlpha) * static_cast<float>(it->second));
		}
	};

	struct ScheduledTask {
		std::size_t     node_id;
		int             start_time;
		int             end_time;
		ExecutionDomain domain;
		std::size_t     queue_id;
	};

	// `names[index]` if present, else empty -- every DynamicWeights lookup below goes through
	// this instead of indexing `names` directly, so a caller with no stable names to offer yet
	// (every pre-S5 test, which only ever queries a default-constructed DynamicWeights{} and so
	// never actually looks anything up by key) can simply pass an empty span and get identical
	// behavior to before names existed.
	inline std::string_view NodeName(std::span<const std::string_view> names, std::size_t index) {
		return index < names.size() ? names[index] : std::string_view{};
	}

	inline int CalculateUpwardRank(
		std::size_t                                  nodeIndex,
		std::size_t                                  totalNodes,
		const std::vector<ExecutionDomain>&          domains,
		const std::vector<std::vector<std::size_t>>& successors,
		const DynamicWeights&                        weights,
		std::vector<int>&                            ranks,
		std::vector<bool>&                           visited,
		std::vector<bool>&                           inStack,
		std::span<const std::string_view>            names = {}
	) {
		if (visited[nodeIndex]) {
			return ranks[nodeIndex];
		}
		if (inStack[nodeIndex]) {
			return 0; // Break cycle recursion safely
		}

		inStack[nodeIndex] = true;

		int                   maxSuccessorCost = 0;
		const ExecutionDomain currDomain = (nodeIndex < domains.size()) ? domains[nodeIndex]
																		: ExecutionDomain::Graphics;
		const int             currWeight = weights.GetNodeWeight(NodeName(names, nodeIndex), currDomain);

		if (nodeIndex < successors.size()) {
			for (std::size_t succ : successors[nodeIndex]) {
				const ExecutionDomain succDomain = (succ < domains.size()) ? domains[succ] : ExecutionDomain::Graphics;
				int                   commOverhead =
					weights.GetEdgeLatency(NodeName(names, nodeIndex), NodeName(names, succ), currDomain, succDomain);
				int succRank =
					CalculateUpwardRank(succ, totalNodes, domains, successors, weights, ranks, visited, inStack, names);
				maxSuccessorCost = std::max(maxSuccessorCost, commOverhead + succRank);
			}
		}

		int rank = currWeight + maxSuccessorCost;
		ranks[nodeIndex] = rank;
		inStack[nodeIndex] = false;
		visited[nodeIndex] = true;
		return rank;
	}

	inline std::vector<int> CalculateAllUpwardRanks(
		std::size_t                                  totalNodes,
		const std::vector<ExecutionDomain>&          domains,
		const std::vector<std::vector<std::size_t>>& successors,
		const DynamicWeights&                        weights = {},
		std::span<const std::string_view>            names = {}
	) {
		std::vector<int>  ranks(totalNodes, 0);
		std::vector<bool> visited(totalNodes, false);
		std::vector<bool> inStack(totalNodes, false);
		for (std::size_t i = 0; i < totalNodes; ++i) {
			CalculateUpwardRank(i, totalNodes, domains, successors, weights, ranks, visited, inStack, names);
		}
		return ranks;
	}

	// -- S3: HEFT placement + batching (design plan's §Compiler) --------------------------------
	//
	// Deliberately decoupled from RelationalCompiler.hpp/Graph.hpp: this takes a plain edge list
	// (SchedulingEdge below), not RelEdge, so it stays testable in total isolation and carries no
	// risk to the schedule Graph::Compile() produces today -- nothing calls any of this yet.
	// Wiring it in (replacing Schedule's shape, touching the two execution backends) is S4's job.

	enum class QueuePlacementMode : std::uint8_t { GraphicsOnly, ForceDomain, Heft };

	// One edge CompileRelational's contracted DAG produces, typed real (a RAW/WAR/WAW resource
	// edge, or an explicit relation -- RelEdge::kind != Order) vs order-only (a phase marker or
	// virtual relation-target edge -- RelEdge::kind == Order). The distinction is exactly
	// RelEdge::key != nullptr vs == nullptr (RelationalCompiler.hpp) -- a thin adapter at the S4
	// call site builds this from RelationalResult::edges; this file never needs to know that
	// type exists.
	struct SchedulingEdge {
		std::size_t from;
		std::size_t to;
		bool        real;
	};

	// Which queues (ExecutionDomain standing in for queue identity: Graphics/Compute/Transfer
	// are real GPU queues, Host is "no GPU queue at all" -- it runs inline at record time) a
	// node of this domain may be placed on, under this placement mode. GraphicsOnly and
	// ForceDomain exist to stress/validate the sync layer with a single deterministic queue per
	// domain, independent of whether HEFT's cost model is trustworthy yet (design plan's own
	// "Default domain weights until S5" decision) -- Heft is the only mode that actually lets
	// PlaceNodes choose.
	inline std::vector<ExecutionDomain> EligibleQueues(ExecutionDomain domain, QueuePlacementMode mode) {
		if (domain == ExecutionDomain::Host) {
			return {};
		}
		if (mode == QueuePlacementMode::GraphicsOnly) {
			return {ExecutionDomain::Graphics};
		}
		if (mode == QueuePlacementMode::ForceDomain) {
			return {domain};
		}
		switch (domain) {
		case ExecutionDomain::Graphics:
			return {ExecutionDomain::Graphics};
		case ExecutionDomain::Compute:
			return {ExecutionDomain::Graphics, ExecutionDomain::Compute};
		case ExecutionDomain::Transfer:
			return {ExecutionDomain::Graphics, ExecutionDomain::Compute, ExecutionDomain::Transfer};
		case ExecutionDomain::Host:
			return {};
		}
		return {ExecutionDomain::Graphics};
	}

	// ancestors[v][u] is true iff u has a path to v over the *whole* edge set (real and
	// order-only alike). Needed by PlaceNodes' second ready_Q term ("every already-placed node
	// on this queue that reaches it", design plan's §Compiler): a node that does *not* reach the
	// one being placed must never serialize behind it on the same queue, or every queue would
	// collapse to strict rank order regardless of real independence -- exactly what batching
	// exists to avoid, and the reason this isn't simply "the latest thing already on this queue."
	inline std::vector<std::vector<bool>> ComputeAncestors(std::size_t n, std::span<const SchedulingEdge> edges) {
		std::vector<std::vector<std::size_t>> successors(n);
		std::vector<std::vector<std::size_t>> predecessors(n);
		std::vector<std::size_t>              indegree(n, 0);
		for (const auto& e : edges) {
			successors[e.from].push_back(e.to);
			predecessors[e.to].push_back(e.from);
			++indegree[e.to];
		}

		// Any valid topological order works -- ancestry only needs "every predecessor processed
		// before its dependents," not rank's specific priority ordering.
		std::vector<std::size_t> order;
		order.reserve(n);
		std::vector<std::size_t> remaining = indegree;
		std::vector<std::size_t> frontier;
		for (std::size_t i = 0; i < n; ++i) {
			if (remaining[i] == 0) {
				frontier.push_back(i);
			}
		}
		while (!frontier.empty()) {
			std::vector<std::size_t> next;
			for (std::size_t node : frontier) {
				order.push_back(node);
				for (std::size_t succ : successors[node]) {
					if (--remaining[succ] == 0) {
						next.push_back(succ);
					}
				}
			}
			frontier = std::move(next);
		}
		// order.size() < n would mean a cycle in `edges` -- CompileRelational already rejects
		// that before any of this runs (its own cycle check, independent of this file); any node
		// left unordered here just gets an all-false ancestor row, which is harmless -- it only
		// means PlaceNodes won't find a (nonexistent) reachability relationship for it.

		std::vector<std::vector<bool>> ancestors(n, std::vector<bool>(n, false));
		for (std::size_t node : order) {
			for (std::size_t p : predecessors[node]) {
				ancestors[node][p] = true;
				for (std::size_t i = 0; i < n; ++i) {
					if (ancestors[p][i]) {
						ancestors[node][i] = true;
					}
				}
			}
		}
		return ancestors;
	}

	struct PlacementResult {
		std::vector<ExecutionDomain> queue;  // per node; Host nodes get ExecutionDomain::Host
		std::vector<int>            start;   // per node
		std::vector<int>            finish;  // per node
	};

	// Per-node (by name) placement memory across frames -- design plan's S5 "placement
	// hysteresis": a node only actually switches queues after `requiredGainFrames` consecutive
	// frames where the naive EFT-minimizing choice beats its current ("sticky") queue by at
	// least `thresholdPercent`. Without this, small measurement noise in a real (S5) EMA could
	// thrash a node between queues frame to frame -- each switch costs a real barrier/semaphore
	// hop PlaceNodes' own per-frame EFT comparison has no way to see ahead of time, so flapping
	// could cost more than it ever saves. Keyed by name for the same reason DynamicWeights is.
	// An instance with no entries yet (freshly constructed) just adopts whatever the naive
	// choice is the first time each node is seen -- there's nothing to hold hysteresis against
	// before a node has a sticky queue at all.
	struct PlacementHysteresis {
		struct Entry {
			ExecutionDomain stickyQueue{ExecutionDomain::Graphics};
			int             consecutiveGainFrames = 0;
			bool            hasSticky = false;
		};
		std::unordered_map<std::string, Entry> entries;

		// naiveBestQueue/naiveBestFinish: PlaceNodes' own unconstrained EFT-minimizing answer for
		// this frame. stickyFinish: what EFT this node would get if forced to stay on its
		// current sticky queue (0 cost only in the Host edge case, never produced for a node that
		// reaches here). Returns the queue PlaceNodes should actually place this node on this
		// frame -- which still updates result.start/finish to match whichever queue this returns,
		// not naiveBestQueue's own numbers, when the two differ.
		ExecutionDomain Decide(
			std::string_view name,
			ExecutionDomain  naiveBestQueue,
			int              naiveBestFinish,
			int              stickyFinish,
			int              thresholdPercent,
			int              requiredGainFrames
		) {
			Entry& entry = entries[std::string(name)];
			if (!entry.hasSticky) {
				entry.hasSticky = true;
				entry.stickyQueue = naiveBestQueue;
				return entry.stickyQueue;
			}
			if (naiveBestQueue == entry.stickyQueue) {
				entry.consecutiveGainFrames = 0;
				return entry.stickyQueue;
			}
			const bool isGainFrame =
				stickyFinish > 0 && naiveBestFinish <= stickyFinish - (stickyFinish * thresholdPercent) / 100;
			if (!isGainFrame) {
				entry.consecutiveGainFrames = 0;
				return entry.stickyQueue;
			}
			if (++entry.consecutiveGainFrames >= requiredGainFrames) {
				entry.stickyQueue = naiveBestQueue;
				entry.consecutiveGainFrames = 0;
			}
			return entry.stickyQueue;
		}
	};

	// Append-only HEFT (design plan's §Compiler step 2 -- "why append-only rather than
	// insertion-based" explains the choice): processes nodes in rank order (strictly decreasing
	// along every edge, real or order-only, so every real predecessor of a node is already
	// placed by the time that node is reached) and, for each node, puts it on whichever eligible
	// queue gives the earliest finish time -- the max of (every real predecessor's finish, plus
	// comm cost if the predecessor landed on a different queue) and (every already-placed node
	// on that queue that reaches this one, per ancestors above). Ties prefer Graphics,
	// deterministically, via EligibleQueues' iteration order (Graphics always checked first) --
	// no separate tie-break needed. Host nodes are never placed on a queue at all: zero weight,
	// start/finish both 0, matching "a Host node never gets a GPU wait."
	// hysteresis: nullptr (every existing call site) means the literal naive-EFT-minimizing
	// placement, unchanged from before this parameter existed. A caller that owns a persistent
	// PlacementHysteresis across frames gets queue switches damped by thresholdPercent/
	// requiredGainFrames instead -- see that struct's own comment.
	inline PlacementResult PlaceNodes(
		std::size_t                          n,
		const std::vector<ExecutionDomain>&  domains,
		std::span<const SchedulingEdge>      edges,
		const DynamicWeights&                weights,
		QueuePlacementMode                   mode,
		std::span<const std::string_view>    names = {},
		PlacementHysteresis*                 hysteresis = nullptr,
		int                                  thresholdPercent = 20,
		int                                  requiredGainFrames = 3
	) {
		// Rank: every edge contributes (keeps same-queue phase order's priority signal intact,
		// matching today's leveling, which also levels over the whole contracted edge set) --
		// but comm cost (GetEdgeLatency's domain-mismatch default) only applies to real edges;
		// order-only ones are pinned to 0 regardless of domain, since they never cross a
		// semaphore boundary (ground rule: "Phase order is not a dependency").
		DynamicWeights rankWeights = weights;
		std::vector<std::vector<std::size_t>> successors(n);
		for (const auto& e : edges) {
			successors[e.from].push_back(e.to);
			if (!e.real) {
				rankWeights.edge_latencies[std::string(NodeName(names, e.from))][std::string(NodeName(names, e.to))] = 0;
			}
		}
		std::vector<int> rank = CalculateAllUpwardRanks(n, domains, successors, rankWeights, names);

		std::vector<std::vector<bool>> ancestors = ComputeAncestors(n, edges);

		std::vector<std::size_t> order(n);
		for (std::size_t i = 0; i < n; ++i) {
			order[i] = i;
		}
		std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return rank[a] > rank[b]; });

		std::vector<std::vector<std::size_t>> realPredecessors(n);
		for (const auto& e : edges) {
			if (e.real) {
				realPredecessors[e.to].push_back(e.from);
			}
		}

		PlacementResult result;
		result.queue.assign(n, ExecutionDomain::Host);
		result.start.assign(n, 0);
		result.finish.assign(n, 0);

		std::array<std::vector<std::pair<std::size_t, int>>, kNumExecutionDomains> placedOnQueue;

		for (std::size_t node : order) {
			const ExecutionDomain domain = domains[node];
			if (domain == ExecutionDomain::Host) {
				continue; // defaults above (Host, start=finish=0) already correct
			}

			const std::vector<ExecutionDomain> eligible = EligibleQueues(domain, mode);
			ExecutionDomain                    bestQueue = ExecutionDomain::Graphics;
			int                                bestReady = 0;
			int                                bestFinish = -1;
			// Only populated (and only consulted) when hysteresis != nullptr -- per-candidate
			// (ready, finish), so a sticky queue that isn't the naive best still has its own real
			// numbers available after the loop, rather than only ever knowing the winner's.
			std::array<int, kNumExecutionDomains> candidateReady{};
			std::array<int, kNumExecutionDomains> candidateFinish{-1, -1, -1, -1};

			for (ExecutionDomain candidate : eligible) {
				int ready = 0;
				for (std::size_t p : realPredecessors[node]) {
					// result.queue[p], not domains[p]: p was already placed (rank order
					// guarantees every real predecessor is), possibly onto a *different* queue
					// than its own declared domain (exactly what "sandwiched compute pass stays
					// on Graphics" means) -- charging comm cost against the domain it merely
					// *declared* rather than the queue it actually landed on would double-count
					// a hop that never happens. Host always costs 0 regardless of candidate: it
					// runs inline at record time, before any GPU submission, so there is no
					// semaphore for a GPU node to actually wait on (ground rule: "a Host node
					// never gets a GPU wait").
					const ExecutionDomain predQueue = result.queue[p];
					const int             comm = (predQueue == ExecutionDomain::Host || predQueue == candidate)
						? 0
						: weights.GetEdgeLatency(NodeName(names, p), NodeName(names, node), predQueue, candidate);
					ready = std::max(ready, result.finish[p] + comm);
				}
				for (const auto& [placedNode, placedFinish] : placedOnQueue[DomainToLane(candidate)]) {
					if (ancestors[node][placedNode]) {
						ready = std::max(ready, placedFinish);
					}
				}
				const int finish = ready + weights.GetNodeWeight(NodeName(names, node), candidate);
				candidateReady[DomainToLane(candidate)] = ready;
				candidateFinish[DomainToLane(candidate)] = finish;
				if (bestFinish < 0 || finish < bestFinish) {
					bestFinish = finish;
					bestReady = ready;
					bestQueue = candidate;
				}
			}

			ExecutionDomain chosenQueue = bestQueue;
			if (hysteresis != nullptr) {
				const std::string nodeName = std::string(NodeName(names, node));
				auto              existing = hysteresis->entries.find(nodeName);
				// A sticky queue only means something if it's actually one of *this* node's real
				// candidates this round (candidateFinish[...] >= 0 iff it was in `eligible`
				// above). It won't be if this name's own eligible set changed since the entry was
				// last written -- the same node *type* instantiated more than once with a
				// different runtime domain (test_multiqueue_sync.cpp's CaseA_Writer: two
				// instances, one Graphics, one Compute, same type name), or one instance whose
				// domain genuinely varies frame to frame (that same file's CaseE_Writer). Treating
				// a stale, no-longer-eligible sticky queue as real corrupts placement outright --
				// result.queue[node] could end up holding a queue EligibleQueues never offered
				// this node at all. Reseed from the naive best instead, exactly like first sight.
				const bool hasUsableSticky =
					existing != hysteresis->entries.end() && candidateFinish[DomainToLane(existing->second.stickyQueue)] >= 0;
				if (hasUsableSticky) {
					const std::size_t stickyLane = DomainToLane(existing->second.stickyQueue);
					chosenQueue = hysteresis->Decide(
						nodeName,
						bestQueue,
						bestFinish,
						candidateFinish[stickyLane],
						thresholdPercent,
						requiredGainFrames
					);
				} else {
					chosenQueue = bestQueue;
					PlacementHysteresis::Entry& entry = hysteresis->entries[nodeName];
					entry.hasSticky = true;
					entry.stickyQueue = bestQueue;
					entry.consecutiveGainFrames = 0;
				}
			}

			const std::size_t chosenLane = DomainToLane(chosenQueue);
			result.queue[node] = chosenQueue;
			result.start[node] = candidateFinish[chosenLane] < 0 ? bestReady : candidateReady[chosenLane];
			result.finish[node] = candidateFinish[chosenLane] < 0 ? (bestFinish < 0 ? 0 : bestFinish)
																   : candidateFinish[chosenLane];
			placedOnQueue[DomainToLane(chosenQueue)].push_back({node, result.finish[node]});
		}

		return result;
	}

	struct ScheduleBatchPlan {
		ExecutionDomain          queue; // Host batches are always a singleton
		std::vector<std::size_t> nodes;
	};

	// Batching (design plan's §Compiler step 3), simplified from true first-fit packing to
	// per-queue dependency levels -- level[v] = 1 + max(level[p]) over every real edge p->v
	// (regardless of queue) and every order-only edge p->v where p and v share a queue. That
	// second clause is what keeps same-queue phase order binding while a cross-queue marker
	// edge imposes nothing, matching the ground rules; the first is what makes two nodes with a
	// real edge between them always land in different batches, on any queue. A deliberate
	// simplification, not literal first-fit: it reproduces every property first-fit guarantees
	// (topological record order, no two edge-connected nodes sharing a batch, phase order intact
	// within a queue) via plain Kahn leveling instead of packing against PlaceNodes' literal EFT
	// numbers -- true packing (filling a gap left by a cross-queue wait with independent
	// same-queue work) is S5 scope, alongside the producer-side pre-transition work it's paired
	// with there. Host nodes get their own singleton batch each: no GPU segment, so there is
	// nothing to batch them *with*.
	inline std::vector<ScheduleBatchPlan>
	BuildBatches(std::size_t n, const std::vector<ExecutionDomain>& domains, std::span<const SchedulingEdge> edges, const PlacementResult& placement) {
		std::vector<std::vector<std::size_t>> levelPredecessors(n);
		for (const auto& e : edges) {
			if (e.real || placement.queue[e.from] == placement.queue[e.to]) {
				levelPredecessors[e.to].push_back(e.from);
			}
		}

		// Plain longest-path-from-source level, computed via a topological walk (same shape as
		// ComputeAncestors' own) rather than recursion -- n is a per-frame node count, not deep
		// enough to need memoized recursion's cycle guard, and CompileRelational already
		// rejects a real cycle before any of this runs.
		std::vector<std::size_t> indegree(n, 0);
		std::vector<std::vector<std::size_t>> dependents(n);
		for (std::size_t v = 0; v < n; ++v) {
			for (std::size_t p : levelPredecessors[v]) {
				dependents[p].push_back(v);
			}
			indegree[v] = levelPredecessors[v].size();
		}

		std::vector<int>         level(n, 0);
		std::vector<std::size_t> remaining = indegree;
		std::vector<std::size_t> frontier;
		for (std::size_t i = 0; i < n; ++i) {
			if (remaining[i] == 0) {
				frontier.push_back(i);
			}
		}
		while (!frontier.empty()) {
			std::vector<std::size_t> next;
			for (std::size_t node : frontier) {
				for (std::size_t dep : dependents[node]) {
					level[dep] = std::max(level[dep], level[node] + 1);
					if (--remaining[dep] == 0) {
						next.push_back(dep);
					}
				}
			}
			frontier = std::move(next);
		}

		std::vector<ScheduleBatchPlan> batches;
		// (queue lane, level) -> index into `batches`, so same-(queue,level) nodes merge into one
		// batch regardless of visit order.
		std::unordered_map<std::size_t, std::size_t> batchIndexOf;
		std::vector<std::size_t>                     order(n);
		for (std::size_t i = 0; i < n; ++i) {
			order[i] = i;
		}
		std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
			if (level[a] != level[b]) {
				return level[a] < level[b];
			}
			return a < b;
		});

		for (std::size_t node : order) {
			if (domains[node] == ExecutionDomain::Host) {
				batches.push_back(ScheduleBatchPlan{ExecutionDomain::Host, {node}});
				continue;
			}
			const std::size_t lane = DomainToLane(placement.queue[node]);
			// level fits comfortably in the high bits for any realistic per-frame node count;
			// this is a cache key, not a time value anything else reads.
			const std::size_t key = (static_cast<std::size_t>(level[node]) << 8) | lane;
			auto              it = batchIndexOf.find(key);
			if (it == batchIndexOf.end()) {
				batchIndexOf[key] = batches.size();
				batches.push_back(ScheduleBatchPlan{placement.queue[node], {node}});
			} else {
				batches[it->second].nodes.push_back(node);
			}
		}

		// Record order: by level (every batch's nodes share one level by construction above),
		// then queue, then the lowest original node index -- deterministic, and topological
		// since level strictly increases along every edge BuildBatches respects.
		auto levelOf = [&](const ScheduleBatchPlan& b) { return level[b.nodes.front()]; };
		auto minIndexOf = [&](const ScheduleBatchPlan& b) { return *std::min_element(b.nodes.begin(), b.nodes.end()); };
		std::stable_sort(batches.begin(), batches.end(), [&](const ScheduleBatchPlan& a, const ScheduleBatchPlan& b) {
			if (levelOf(a) != levelOf(b)) {
				return levelOf(a) < levelOf(b);
			}
			if (a.queue != b.queue) {
				return DomainToLane(a.queue) < DomainToLane(b.queue);
			}
			return minIndexOf(a) < minIndexOf(b);
		});

		return batches;
	}

	struct ScheduleSlackInfo {
		std::vector<int> latestStart;
		std::vector<int> latestFinish;
		std::vector<int> slack; // latestStart - placement.start; 0 means on the critical path
	};

	// Classic CPM backward pass over *real* edges only -- diagnostic (slack/critical path),
	// never a scheduling input, so it doesn't need BuildBatches' same-queue-order-only clause:
	// "how much can this node's finish slip before something real downstream is delayed" is
	// well-defined from real edges alone, independent of how same-queue phase order groups
	// batches. Ports identify_bottlenecks' backward pass (the deleted legacy scheduler, S0) onto
	// PlacementResult's per-node EST/EFT instead of a strict per-queue execution history, since
	// batching here allows several nodes to share one queue's batch with no relative order
	// between them at all.
	inline ScheduleSlackInfo ComputeSlack(
		std::size_t                          n,
		const std::vector<ExecutionDomain>&  domains,
		std::span<const SchedulingEdge>      edges,
		const DynamicWeights&                weights,
		const PlacementResult&               placement,
		std::span<const std::string_view>    names = {}
	) {
		ScheduleSlackInfo info;
		info.latestStart.assign(n, 0);
		info.latestFinish.assign(n, 0);
		info.slack.assign(n, 0);
		if (n == 0) {
			return info;
		}

		int makespan = 0;
		for (std::size_t i = 0; i < n; ++i) {
			makespan = std::max(makespan, placement.finish[i]);
		}
		info.latestFinish.assign(n, makespan);

		std::vector<std::vector<std::size_t>> realSuccessors(n);
		std::vector<std::vector<std::size_t>> allSuccessors(n);
		for (const auto& e : edges) {
			allSuccessors[e.from].push_back(e.to);
			if (e.real) {
				realSuccessors[e.from].push_back(e.to);
			}
		}

		// Ascending rank order visits every successor strictly before its predecessors (rank
		// decreases along every edge -- the same property PlaceNodes relies on, just walked in
		// the opposite direction here), so each node's successors already have a final
		// latestStart by the time this reads it.
		std::vector<int> rank = CalculateAllUpwardRanks(n, domains, allSuccessors, weights, names);
		std::vector<std::size_t> order(n);
		for (std::size_t i = 0; i < n; ++i) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return rank[a] < rank[b]; });

		for (std::size_t node : order) {
			for (std::size_t succ : realSuccessors[node]) {
				// Same Host/same-queue zero-cost rule as PlaceNodes' own comm computation --
				// see its comment.
				const int comm = (placement.queue[node] == ExecutionDomain::Host || placement.queue[node] == placement.queue[succ])
					? 0
					: weights.GetEdgeLatency(NodeName(names, node), NodeName(names, succ), placement.queue[node], placement.queue[succ]);
				info.latestFinish[node] = std::min(info.latestFinish[node], info.latestStart[succ] - comm);
			}
			const int weight = (domains[node] == ExecutionDomain::Host)
				? 0
				: weights.GetNodeWeight(NodeName(names, node), placement.queue[node]);
			info.latestStart[node] = info.latestFinish[node] - weight;
			info.slack[node] = info.latestStart[node] - placement.start[node];
		}

		return info;
	}

	inline std::string_view QueueName(ExecutionDomain domain) {
		switch (domain) {
		case ExecutionDomain::Graphics:
			return "G";
		case ExecutionDomain::Compute:
			return "C";
		case ExecutionDomain::Transfer:
			return "T";
		case ExecutionDomain::Host:
			return "Host";
		}
		return "?";
	}

	// Queue -> batch -> nodes, with EST/EFT/slack and the critical path marked -- the one piece
	// of identify_bottlenecks' report this keeps (minus its std::cout and global-namespace
	// types). `names` is caller-supplied (NodeDescriptor::name at the real S4 call site) rather
	// than looked up here, since this file deliberately carries no dependency on
	// RelationalCompiler.hpp/Node.hpp -- any name shorter than `n` entries falls back to "?".
	inline void DumpSchedule(
		std::span<const std::string_view>     names,
		const std::vector<ExecutionDomain>&   domains,
		const PlacementResult&                placement,
		const std::vector<ScheduleBatchPlan>& batches,
		const ScheduleSlackInfo&              slack
	) {
		auto nameOf = [&](std::size_t node) -> std::string_view {
			return node < names.size() ? names[node] : std::string_view{"?"};
		};

		spdlog::info("==== Schedule dump ({} batches) ====", batches.size());
		for (std::size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			const ScheduleBatchPlan& batch = batches[batchIndex];
			spdlog::info("  Batch {} [{}]:", batchIndex, QueueName(batch.queue));
			for (std::size_t node : batch.nodes) {
				const bool onCriticalPath = domains[node] != ExecutionDomain::Host && slack.slack[node] == 0;
				spdlog::info(
					"    {} EST={} EFT={} slack={}{}",
					nameOf(node),
					placement.start[node],
					placement.finish[node],
					slack.slack[node],
					onCriticalPath ? " [CRITICAL]" : ""
				);
			}
		}
	}

} // namespace brassica::graph
