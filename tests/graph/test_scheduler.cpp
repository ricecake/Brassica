// Vulkan-free, like test_graph.cpp -- builds and runs via the normal CMake `tests` target (and,
// in principle, the fast syntax-only loop) with no Vulkan SDK involved. Proves S3's HEFT
// placement + batching algorithm (include/Scheduler.hpp) in total isolation: nothing here goes
// through CompileRelational/Graph::Compile() at all -- that integration (replacing Schedule's
// shape, touching the two execution backends) is S4's job. SchedulingEdge is hand-constructed
// throughout, standing in for what a thin S4 adapter will build from RelationalResult::edges.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "doctest/doctest.h"

#include "Scheduler.hpp"

using namespace brassica::graph;

namespace {

	// Reference implementation of today's leveling (RelationalCompiler.hpp's rel.stages: plain
	// Kahn/BFS frontier peeling over the whole edge set, ties irrelevant to grouping since the
	// tie-break there only orders *within* an already-decided level for display). Used only to
	// check BuildBatches' GraphicsOnly output partitions nodes identically.
	std::vector<std::vector<std::size_t>> ReferenceKahnLevels(std::size_t n, const std::vector<SchedulingEdge>& edges) {
		std::vector<std::vector<std::size_t>> dependents(n);
		std::vector<std::size_t>              indegree(n, 0);
		for (const auto& e : edges) {
			dependents[e.from].push_back(e.to);
			++indegree[e.to];
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
			levels.push_back(frontier);
			std::vector<std::size_t> next;
			for (std::size_t node : frontier) {
				for (std::size_t dep : dependents[node]) {
					if (--indegree[dep] == 0) {
						next.push_back(dep);
					}
				}
			}
			frontier = std::move(next);
		}
		return levels;
	}

	std::set<std::size_t> AsSet(const std::vector<std::size_t>& v) { return std::set<std::size_t>(v.begin(), v.end()); }

} // namespace

TEST_CASE("GraphicsOnly batching reproduces today's Kahn levels exactly") {
	// A small diamond-plus-tail DAG: 0 -> {1,2} -> 3 -> 4, plus an unrelated pair 5->6.
	constexpr std::size_t            n = 7;
	const std::vector<SchedulingEdge> edges{
		{0, 1, true},
		{0, 2, true},
		{1, 3, true},
		{2, 3, true},
		{3, 4, true},
		{5, 6, true},
	};
	const std::vector<ExecutionDomain> domains(n, ExecutionDomain::Graphics);

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::GraphicsOnly);
	for (std::size_t i = 0; i < n; ++i) {
		CHECK(placement.queue[i] == ExecutionDomain::Graphics);
	}

	std::vector<ScheduleBatchPlan> batches = BuildBatches(n, domains, edges, placement);
	std::vector<std::vector<std::size_t>> referenceLevels = ReferenceKahnLevels(n, edges);

	REQUIRE(batches.size() == referenceLevels.size());
	for (std::size_t i = 0; i < batches.size(); ++i) {
		CHECK(AsSet(batches[i].nodes) == AsSet(referenceLevels[i]));
	}
}

TEST_CASE("A compute pass sandwiched in a graphics chain stays on Graphics under default weights") {
	constexpr std::size_t n = 3;
	const std::vector<SchedulingEdge> edges{{0, 1, true}, {1, 2, true}};
	const std::vector<ExecutionDomain> domains{
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics,
	};

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::Heft);

	// Default weights: Graphics=10, Compute=8, cross-domain latency=5. Hopping to Compute costs
	// 5 (latency) + 8 (weight) = 13 past node 0's finish; staying on Graphics costs just 10.
	// Graphics wins -- this is the exact arithmetic HEFT's EFT comparison is supposed to make.
	CHECK(placement.queue[1] == ExecutionDomain::Graphics);
}

TEST_CASE("An independent compute chain goes to Compute under default weights") {
	constexpr std::size_t n = 3;
	const std::vector<SchedulingEdge> edges{{0, 1, true}, {1, 2, true}};
	const std::vector<ExecutionDomain> domains(n, ExecutionDomain::Compute);

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::Heft);

	for (std::size_t i = 0; i < n; ++i) {
		// No Graphics work anywhere in this chain to make hopping costly, and Compute's own
		// weight (8) is cheaper than Graphics' (10) -- every node should stay on Compute.
		CHECK(placement.queue[i] == ExecutionDomain::Compute);
	}
}

TEST_CASE("A Host node never gets a GPU wait") {
	// Host (0) -> Graphics (1): a real edge from a Host-domain node (an Import, in practice) to
	// a GPU node. Host nodes run inline at record time with zero weight -- node 1's start must
	// be 0, not bumped by any weight/latency contribution from node 0.
	constexpr std::size_t n = 2;
	const std::vector<SchedulingEdge> edges{{0, 1, true}};
	const std::vector<ExecutionDomain> domains{ExecutionDomain::Host, ExecutionDomain::Graphics};

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::Heft);

	CHECK(placement.queue[0] == ExecutionDomain::Host);
	CHECK(placement.start[0] == 0);
	CHECK(placement.finish[0] == 0);
	CHECK(placement.queue[1] == ExecutionDomain::Graphics);
	CHECK(placement.start[1] == 0);

	std::vector<ScheduleBatchPlan> batches = BuildBatches(n, domains, edges, placement);
	// The Host node gets its own singleton batch -- never grouped with (or ordered relative to
	// a GPU wait against) anything else.
	bool foundHostSingleton = false;
	for (const auto& batch : batches) {
		if (batch.queue == ExecutionDomain::Host) {
			CHECK(batch.nodes.size() == 1);
			CHECK(batch.nodes.front() == 0);
			foundHostSingleton = true;
		}
	}
	CHECK(foundHostSingleton);
}

TEST_CASE("Same-queue order-only edges stay binding; cross-queue ones impose nothing") {
	// Two independent chains under ForceDomain (0,1 Graphics; 2,3 Compute), joined only by an
	// order-only marker edge 1->2 (phase-marker-style). Ground rule: that edge binds nothing
	// across queues -- node 2 must not wait on node 1 at all.
	constexpr std::size_t n = 4;
	const std::vector<SchedulingEdge> edges{
		{0, 1, true},  // real, Graphics -> Graphics
		{1, 2, false}, // order-only, Graphics -> Compute (crosses queues)
		{2, 3, true},  // real, Compute -> Compute
	};
	const std::vector<ExecutionDomain> domains{
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute,
		ExecutionDomain::Compute,
	};

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::ForceDomain);
	CHECK(placement.queue[0] == ExecutionDomain::Graphics);
	CHECK(placement.queue[1] == ExecutionDomain::Graphics);
	CHECK(placement.queue[2] == ExecutionDomain::Compute);
	CHECK(placement.queue[3] == ExecutionDomain::Compute);

	// Node 2 has no real predecessor and no already-placed Compute-queue node reaching it, so
	// its ready time -- and start time -- must be 0, not bumped by node 1's order-only edge.
	CHECK(placement.start[2] == 0);

	std::vector<ScheduleBatchPlan> batches = BuildBatches(n, domains, edges, placement);
	// Node 1 and node 2 must not share a batch (different queues can never merge), and the
	// order-only edge must not force node 2's batch after node 1's -- both land at level 0 on
	// their own queue.
	auto batchIndexOfNode = [&](std::size_t node) -> std::size_t {
		for (std::size_t i = 0; i < batches.size(); ++i) {
			if (std::find(batches[i].nodes.begin(), batches[i].nodes.end(), node) != batches[i].nodes.end()) {
				return i;
			}
		}
		FAIL("node not found in any batch");
		return static_cast<std::size_t>(-1);
	};
	CHECK(batchIndexOfNode(1) != batchIndexOfNode(2));
}

TEST_CASE("Randomized DAGs: topological record order, acyclic batches, respected eligibility") {
	std::mt19937 rng(12345);

	for (int trial = 0; trial < 25; ++trial) {
		const std::size_t n = 8 + (rng() % 20);
		std::vector<ExecutionDomain> domains(n);
		for (std::size_t i = 0; i < n; ++i) {
			domains[i] = static_cast<ExecutionDomain>(rng() % 3); // Graphics/Compute/Transfer -- Host excluded, it has no edges to respect here
		}

		std::vector<SchedulingEdge> edges;
		// Only forward edges (i < j) -- guarantees acyclicity by construction, same as any real
		// contracted active DAG (CompileRelational rejects cycles before this would ever run).
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = i + 1; j < n; ++j) {
				if (rng() % 6 == 0) {
					edges.push_back(SchedulingEdge{i, j, (rng() % 2) == 0});
				}
			}
		}

		PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::Heft);

		// Eligibility respected.
		for (std::size_t i = 0; i < n; ++i) {
			std::vector<ExecutionDomain> eligible = EligibleQueues(domains[i], QueuePlacementMode::Heft);
			CHECK(std::find(eligible.begin(), eligible.end(), placement.queue[i]) != eligible.end());
		}

		std::vector<ScheduleBatchPlan> batches = BuildBatches(n, domains, edges, placement);

		std::vector<std::size_t> batchIndexOf(n);
		for (std::size_t i = 0; i < batches.size(); ++i) {
			for (std::size_t node : batches[i].nodes) {
				batchIndexOf[node] = i;
			}
		}

		for (const auto& e : edges) {
			const bool sameQueue = placement.queue[e.from] == placement.queue[e.to];
			if (e.real || sameQueue) {
				// Real edges always order (any queue); order-only edges only order same-queue.
				CHECK(batchIndexOf[e.from] < batchIndexOf[e.to]);
			}
		}

		// No two nodes directly connected by an edge that matters to batching share a batch --
		// a direct consequence of the strict '<' just checked, re-asserted at the batch level
		// for clarity.
		for (const auto& batch : batches) {
			for (std::size_t a : batch.nodes) {
				for (std::size_t b : batch.nodes) {
					if (a == b) {
						continue;
					}
					for (const auto& e : edges) {
						if (e.from == a && e.to == b) {
							const bool sameQueue = placement.queue[e.from] == placement.queue[e.to];
							const bool mattersToBatching = e.real || sameQueue;
							CHECK_FALSE(mattersToBatching);
						}
					}
				}
			}
		}
	}
}

TEST_CASE("DumpSchedule runs without throwing on a small real-ish plan") {
	constexpr std::size_t n = 4;
	const std::vector<SchedulingEdge> edges{{0, 1, true}, {1, 2, true}, {0, 3, true}};
	const std::vector<ExecutionDomain> domains{
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics,
		ExecutionDomain::Transfer,
	};
	const std::vector<std::string_view> names{"Writer", "Reader", "Final", "Copier"};

	PlacementResult placement = PlaceNodes(n, domains, edges, DynamicWeights{}, QueuePlacementMode::Heft);
	std::vector<ScheduleBatchPlan> batches = BuildBatches(n, domains, edges, placement);
	ScheduleSlackInfo slack = ComputeSlack(n, domains, edges, DynamicWeights{}, placement);

	CHECK_NOTHROW(DumpSchedule(names, domains, placement, batches, slack));

	// The longest chain (0 -> 1 -> 2) should have zero slack throughout -- it's the makespan.
	CHECK(slack.slack[0] == 0);
}
