#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

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

	struct DynamicWeights {
		std::unordered_map<std::size_t, int>                                  node_execution_weights;
		std::unordered_map<std::size_t, std::unordered_map<std::size_t, int>> edge_latencies;

		[[nodiscard]] int GetNodeWeight(std::size_t nodeIndex, ExecutionDomain domain) const {
			auto it = node_execution_weights.find(nodeIndex);
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

		[[nodiscard]] int
		GetEdgeLatency(std::size_t from, std::size_t to, ExecutionDomain srcDomain, ExecutionDomain dstDomain) const {
			if (srcDomain == dstDomain) {
				return 0;
			}
			auto itFrom = edge_latencies.find(from);
			if (itFrom != edge_latencies.end()) {
				auto itTo = itFrom->second.find(to);
				if (itTo != itFrom->second.end()) {
					return itTo->second;
				}
			}
			return 5;
		}
	};

	struct ScheduledTask {
		std::size_t     node_id;
		int             start_time;
		int             end_time;
		ExecutionDomain domain;
		std::size_t     queue_id;
	};

	inline int CalculateUpwardRank(
		std::size_t                                  nodeIndex,
		std::size_t                                  totalNodes,
		const std::vector<ExecutionDomain>&          domains,
		const std::vector<std::vector<std::size_t>>& successors,
		const DynamicWeights&                        weights,
		std::vector<int>&                            ranks,
		std::vector<bool>&                           visited,
		std::vector<bool>&                           inStack
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
		const int             currWeight = weights.GetNodeWeight(nodeIndex, currDomain);

		if (nodeIndex < successors.size()) {
			for (std::size_t succ : successors[nodeIndex]) {
				const ExecutionDomain succDomain = (succ < domains.size()) ? domains[succ] : ExecutionDomain::Graphics;
				int                   commOverhead = weights.GetEdgeLatency(nodeIndex, succ, currDomain, succDomain);
				int                   succRank =
					CalculateUpwardRank(succ, totalNodes, domains, successors, weights, ranks, visited, inStack);
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
		const DynamicWeights&                        weights = {}
	) {
		std::vector<int>  ranks(totalNodes, 0);
		std::vector<bool> visited(totalNodes, false);
		std::vector<bool> inStack(totalNodes, false);
		for (std::size_t i = 0; i < totalNodes; ++i) {
			CalculateUpwardRank(i, totalNodes, domains, successors, weights, ranks, visited, inStack);
		}
		return ranks;
	}

} // namespace brassica::graph

struct Node {
	int id;
	int allowed_queue;
};

struct Edge {
	int from;
	int to;
};

using DynamicWeights = brassica::graph::DynamicWeights;

struct ScheduledTask {
	int node_id;
	int start_time;
	int end_time;
	int queue_id;
};

struct CompilerGraph {
	std::unordered_map<int, Node>              nodes;
	std::unordered_map<int, std::vector<Edge>> adj_list;
	std::unordered_map<int, std::vector<Edge>> rev_adj_list; // Required for backward pass
	std::unordered_map<int, int>               upward_ranks;

	void add_node(int id, int queue) { nodes[id] = {id, queue}; }

	void add_edge(int from, int to) {
		adj_list[from].push_back({from, to});
		rev_adj_list[to].push_back({from, to});
	}

	void clear_evaluations() { upward_ranks.clear(); }
};

// Calculates the Upward Rank for priority sorting
inline int calculate_upward_rank(int node_id, CompilerGraph& graph, const DynamicWeights& weights) {
	if (graph.upward_ranks.find(node_id) != graph.upward_ranks.end()) {
		return graph.upward_ranks[node_id];
	}

	int         max_successor_cost = 0;
	const Node& curr_node = graph.nodes[node_id];
	int current_weight = weights.node_execution_weights.count(node_id) ? weights.node_execution_weights.at(node_id)
																	   : 10;

	if (graph.adj_list.find(node_id) != graph.adj_list.end()) {
		for (const auto& edge : graph.adj_list[node_id]) {
			int comm_overhead = 0;
			if (curr_node.allowed_queue != graph.nodes[edge.to].allowed_queue) {
				if (weights.edge_latencies.count(edge.from) && weights.edge_latencies.at(edge.from).count(edge.to)) {
					comm_overhead = weights.edge_latencies.at(edge.from).at(edge.to);
				} else {
					comm_overhead = 5;
				}
			}
			int successor_rank = calculate_upward_rank(edge.to, graph, weights);
			max_successor_cost = std::max(max_successor_cost, comm_overhead + successor_rank);
		}
	}

	int rank = current_weight + max_successor_cost;
	graph.upward_ranks[node_id] = rank;
	return rank;
}

// Main scheduling and bottleneck detection algorithm
inline void identify_bottlenecks(CompilerGraph& graph, const DynamicWeights& weights, int num_queues) {
	graph.clear_evaluations();
	for (const auto& pair : graph.nodes) {
		calculate_upward_rank(pair.first, graph, weights);
	}

	std::unordered_map<int, int>           in_degree;
	std::unordered_map<int, int>           data_ready_time;
	std::unordered_map<int, ScheduledTask> final_schedule;

	for (const auto& pair : graph.nodes) {
		in_degree[pair.first] = 0;
		data_ready_time[pair.first] = 0;
	}
	for (const auto& pair : graph.adj_list) {
		for (const auto& edge : pair.second) {
			in_degree[edge.to]++;
		}
	}

	auto rank_comp = [&](int left, int right) { return graph.upward_ranks[left] < graph.upward_ranks[right]; };
	using QueueType = std::priority_queue<int, std::vector<int>, decltype(rank_comp)>;

	std::vector<QueueType> ready_lanes;
	for (int i = 0; i < num_queues; ++i) {
		ready_lanes.emplace_back(rank_comp);
	}

	std::vector<int> queue_free_at(num_queues, 0);
	// Track execution sequences inside each individual queue for the backward pass
	std::vector<std::vector<int>> queue_execution_histories(num_queues);

	using Event = std::pair<int, int>;
	std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue;

	auto try_schedule_task = [&](int node_id, int current_time) {
		const auto& node = graph.nodes[node_id];
		int         q = node.allowed_queue;
		int         weight = weights.node_execution_weights.at(node_id);

		int start_time = std::max({current_time, queue_free_at[q], data_ready_time[node_id]});
		int end_time = start_time + weight;

		queue_free_at[q] = end_time;
		event_queue.push({end_time, node_id});
		final_schedule[node_id] = {node_id, start_time, end_time, q};
		queue_execution_histories[q].push_back(node_id);
	};

	for (const auto& pair : graph.nodes) {
		if (in_degree[pair.first] == 0) {
			ready_lanes[pair.second.allowed_queue].push(pair.first);
		}
	}

	for (int q = 0; q < num_queues; ++q) {
		if (!ready_lanes[q].empty()) {
			int node_id = ready_lanes[q].top();
			ready_lanes[q].pop();
			try_schedule_task(node_id, 0);
		}
	}

	int total_makespan = 0;
	while (!event_queue.empty()) {
		auto [event_time, completed_node] = event_queue.top();
		event_queue.pop();

		total_makespan = event_time;
		int q_finished = graph.nodes[completed_node].allowed_queue;

		if (graph.adj_list.find(completed_node) != graph.adj_list.end()) {
			for (const auto& edge : graph.adj_list[completed_node]) {
				int target_q = graph.nodes[edge.to].allowed_queue;
				int transfer_overhead = 0;
				if (q_finished != target_q) {
					if (weights.edge_latencies.count(edge.from) &&
					    weights.edge_latencies.at(edge.from).count(edge.to)) {
						transfer_overhead = weights.edge_latencies.at(edge.from).at(edge.to);
					}
				}
				int arrival_time = total_makespan + transfer_overhead;
				data_ready_time[edge.to] = std::max(data_ready_time[edge.to], arrival_time);

				in_degree[edge.to]--;
				if (in_degree[edge.to] == 0) {
					ready_lanes[target_q].push(edge.to);
				}
			}
		}

		for (int q = 0; q < num_queues; ++q) {
			if (queue_free_at[q] <= total_makespan && !ready_lanes[q].empty()) {
				int next_node = ready_lanes[q].top();
				ready_lanes[q].pop();
				try_schedule_task(next_node, total_makespan);
			}
		}
	}

	// ==========================================
	// BACKWARD PASS FOR BOTTLENECK ANALYSIS
	// ==========================================
	std::unordered_map<int, int> latest_start;
	std::unordered_map<int, int> latest_end;

	// Initialize all tasks to end at least at the total makespan limit
	for (const auto& pair : graph.nodes) {
		latest_end[pair.first] = total_makespan;
		latest_start[pair.first] = total_makespan - weights.node_execution_weights.at(pair.first);
	}

	// Constraint 1: Sequential dependency within the same execution queue
	for (int q = 0; q < num_queues; ++q) {
		const auto& history = queue_execution_histories[q];
		if (history.empty())
			continue;
		for (size_t i = history.size() - 1; i > 0; --i) {
			int current = history[i];
			int previous = history[i - 1];
			latest_end[previous] = std::min(latest_end[previous], latest_start[current]);
			latest_start[previous] = latest_end[previous] - weights.node_execution_weights.at(previous);
		}
	}

	// Constraint 2: Data dependency cross-lane requirements (Reverse topological sweep)
	// To do this simply, we sort nodes by scheduled end_time descending
	std::vector<int> sorted_node_ids;
	for (const auto& pair : graph.nodes)
		sorted_node_ids.push_back(pair.first);
	std::sort(sorted_node_ids.begin(), sorted_node_ids.end(), [&](int a, int b) {
		return final_schedule[a].end_time > final_schedule[b].end_time;
	});

	for (int node_id : sorted_node_ids) {
		if (graph.rev_adj_list.find(node_id) != graph.rev_adj_list.end()) {
			for (const auto& edge : graph.rev_adj_list[node_id]) {
				int parent = edge.from;
				int child = edge.to;
				int transfer_overhead = 0;
				if (graph.nodes[parent].allowed_queue != graph.nodes[child].allowed_queue) {
					if (weights.edge_latencies.count(parent) && weights.edge_latencies.at(parent).count(child)) {
						transfer_overhead = weights.edge_latencies.at(parent).at(child);
					}
				}
				// The parent must end before the child starts minus data transfer flight time
				latest_end[parent] = std::min(latest_end[parent], latest_start[child] - transfer_overhead);
				latest_start[parent] = latest_end[parent] - weights.node_execution_weights.at(parent);
			}
		}
	}

	// ==========================================
	// REPORT GENERATION & CRITICAL PATH SLACK
	// ==========================================
	std::cout << "====================================================\n";
	std::cout << "SCHEDULE TIMELINE & BOTTLENECK ANALYSIS\n";
	std::cout << "====================================================\n";

	std::vector<ScheduledTask> sorted_plan;
	for (const auto& pair : final_schedule)
		sorted_plan.push_back(pair.second);
	std::sort(sorted_plan.begin(), sorted_plan.end(), [](const ScheduledTask& a, const ScheduledTask& b) {
		return a.start_time < b.start_time;
	});

	std::vector<int> critical_path_nodes;

	for (const auto& t : sorted_plan) {
		int         slack = latest_start[t.node_id] - t.start_time;
		std::string tag = (slack == 0) ? " [BOTTLENECK!]" : "";
		if (slack == 0)
			critical_path_nodes.push_back(t.node_id);

		std::cout << "Task " << t.node_id << " -> Q[" << t.queue_id << "] | "
		          << "Interval: [" << t.start_time << " -> " << t.end_time << "] | "
		          << "Slack: " << slack << tag << "\n";
	}

	std::cout << "----------------------------------------------------\n";
	std::cout << "Total System Makespan: " << total_makespan << " units\n";
	std::cout << "Bottleneck Critical Path Sequence: ";
	for (size_t i = 0; i < critical_path_nodes.size(); ++i) {
		std::cout << critical_path_nodes[i] << (i == critical_path_nodes.size() - 1 ? "" : " -> ");
	}
	std::cout << "\n====================================================\n";
}
