#pragma once
#include <algorithm>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"

// Recursive Graphviz DOT export for a Graph. Free function rather than a Graph method, on
// purpose: Graph.hpp stays about scheduling, this file is the one place that knows what a
// debug rendering should look like. Safe to call before Compile() (draws declared structure
// only) or after (adds stage grouping and barrier annotations).

namespace brassica::graph {

	namespace detail {

		// Node ids are qualified by the path of Subgraph indices leading to them, so
		// identifiers stay unique once subgraphs are inlined as nested clusters
		// ("n0_2_1" = top-level node 0's subgraph, its node 2's subgraph, its node 1).
		inline std::string DotId(const std::vector<std::size_t>& path) {
			std::string id = "n";
			for (std::size_t index : path) {
				id += '_';
				id += std::to_string(index);
			}
			return id;
		}

		inline std::string Escape(std::string_view s) {
			std::string out;
			out.reserve(s.size());
			for (char c : s) {
				if (c == '"' || c == '\\') {
					out += '\\';
				}
				out += c;
			}
			return out;
		}

		// A node found by FindTouching, together with the domain it actually runs in --
		// Subgraph::Setup reports a fixed placeholder domain for the subgraph-as-a-whole, so
		// once we've resolved an edge down to the real inner node, its domain must be read
		// from the inner Graph's own Recipes(), not the outer Subgraph node's.
		struct TouchResult {
			std::vector<std::size_t> path;
			ExecutionDomain          domain;
		};

		// Finds the specific inner node (recursing through nested subgraphs) that produces or
		// consumes `key`, so a boundary-crossing edge can point at the real node -- with its
		// real domain -- rather than stopping at the cluster's outer box. Returns nullopt if
		// none is found (the caller then falls back to the subgraph node itself).
		inline std::optional<TouchResult>
		FindTouching(const Graph& graph, ResourceId key, bool wantProducer, std::vector<std::size_t> prefix) {
			const auto nodes = graph.Nodes();
			const auto recipes = graph.Recipes();
			for (std::size_t i = 0; i < nodes.size(); ++i) {
				const auto&              desc = nodes[i].Descriptor();
				const auto&              side = wantProducer ? desc.produces : desc.consumes;
				std::vector<std::size_t> path = prefix;
				path.push_back(i);

				if (std::find(side.begin(), side.end(), key) != side.end()) {
					const ExecutionDomain domain = i < recipes.size() ? recipes[i].domain : ExecutionDomain::Graphics;
					return TouchResult{std::move(path), domain};
				}
				if (const Graph* inner = nodes[i].InnerGraphIfAny()) {
					if (auto found = FindTouching(*inner, key, wantProducer, path)) {
						return found;
					}
				}
			}
			return std::nullopt;
		}

		inline std::string_view FillColorFor(NodeKind kind) {
			switch (kind) {
			case NodeKind::PreviousFrame:
			case NodeKind::NextFrame:
				return "gold";
			case NodeKind::Import:
				return "lightsteelblue";
			case NodeKind::Subgraph:
				return "white";
			case NodeKind::Ordinary:
				break;
			}
			return "lightblue";
		}

		inline std::string_view ShapeFor(NodeKind kind) {
			switch (kind) {
			case NodeKind::PreviousFrame:
			case NodeKind::NextFrame:
				return "cds";
			default:
				return "ellipse";
			}
		}

		// Edge decoration for a single (producer, consumer, key) resource-flow edge, derived
		// the same way Graph::Compile() derives it for barrier synthesis: same access-kind
		// rule, same RAR-is-hazard-free rule. Kept independent of Compile() having actually
		// run, so ToDot() works on an uncompiled graph too (domain then falls back to
		// whatever Setup() last produced, or the default Graphics if Setup() hasn't run either).
		struct EdgeStyle {
			std::string color;
			std::string style;
			std::string label;
		};

		// Every resource-flow edge has a producer that writes the key (that's what makes it a
		// producer), so every edge needs some synchronization -- there is no hazard-free case
		// to fall back to here, only which kind: same-domain barrier or cross-domain transfer.
		inline EdgeStyle StyleFor(ResourceId key, ExecutionDomain producerDomain, ExecutionDomain consumerDomain) {
			EdgeStyle edgeStyle;
			edgeStyle.label = std::string(key->name);
			edgeStyle.style = key->isHistory ? "dashed" : "solid";

			if (producerDomain != consumerDomain) {
				edgeStyle.color = "red";
				edgeStyle.label += " [transfer]";
			} else {
				edgeStyle.color = "black";
				edgeStyle.label += " [barrier]";
			}
			return edgeStyle;
		}

		inline void EmitGraph(
			const Graph&                    graph,
			const std::vector<std::size_t>& path,
			std::string_view                label,
			std::ostringstream&             out,
			bool                            asCluster
		) {
			const std::string clusterId = "cluster_" + DotId(path);
			if (asCluster) {
				out << "subgraph " << clusterId << " {\n";
				out << "label=\"" << Escape(label) << "\";\nstyle=dashed;\n";
			}

			const auto nodes = graph.Nodes();

			for (std::size_t i = 0; i < nodes.size(); ++i) {
				std::vector<std::size_t> nodePath = path;
				nodePath.push_back(i);
				const auto& desc = nodes[i].Descriptor();

				if (const Graph* inner = nodes[i].InnerGraphIfAny()) {
					EmitGraph(*inner, nodePath, desc.name, out, /*asCluster=*/true);
					continue;
				}

				out << DotId(nodePath) << " [label=\"" << Escape(desc.name) << "\", shape=" << ShapeFor(desc.kind)
				    << ", style=filled, fillcolor=" << FillColorFor(desc.kind) << "];\n";
			}

			// Resource-flow edges: one per (producer, consumer, key) touching this graph's own
			// nodes. Re-derived here rather than reusing Schedule so this also works before
			// Compile() has run.
			for (std::size_t consumer = 0; consumer < nodes.size(); ++consumer) {
				const auto& consumerDesc = nodes[consumer].Descriptor();
				for (ResourceId key : consumerDesc.consumes) {
					for (std::size_t producer = 0; producer < nodes.size(); ++producer) {
						if (producer == consumer) {
							continue;
						}
						const auto& producerDesc = nodes[producer].Descriptor();
						if (std::find(producerDesc.produces.begin(), producerDesc.produces.end(), key) ==
						    producerDesc.produces.end()) {
							continue;
						}

						const auto      recipes = graph.Recipes();
						ExecutionDomain producerDomain = producer < recipes.size() ? recipes[producer].domain
																				   : ExecutionDomain::Graphics;
						ExecutionDomain consumerDomain = consumer < recipes.size() ? recipes[consumer].domain
																				   : ExecutionDomain::Graphics;

						std::vector<std::size_t> producerPath = path;
						producerPath.push_back(producer);
						std::vector<std::size_t> consumerPath = path;
						consumerPath.push_back(consumer);

						// If the producer/consumer is itself a Subgraph, resolve the edge down to
						// the real inner node -- and its real domain, not the Subgraph's own
						// placeholder Recipe -- so the drawn edge and its barrier annotation both
						// reflect what's actually running, not the opaque box wrapping it.
						if (const Graph* innerProducer = nodes[producer].InnerGraphIfAny()) {
							if (auto found = FindTouching(*innerProducer, key, /*wantProducer=*/true, producerPath)) {
								producerPath = found->path;
								producerDomain = found->domain;
							}
						}
						if (const Graph* innerConsumer = nodes[consumer].InnerGraphIfAny()) {
							if (auto found = FindTouching(*innerConsumer, key, /*wantProducer=*/false, consumerPath)) {
								consumerPath = found->path;
								consumerDomain = found->domain;
							}
						}

						EdgeStyle edgeStyle = StyleFor(key, producerDomain, consumerDomain);

						out << DotId(producerPath) << " -> " << DotId(consumerPath) << " [label=\""
						    << Escape(edgeStyle.label) << "\", color=" << edgeStyle.color
						    << ", style=" << edgeStyle.style << "];\n";
					}
				}
			}

			// Temporal edge: PreviousFrame -> NextFrame for every History<K>/K pair they share,
			// drawn distinctly from ordinary resource-flow edges.
			for (std::size_t prev = 0; prev < nodes.size(); ++prev) {
				if (nodes[prev].Descriptor().kind != NodeKind::PreviousFrame) {
					continue;
				}
				for (std::size_t next = 0; next < nodes.size(); ++next) {
					if (nodes[next].Descriptor().kind != NodeKind::NextFrame) {
						continue;
					}
					for (ResourceId historyKey : nodes[prev].Descriptor().produces) {
						if (!historyKey->isHistory) {
							continue;
						}
						const auto& nextConsumes = nodes[next].Descriptor().consumes;
						if (std::find(nextConsumes.begin(), nextConsumes.end(), historyKey->historyTarget) ==
						    nextConsumes.end()) {
							continue;
						}

						std::vector<std::size_t> prevPath = path;
						prevPath.push_back(prev);
						std::vector<std::size_t> nextPath = path;
						nextPath.push_back(next);

						out << DotId(prevPath) << " -> " << DotId(nextPath) << " [label=\""
						    << Escape(historyKey->historyTarget->name)
						    << "\", color=purple, style=dotted, penwidth=2, constraint=false];\n";
					}
				}
			}

			if (asCluster) {
				out << "}\n";
			}
		}

	} // namespace detail

	inline std::string ToDot(const Graph& graph, std::string_view label = "Frame") {
		std::ostringstream out;
		out << "digraph {\n";
		out << "rankdir=LR;\n";
		out << "label=\"" << detail::Escape(label) << "\";\n";

		detail::EmitGraph(graph, {}, label, out, /*asCluster=*/false);

		const auto& schedule = graph.GetSchedule();
		for (std::size_t stageIndex = 0; stageIndex < schedule.stages.size(); ++stageIndex) {
			const auto& stage = schedule.stages[stageIndex];
			if (stage.nodes.size() < 2) {
				continue;
			}
			out << "{ rank=same; ";
			for (std::size_t index : stage.nodes) {
				out << detail::DotId({index}) << "; ";
			}
			out << "}\n";
		}

		out << "}\n";
		return out.str();
	}

} // namespace brassica::graph
