#pragma once
#include <expected>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "graph/Declaration.hpp"
#include "graph/Dot.hpp"
#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"
#include "graph/Validation.hpp"

namespace brassica::graph {

	// A tightly-coupled cluster of nodes (SVGF, bloom, ...) that presents itself to a parent
	// as a single node. Its Resources are its *net* interface: whatever it could not satisfy
	// internally (Spec::Unsatisfied), plus everything it produces. Because that is exactly
	// the same ResourceInterface shape a leaf node exposes, a parent's FrameSpec absorbs a
	// Subgraph with no special-casing.
	//
	// Renderability is deliberately NOT asserted here -- unsatisfied inputs are this type's
	// interface, not a bug. Only Frame (the root) asserts.
	template <typename Spec>
	class Subgraph {
	public:
		using Resources = ResourceInterface<typename Spec::Unsatisfied, typename Spec::Produces>;

		// Compiles the inner graph here, not lazily -- Graph::Compile() culls inactive nodes
		// using this frame's Recipes (isActive), so it must run after this frame's own Setup,
		// exactly the order PhysicalExecutionBackend::Execute uses for the outer graph. A
		// backend that recognizes this node as a Subgraph (InnerGraphIfAny() != nullptr) reads
		// the now-current inner Schedule/Recipes directly rather than going through Execute
		// below -- see PhysicalExecutionBackend::RunSchedule's recursion.
		Recipe Setup(const FrameContext& ctx) {
			m_graph.Setup(ctx);
			// Spec::Unsatisfied is this Subgraph's declared net input -- by construction, no
			// node inside m_graph produces it, so it must be named here or ValidateRuntime
			// (inside Compile) would report it missing every time.
			if (auto result = m_graph.Compile(IdsOf<typename Spec::Unsatisfied>()); !result) {
				throw std::runtime_error("Subgraph compilation failed: " + result.error().message);
			}
			return Recipe{.domain = ExecutionDomain::Graphics};
		}

		// Naive fallback for a caller that executes this node directly rather than through a
		// backend that recurses into InnerGraphIfAny() (PhysicalExecutionBackend does, so this
		// never actually runs on the real render path) -- no barrier synthesis, no dynamic-
		// rendering wrapping, just runs the inner nodes in schedule order. Forwards the full
		// NodeContext so bindless/pipeline/globalSet state still reaches inner nodes even here.
		void Execute(NodeContext& ctx) { m_graph.Execute(ctx); }

		void Execute(CommandBuffer& cmd) { m_graph.Execute(cmd); }

		[[nodiscard]] Graph& InnerGraph() { return m_graph; }

		[[nodiscard]] const Graph& InnerGraph() const { return m_graph; }

	private:
		Graph m_graph;
	};

	template <typename Spec>
	struct NodeKindOfT<Subgraph<Spec>> {
		static constexpr NodeKind value = NodeKind::Subgraph;
	};

	template <typename Temporal>
	struct HistoryListT;

	template <typename... Ks>
	struct HistoryListT<TypeList<Ks...>> {
		using type = TypeList<History<Ks>...>;
	};

	// The set of keys carried across the frame boundary, wrapped as History<K>.
	template <typename Temporal>
	using HistoryList = typename HistoryListT<Temporal>::type;

	// Produces History<K> for every K in Temporal -- i.e. hands last frame's data to this
	// frame. Paired with NextFrame<Temporal> below, generated from the same Temporal list,
	// so the "anything PreviousFrame provides, NextFrame must require" invariant holds by
	// construction rather than by convention. A specialization of the Import idea (Node.hpp):
	// it produces resources and consumes nothing.
	template <typename Temporal>
	struct PreviousFrame {
		using Resources = ResourceInterface<TypeList<>, HistoryList<Temporal>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Host}; }

		void Execute(NodeContext&) {}
	};

	template <typename Temporal>
	struct NodeKindOfT<PreviousFrame<Temporal>> {
		static constexpr NodeKind value = NodeKind::PreviousFrame;
	};

	// Brackets every other node's phase from below, with no risk of a user-defined phase
	// colliding with it (Phase::PreviousFrame is INT32_MIN) -- so last frame's data is always
	// available before anything that might read it, with no resource-level wiring required.
	template <typename Temporal>
	struct PhaseOfT<PreviousFrame<Temporal>> {
		static constexpr Phase value = Phase::PreviousFrame;
	};

	// Consumes the bare (non-History) keys in Temporal -- this frame's values, to be carried
	// forward as next frame's History<K>. The asymmetry (Produces History<K>, Consumes K) is
	// the reason HistoryTarget exists at all.
	template <typename Temporal>
	struct NextFrame {
		using Resources = ResourceInterface<Temporal, TypeList<>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Host}; }

		void Execute(NodeContext&) {}
	};

	template <typename Temporal>
	struct NodeKindOfT<NextFrame<Temporal>> {
		static constexpr NodeKind value = NodeKind::NextFrame;
	};

	// Brackets every other node's phase from above, mirroring PreviousFrame's placement below.
	template <typename Temporal>
	struct PhaseOfT<NextFrame<Temporal>> {
		static constexpr Phase value = Phase::NextFrame;
	};

	// The public entry point. Renderability is asserted here, and only here: a Frame is the
	// root of a resource-flow graph, so an unsatisfied Consumes is a real bug, not another
	// level's interface.
	template <NodeLike... Nodes>
	class Frame {
	public:
		using Spec = FrameSpec<Nodes...>;
		static constexpr bool kRenderable = AssertRenderable<Spec>;

		template <typename T, typename... Args>
		void Register(Args&&... args) {
			m_graph.template Register<T>(std::forward<Args>(args)...);
		}

		void Setup(const FrameContext& ctx) { m_graph.Setup(ctx); }

		std::expected<void, ValidationError> Compile() { return m_graph.Compile(); }

		void Execute(CommandBuffer& cmd) { m_graph.Execute(cmd); }

		[[nodiscard]] std::string ToDot(std::string_view label = "Frame") const {
			return brassica::graph::ToDot(m_graph, label);
		}

	private:
		Graph m_graph;
	};

} // namespace brassica::graph
