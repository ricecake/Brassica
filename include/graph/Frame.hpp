#pragma once
#include "graph/Execution.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/SetupBuilder.hpp"
#include "graph/TypeList.hpp"

namespace brassica::graph {

	// Produces History<K> for every K in Temporal -- i.e. hands last frame's data to this
	// frame. Paired with NextFrame<Temporal> below: as long as both share one `using Temporal
	// = TypeList<Ks...>` alias, "anything PreviousFrame provides, NextFrame must require"
	// holds by construction, with nothing to check -- there is no production code sharing a
	// mismatched pair to protect against (a prior compile-time check enforcing this,
	// TemporalInvariantHolds, was deleted along with this conversion: it was built on
	// DeclaresResources/Resources, which a builder node doesn't have). A specialization of the
	// Import idea (Node.hpp): it produces resources and consumes nothing.
	template <typename Temporal>
	struct PreviousFrame;

	template <typename... Ks>
	struct PreviousFrame<TypeList<Ks...>> {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Host);
			(r.Create<History<Ks>>(), ...);
		}

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
	// forward as next frame's History<K>. The asymmetry (PreviousFrame produces History<K>,
	// this consumes the bare K) is why HistoryTarget exists at all.
	template <typename Temporal>
	struct NextFrame;

	template <typename... Ks>
	struct NextFrame<TypeList<Ks...>> {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Host);
			(r.Read<Ks>(), ...);
		}

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

} // namespace brassica::graph
