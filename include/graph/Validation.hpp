#pragma once
#include <array>
#include <string_view>
#include <type_traits>

#include "graph/Declaration.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"

// This header uses P2741 (user-generated static_assert messages), which is standardized in
// C++26 and available as an extension under -std=c++23 on the compilers this project targets.
// __cpp_static_assert reports 202306L regardless of -std, so the macro alone doesn't tell you
// whether the extension warning needs suppressing -- hence both the feature-test gate below
// and the pragma. If this ever stops working, AssertRenderableT falls back to the
// instantiation-note trick, which still names the missing keys, just in a `note:` instead of
// the primary diagnostic.
#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wc++26-extensions"
#endif

namespace brassica::graph {

	// FrameSpec accumulates the declared contracts of a set of nodes. Nodes and subgraphs
	// expose an identical ResourceInterface shape (see Declaration.hpp), so this accumulation
	// requires no special-casing for subgraphs at all.
	template <DeclaresResources... Nodes>
	struct FrameSpec {
		using Produces = Dedup<Concat<ProducesOf<Nodes>...>>;
		using Consumes = Dedup<Concat<ConsumesOf<Nodes>...>>;

		// Everything consumed that nothing in this set produces. Non-empty at the root means
		// the frame cannot render; non-empty in a subgraph is just that subgraph's own
		// interface, to be satisfied by whoever composes it in.
		using Unsatisfied = Difference<Consumes, Produces>;
	};

	template <typename Spec>
	using MissingOf = typename Spec::Unsatisfied;

	// A total function -- a set difference never errors -- so it is safe to assert *false* on.
	// That is what makes the negative test in tests/graph/test_graph.cpp an ordinary positive
	// assertion instead of requiring a `requires`-expression or a subprocess harness.
	template <typename Spec>
	inline constexpr bool IsRenderable = Size<MissingOf<Spec>> == 0;

	// -- diagnosable hard assert -----------------------------------------------------

	template <std::size_t N>
	struct FixedMessage {
		std::array<char, N> buf{};
		std::size_t         len = 0;

		constexpr void append(std::string_view s) {
			for (char c : s) {
				buf[len++] = c;
			}
		}

		constexpr const char* data() const { return buf.data(); }

		constexpr std::size_t size() const { return len; }
	};

	template <typename... Missing>
	struct MissingMsg {
		static constexpr auto Build() {
			FixedMessage<2048> m{};
			m.append("frame is not renderable; no node or import produces these resource keys:");
			((m.append(" "), m.append(TypeName<Missing>())), ...);
			return m;
		}

		static constexpr auto value = Build();
	};

	template <typename Unsatisfied>
	struct AssertRenderableT;

#if __cpp_static_assert >= 202306L
	template <typename... Missing>
	struct AssertRenderableT<TypeList<Missing...>> {
		static_assert(sizeof...(Missing) == 0, MissingMsg<Missing...>::value);
		static constexpr bool ok = true;
	};
#else
	template <typename Key>
	struct RESOURCE_IS_NOT_PRODUCED_BY_ANY_NODE {
		static_assert(sizeof(Key) == 0, "resource key is consumed but never produced -- see the type named above");
	};

	template <typename... Missing>
	struct AssertRenderableT<TypeList<Missing...>> {
		static_assert(
			sizeof...(Missing) == 0,
			"frame is not renderable; see instantiation notes below for the missing keys"
		);
		static constexpr bool ok = sizeof...(Missing) == 0;
	};
#endif

	// Instantiated only by Frame<...> (the root). A subgraph must never instantiate this --
	// its Unsatisfied set is its interface, not an error.
	template <typename Spec>
	inline constexpr bool AssertRenderable = AssertRenderableT<typename Spec::Unsatisfied>::ok;

	// -- PreviousFrame / NextFrame invariant ------------------------------------------
	// Anything PreviousFrame provides, NextFrame must require. Structural by construction
	// when both are generated from one `using Temporal = TypeList<Ks...>` (see Frame.hpp);
	// this check is the backstop for hand-rolled pairs. It must unwrap History because
	// PreviousFrame produces History<K> (last frame's data) while NextFrame consumes the
	// bare K (this frame's data, to be carried forward) -- the one place HistoryTarget
	// earns its keep.

	template <typename PrevProduces, typename NextConsumes>
	struct TemporalInvariantT;

	template <typename... Hs, typename NextConsumes>
	struct TemporalInvariantT<TypeList<Hs...>, NextConsumes>
		: std::bool_constant<(Contains<HistoryTarget<Hs>, NextConsumes> && ...)> {};

	template <DeclaresResources Prev, DeclaresResources Next>
	inline constexpr bool TemporalInvariantHolds = TemporalInvariantT<ProducesOf<Prev>, ConsumesOf<Next>>::value;

} // namespace brassica::graph

#if defined(__clang__)
	#pragma clang diagnostic pop
#endif
