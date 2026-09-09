#pragma once
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"

namespace brassica::graph {

	// A node's resource contract, in canonical form: what it consumes and what it produces.
	// This pair -- not the operations below -- is the composition currency. Operations don't
	// compose (a subgraph has no single "Create"); a (Consumes, Produces) pair does. Leaf
	// nodes author one via Declares<Ops...>; subgraphs derive one from their children. Both
	// expose it identically, so accumulation never has to special-case a subgraph.
	template <typename ConsumesList, typename ProducesList>
	struct ResourceInterface {
		using Consumes = ConsumesList;
		using Produces = ProducesList;
	};

	// -- operations: the authoring vocabulary ---------------------------------------

	template <ResourceRef K>
	struct Create {
		using Consumes = TypeList<>;
		using Produces = TypeList<K>;
	};

	template <ResourceRef K>
	struct Read {
		using Consumes = TypeList<K>;
		using Produces = TypeList<>;
	};

	// Modify<K> is a read-modify-write of a single key in place. Equivalent to
	// Transform<K, K>; kept as its own name because it's the common case.
	template <ResourceRef K>
	struct Modify {
		using Consumes = TypeList<K>;
		using Produces = TypeList<K>;
	};

	// Transform<From, To> consumes one key and produces a different one -- e.g. tonemapping
	// HdrColor into the Swapchain key. Unlike Modify, the two keys are not the same
	// allocation-identity; this is the one case a flat Reads/Writes/Creates split can't
	// express, which is why operations lower into a Consumes/Produces pair instead.
	template <ResourceRef From, ResourceRef To>
	struct Transform {
		using Consumes = TypeList<From>;
		using Produces = TypeList<To>;
	};

	// Lowers a list of operations to a single canonical ResourceInterface.
	template <typename... Ops>
	using Declares =
		ResourceInterface<Dedup<Concat<typename Ops::Consumes...>>, Dedup<Concat<typename Ops::Produces...>>>;

	template <typename T>
	concept DeclaresResources = requires {
		typename T::Resources::Consumes;
		typename T::Resources::Produces;
	};

	template <DeclaresResources T>
	using ConsumesOf = typename T::Resources::Consumes;

	template <DeclaresResources T>
	using ProducesOf = typename T::Resources::Produces;

} // namespace brassica::graph
