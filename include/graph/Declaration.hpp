#pragma once
#include <cstddef>

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
	//
	// Modify<K, Version> (Version >= 1) is the multi-writer case: a chain of nodes each
	// read-modify-writing the same physical resource in sequence, e.g. deferred shading producing
	// Modify<Swapchain, 1> and a later forward pass producing Modify<Swapchain, 2>. VersionOf<K, 0>
	// collapses to K itself (ResourceKey.hpp), so consuming version N-1 and producing version N is
	// exactly Modify<K> generalized -- version 0 is whatever Create<K>/an import already produces.
	// PhysicalResourceRegistry::ResolveId (PhysicalRegistry.hpp) maps every version back to the one
	// underlying physical resource automatically, from the type alone -- no node-side registration.
	//
	// This is deliberately not how two writers of one resource get *ordered* -- that's a node's
	// Phase (Node.hpp), which is coarser and doesn't require naming a version number at all. Modify
	// with a version is for the finer-grained case: multiple writers in the *same* phase, where the
	// write order needs to be explicit because phase alone can't disambiguate it.
	template <ResourceRef K, std::size_t Version = 0>
	struct Modify {
		using Consumes = TypeList<VersionOf<K, Version - 1>>;
		using Produces = TypeList<VersionOf<K, Version>>;
	};

	template <ResourceRef K>
	struct Modify<K, 0> {
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
