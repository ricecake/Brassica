#pragma once
#include <concepts>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/ResourceKey.hpp"

namespace brassica::graph {

	// Forward-declared only: Node.hpp needs to hand back a pointer to a node's inner Graph
	// (for recursive rendering, see Dot.hpp) without depending on Graph.hpp, which itself
	// includes Node.hpp.
	class Graph;

	// Single definition. A node must declare its resource contract (DeclaresResources) and
	// implement the two-phase Setup/Execute shape: Setup evaluates render-context state
	// (resolution, etc.) into a concrete Recipe for this frame, Execute records commands via
	// NodeContext (pipeline/globalSet/bindless access). Every node in the codebase was ported
	// off the older bare-CommandBuffer& signature as of the Node/Pass unification's Stage 8 --
	// there is no dual-dispatch left to keep here.
	template <typename T>
	concept NodeLike = DeclaresResources<T> && requires(T t, const FrameContext& ctx, NodeContext& nctx) {
		{ t.Setup(ctx) } -> std::same_as<Recipe>;
		{ t.Execute(nctx) };
	};

	// What kind of special role (if any) a node plays, captured at NodeHandle::Make<T> time
	// from compile-time knowledge of T. Used by Dot.hpp to style/route rendering without
	// string-matching TypeName() output. Ordinary is the default for any type that doesn't
	// specialize NodeKindOfT; Import is specialized below, PreviousFrame/NextFrame/Subgraph
	// are specialized in Frame.hpp where those types are defined.
	enum class NodeKind : std::uint8_t { Ordinary, Import, PreviousFrame, NextFrame, Subgraph };

	template <typename T>
	struct NodeKindOfT {
		static constexpr NodeKind value = NodeKind::Ordinary;
	};

	template <typename T>
	inline constexpr NodeKind NodeKindOf = NodeKindOfT<T>::value;

	// A node's coarse ordering (Execution.hpp's Phase), captured at NodeHandle::Make<T> time from
	// compile-time knowledge of T -- mirrors NodeKindOfT exactly. Default is Phase::Default for any
	// type that doesn't opt in; PreviousFrame/NextFrame are specialized in Frame.hpp, the same place
	// their NodeKindOfT specializations live. Most leaf nodes should prefer the `kPhase` detection
	// just below over specializing this directly -- specializing is for framework-level node kinds.
	template <typename T>
	struct PhaseOfT {
		static constexpr Phase value = Phase::Default;
	};

	// A node opts into a non-default phase by declaring `static constexpr graph::Phase kPhase = ...;`
	// rather than specializing PhaseOfT -- the same ergonomics as Recipe::domain, just compile-time.
	template <typename T>
	concept HasPhase = requires { T::kPhase; };

	template <HasPhase T>
	struct PhaseOfT<T> {
		static constexpr Phase value = T::kPhase;
	};

	template <typename T>
	inline constexpr Phase PhaseOf = PhaseOfT<T>::value;

	// A node's declared contract, lowered to runtime spans. This is what the compile-time
	// declaration looks like once type-erased -- see NodeHandle::Make below, the one place
	// where T is still known and the spans can be captured.
	struct NodeDescriptor {
		std::string_view            name;
		std::span<const ResourceId> consumes;
		std::span<const ResourceId> produces;
		NodeKind                    kind = NodeKind::Ordinary;
		Phase                       phase = Phase::Default;
	};

	template <typename T>
	concept HasInnerGraph = requires(const T& t) {
		{ t.InnerGraph() } -> std::same_as<const Graph&>;
	};

	// Type-erased holder for any NodeLike type, including Subgraph (Frame.hpp) since it
	// satisfies the same concept. Owns the node; the runtime Graph holds these.
	class NodeHandle {
		struct IErased {
			virtual ~IErased() = default;
			virtual Recipe Setup(const FrameContext&) = 0;
			virtual void   Execute(NodeContext&) = 0;

			// Non-null only for a Subgraph -- the hook that lets Dot.hpp descend into it
			// recursively without NodeHandle/Graph knowing Subgraph exists.
			virtual const Graph* InnerGraphIfAny() const { return nullptr; }

			// Mutable counterpart of the above, for PhysicalExecutionBackend::RunSchedule to
			// recurse into a Subgraph's inner graph with the same Provision/barrier/dynamic-
			// rendering treatment it gives this one -- Dot.hpp only ever reads, so it doesn't
			// need this overload, but a backend actually running the inner schedule does.
			virtual Graph* InnerGraphIfAny() { return nullptr; }
		};

		template <NodeLike T>
		struct Erased final: IErased {
			T value;

			template <typename... Args>
			explicit Erased(Args&&... args): value(std::forward<Args>(args)...) {}

			Recipe Setup(const FrameContext& ctx) override { return value.Setup(ctx); }

			void Execute(NodeContext& ctx) override { value.Execute(ctx); }

			const Graph* InnerGraphIfAny() const override {
				if constexpr (HasInnerGraph<T>) {
					return &value.InnerGraph();
				} else {
					return nullptr;
				}
			}

			Graph* InnerGraphIfAny() override {
				if constexpr (HasInnerGraph<T>) {
					// Resolves to Subgraph::InnerGraph()'s non-const overload since value is
					// non-const here -- HasInnerGraph only checks the const-qualified call, but
					// every type satisfying it in this codebase (just Subgraph) provides both.
					return &value.InnerGraph();
				} else {
					return nullptr;
				}
			}
		};

		std::unique_ptr<IErased> m_impl;
		NodeDescriptor           m_desc;

		NodeHandle(std::unique_ptr<IErased> impl, NodeDescriptor desc): m_impl(std::move(impl)), m_desc(desc) {}

	public:
		template <NodeLike T, typename... Args>
		static NodeHandle Make(Args&&... args) {
			return NodeHandle(
				std::make_unique<Erased<T>>(std::forward<Args>(args)...),
				NodeDescriptor{
					TypeName<T>(),
					IdsOf<ConsumesOf<T>>(),
					IdsOf<ProducesOf<T>>(),
					NodeKindOf<T>,
					PhaseOf<T>,
				}
			);
		}

		Recipe Setup(const FrameContext& ctx) { return m_impl->Setup(ctx); }

		void Execute(NodeContext& ctx) { m_impl->Execute(ctx); }

		[[nodiscard]] const NodeDescriptor& Descriptor() const { return m_desc; }

		[[nodiscard]] const Graph* InnerGraphIfAny() const { return m_impl->InnerGraphIfAny(); }

		[[nodiscard]] Graph* InnerGraphIfAny() { return m_impl->InnerGraphIfAny(); }
	};

	// An externally-registered resource (e.g. the swapchain) is just a node that produces
	// keys and consumes nothing. This removes the need for a separate
	// InjectExternalResource<K>(Resource*) registration path -- the validator has no notion
	// of "external", only "produced by something in the set". PreviousFrame (Frame.hpp) is a
	// specialized Import.
	template <ResourceRef... Ks>
	struct Import {
		using Resources = ResourceInterface<TypeList<>, TypeList<Ks...>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Host}; }

		void Execute(NodeContext&) {}
	};

	template <ResourceRef... Ks>
	struct NodeKindOfT<Import<Ks...>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

} // namespace brassica::graph
