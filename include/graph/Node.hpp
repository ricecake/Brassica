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
	// (resolution, etc.) into a concrete Recipe for this frame, Execute records commands.
	template <typename T>
	concept NodeLike = DeclaresResources<T> && requires(T t, const FrameContext& ctx, CommandBuffer& cmd) {
		{ t.Setup(ctx) } -> std::same_as<Recipe>;
		{ t.Execute(cmd) };
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

	// A node's declared contract, lowered to runtime spans. This is what the compile-time
	// declaration looks like once type-erased -- see NodeHandle::Make below, the one place
	// where T is still known and the spans can be captured.
	struct NodeDescriptor {
		std::string_view            name;
		std::span<const ResourceId> consumes;
		std::span<const ResourceId> produces;
		NodeKind                    kind = NodeKind::Ordinary;
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
			virtual void   Execute(CommandBuffer&) = 0;

			// Non-null only for a Subgraph -- the hook that lets Dot.hpp descend into it
			// recursively without NodeHandle/Graph knowing Subgraph exists.
			virtual const Graph* InnerGraphIfAny() const { return nullptr; }
		};

		template <NodeLike T>
		struct Erased final: IErased {
			T value;

			template <typename... Args>
			explicit Erased(Args&&... args): value(std::forward<Args>(args)...) {}

			Recipe Setup(const FrameContext& ctx) override { return value.Setup(ctx); }

			void Execute(CommandBuffer& cmd) override { value.Execute(cmd); }

			const Graph* InnerGraphIfAny() const override {
				if constexpr (HasInnerGraph<T>) {
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
				NodeDescriptor{TypeName<T>(), IdsOf<ConsumesOf<T>>(), IdsOf<ProducesOf<T>>(), NodeKindOf<T>}
			);
		}

		Recipe Setup(const FrameContext& ctx) { return m_impl->Setup(ctx); }

		void Execute(CommandBuffer& cmd) { m_impl->Execute(cmd); }

		[[nodiscard]] const NodeDescriptor& Descriptor() const { return m_desc; }

		[[nodiscard]] const Graph* InnerGraphIfAny() const { return m_impl->InnerGraphIfAny(); }
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

		void Execute(CommandBuffer&) {}
	};

	template <ResourceRef... Ks>
	struct NodeKindOfT<Import<Ks...>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

} // namespace brassica::graph
