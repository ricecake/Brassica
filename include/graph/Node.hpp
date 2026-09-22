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

	// Forward-declared only, same reason as HistoryTargetIdOf's forward inversions elsewhere in
	// this seam: the real definition lives in PhysicalResource.hpp (Vulkan-aware), resolved at
	// link time -- this file itself stays Vulkan-free.
	ResourceDesc StagedStorageBufferDesc(std::uint64_t byteSize);

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

		template <NodeLike T>
		struct ErasedRef final: IErased {
			T* ptr;

			explicit ErasedRef(T& ref): ptr(&ref) {}

			Recipe Setup(const FrameContext& ctx) override { return ptr->Setup(ctx); }

			void Execute(NodeContext& ctx) override { ptr->Execute(ctx); }

			const Graph* InnerGraphIfAny() const override {
				if constexpr (HasInnerGraph<T>) {
					return &ptr->InnerGraph();
				} else {
					return nullptr;
				}
			}

			Graph* InnerGraphIfAny() override {
				if constexpr (HasInnerGraph<T>) {
					return &ptr->InnerGraph();
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

		template <NodeLike T>
		static NodeHandle MakeRef(T& node) {
			return NodeHandle(
				std::make_unique<ErasedRef<T>>(node),
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

	// One CPU-write primitive for both buffers and textures -- ctx.WriteSpan<Key> (Execution.hpp)
	// dispatches on whatever ResourceServices::BeginHostWrite resolves Key to (a buffer or a
	// texture; see PhysicalRegistry.hpp), so this type never needs to know which. Cadence is just
	// everyFrame/dirty on Recipe::isActive: one-shot (dirty starts true, clears after the first
	// write), write-on-change (SetData sets dirty again), and every-frame are the same Setup/
	// Execute pair, not three separate mechanisms. domain is Transfer, not Host: a real write
	// here goes through BeginHostWrite/EndHostWrite's Staged or Mapped path, either of which
	// records/flushes correctly under Transfer -- Host is reserved for a node that bypasses this
	// primitive and writes a mapped pointer directly outside any Write<K>/WriteSpan<K> call (see
	// ResourceState.hpp's DeriveBufferState Host branch).
	//
	// No vk:: types anywhere here (unlike this file's old separate PredefinedTextureNode, which
	// carried a vk::ImageLayout targetLayout) -- EndHostWrite's real Staged-texture path always
	// transitions through whatever layout the barrier machinery already derived for this
	// resource's next access, so nothing here needs to name or track a target layout itself.
	// Phase defaults to Phase::Default (0), same as ever -- but a caller whose early-phase reader
	// needs this data upfront (e.g. a SubPhase::Prepare node Read<>-ing a predefined lookup table)
	// can now say so directly instead of writing a one-off subclass just to add a kPhase member.
	// This node produces its data unconditionally on the first frame (or every frame, or on
	// SetData) regardless of P -- P only affects *when in the frame* that write is scheduled
	// relative to other nodes, never whether it happens.
	template <ResourceRef Key, typename T = std::uint8_t, Phase P = Phase::Default>
	struct HostWriteNode {
		using Resources = Declares<Create<Key>>;
		static constexpr Phase kPhase = P;

		std::vector<T> data;
		ResourceDesc   desc{};
		bool           everyFrame = false;
		bool           dirty = true;

		HostWriteNode() = default;

		HostWriteNode(std::span<const T> initialData, const ResourceDesc& customDesc, bool writeEveryFrame = false):
			data(initialData.begin(), initialData.end()), desc(customDesc), everyFrame(writeEveryFrame) {}

		void SetData(std::span<const T> newData, const ResourceDesc& newDesc) {
			data.assign(newData.begin(), newData.end());
			desc = newDesc;
			dirty = true;
		}

		Recipe Setup(const FrameContext&) {
			return Recipe{
				.domain = ExecutionDomain::Transfer,
				.isActive = everyFrame || dirty,
				.realizations = {ResourceRealization{
					.key = IdOf<Key>(),
					.access = AccessKind::Write,
					.desc = desc,
				}}
			};
		}

		// No isActive/dirty guard needed here: Graph::Compile culls a node whose Recipe came back
		// !isActive before scheduling, so Execute is provably only ever called when this frame's
		// Setup already said there's something to write.
		void Execute(NodeContext& ctx) {
			ctx.WriteSpan<Key>(std::span<const T>(data));
			dirty = false;
		}

		void Reset() { dirty = true; }
	};

	template <ResourceRef Key, typename T, Phase P>
	struct NodeKindOfT<HostWriteNode<Key, T, P>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

	// Thin wrapper: infers a Staged storage-buffer desc from data.size() when the caller doesn't
	// supply one, the convenience the bare HostWriteNode deliberately doesn't have (a byteSize of
	// 0 is also a legitimate image desc's steady state, so that inference can't live there
	// without risking silently overwriting a caller's real Image2D/3D desc).
	template <ResourceRef Key, typename T = std::uint8_t, Phase P = Phase::Default>
	struct PredefinedBufferNode: HostWriteNode<Key, T, P> {
		using Base = HostWriteNode<Key, T, P>;

		PredefinedBufferNode() = default;

		explicit PredefinedBufferNode(std::span<const T> initialData, const ResourceDesc& customDesc = {}):
			Base(
				initialData,
				customDesc.byteSize ? customDesc : StagedStorageBufferDesc(initialData.size() * sizeof(T))
			) {}

		void SetData(std::span<const T> newData, const ResourceDesc& newDesc = {}) {
			Base::SetData(newData, newDesc.byteSize ? newDesc : StagedStorageBufferDesc(newData.size() * sizeof(T)));
		}
	};

	template <ResourceRef Key, typename T, Phase P>
	struct NodeKindOfT<PredefinedBufferNode<Key, T, P>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

	// Thin wrapper, named for symmetry with PredefinedBufferNode above -- an image desc always
	// needs real width/height from the caller, so there is no default-desc inference to add here;
	// this exists purely to spell out "this HostWriteNode<Key, uint8_t> is pixel data" at the call
	// site.
	template <ResourceRef Key, Phase P = Phase::Default>
	struct PredefinedTextureNode: HostWriteNode<Key, std::uint8_t, P> {
		using Base = HostWriteNode<Key, std::uint8_t, P>;

		PredefinedTextureNode() = default;

		PredefinedTextureNode(std::span<const std::uint8_t> pixelData, const ResourceDesc& imageDesc):
			Base(pixelData, imageDesc) {}
	};

	template <ResourceRef Key, Phase P>
	struct NodeKindOfT<PredefinedTextureNode<Key, P>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

	template <NodeLike T>
	struct ExecuteOnce {
		using Resources = typename T::Resources;

		T    inner{};
		bool executed{false};

		ExecuteOnce() = default;

		explicit ExecuteOnce(T n): inner(std::move(n)) {}

		Recipe Setup(const FrameContext& ctx) {
			if (executed) {
				return Recipe{.domain = ExecutionDomain::Host, .isActive = false};
			}
			return inner.Setup(ctx);
		}

		void Execute(NodeContext& ctx) {
			if (executed)
				return;
			inner.Execute(ctx);

			if (ctx.resources) {
				ctx.resources->WaitIdle();
			}
			executed = true;
		}

		void Reset() { executed = false; }
	};

	template <NodeLike T>
	struct NodeKindOfT<ExecuteOnce<T>> {
		static constexpr NodeKind value = NodeKindOf<T>;
	};

} // namespace brassica::graph
