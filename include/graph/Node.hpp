#pragma once
#include <concepts>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "graph/Execution.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/SetupBuilder.hpp"

namespace brassica::graph {

	// Inline definition for Vulkan-free graph nodes (e.g. PredefinedBufferNode).
	inline ResourceDesc StagedStorageBufferDesc(std::uint64_t byteSize) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Buffer,
			.usageMask = 0x22u, // eStorageBuffer | eTransferDst
			.byteSize = byteSize,
			.hostAccess = HostAccess::Staged,
		};
	}

	// A node declares its resource contract at runtime by calling SetupBuilder's methods inside
	// Setup (the design plan's §B) -- Setup evaluates render-context state (resolution, etc.)
	// into both the declared contract and a concrete Recipe for this frame, in one call; Execute
	// records commands via NodeContext (pipeline/globalSet/bindless access). Every node in the
	// codebase is this one shape now -- the earlier compile-time-Resources (NodeLike) shape this
	// was ported off of is gone entirely, see the design plan's §S6.
	template <typename T>
	concept BuilderNode = requires(T t, SetupBuilder& builder, const FrameContext& ctx, NodeContext& nctx) {
		{ t.Setup(builder, ctx) } -> std::same_as<void>;
		{ t.Execute(nctx) };
	};

	// What kind of special role (if any) a node plays, captured at NodeHandle::Make<T> time
	// from compile-time knowledge of T. Used by Dot.hpp to style/route rendering without
	// string-matching TypeName() output. Ordinary is the default for any type that doesn't
	// specialize NodeKindOfT; Import is specialized below, PreviousFrame/NextFrame are
	// specialized in Frame.hpp where those types are defined.
	enum class NodeKind : std::uint8_t { Ordinary, Import, PreviousFrame, NextFrame };

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

	// Type-erased holder for any BuilderNode type. Owns the node; the runtime Graph holds these.
	class NodeHandle {
		struct IErased {
			virtual ~IErased() = default;
			virtual Recipe Setup(const FrameContext&) = 0;
			virtual void   Execute(NodeContext&) = 0;

			// Relations are only known after this frame's Setup() has run -- see Erased's
			// m_relations.
			[[nodiscard]] virtual std::span<const NodeRelation> Relations() const { return {}; }

			// Computed fresh on every call, not cached once at construction: a builder node's
			// consumes/produces are only known after Setup has run for this frame, so Erased
			// below must build this from per-instance state Setup just populated, not from
			// anything static.
			virtual NodeDescriptor Descriptor() const = 0;
		};

		// Splits an already-run SetupBuilder's recorded accesses into the two owned id vectors
		// Descriptor() hands out as spans, and the Recipe Setup() returns -- shared by Erased
		// and ErasedRef below so the split logic can't drift between the owning and the
		// referencing case.
		static Recipe LowerBuilder(
			SetupBuilder&              builder,
			std::vector<ResourceId>&   consumes,
			std::vector<ResourceId>&   produces,
			std::vector<NodeRelation>& relations
		) {
			consumes.clear();
			produces.clear();
			Recipe recipe{.domain = builder.Domain(), .isActive = builder.Active()};
			for (const ResourceRealization& r : builder.Realizations()) {
				if (r.access == AccessKind::Read || r.access == AccessKind::ReadWrite) {
					consumes.push_back(r.key);
				}
				if (r.access == AccessKind::Write || r.access == AccessKind::ReadWrite) {
					produces.push_back(r.key);
				}
				recipe.realizations.push_back(r);
			}
			relations.assign(builder.Relations().begin(), builder.Relations().end());
			return recipe;
		}

		// A builder node has no compile-time Resources, so consumes/produces can't be captured
		// once at construction -- they're only known after Setup has actually run for this
		// frame. m_consumes/m_produces are therefore refreshed inside Setup() every frame, and
		// Descriptor() just hands back spans into whichever instance of them Setup() last
		// populated. Safe across the gap between Graph::Setup (which calls this Setup) and
		// Graph::Compile (which calls Descriptor()) because both run against the same
		// NodeHandle within one frame, and nothing resizes these vectors in between.
		template <BuilderNode T>
		struct Erased final: IErased {
			T                         value;
			std::vector<ResourceId>   m_consumes;
			std::vector<ResourceId>   m_produces;
			std::vector<NodeRelation> m_relations;

			template <typename... Args>
			explicit Erased(Args&&... args): value(std::forward<Args>(args)...) {}

			Recipe Setup(const FrameContext& ctx) override {
				SetupBuilder builder;
				value.Setup(builder, ctx);
				return LowerBuilder(builder, m_consumes, m_produces, m_relations);
			}

			void Execute(NodeContext& ctx) override { value.Execute(ctx); }

			NodeDescriptor Descriptor() const override {
				return NodeDescriptor{TypeName<T>(), m_consumes, m_produces, NodeKindOf<T>, PhaseOf<T>};
			}

			std::span<const NodeRelation> Relations() const override { return m_relations; }
		};

		template <BuilderNode T>
		struct ErasedRef final: IErased {
			T*                        ptr;
			std::vector<ResourceId>   m_consumes;
			std::vector<ResourceId>   m_produces;
			std::vector<NodeRelation> m_relations;

			explicit ErasedRef(T& ref): ptr(&ref) {}

			Recipe Setup(const FrameContext& ctx) override {
				SetupBuilder builder;
				ptr->Setup(builder, ctx);
				return LowerBuilder(builder, m_consumes, m_produces, m_relations);
			}

			void Execute(NodeContext& ctx) override { ptr->Execute(ctx); }

			NodeDescriptor Descriptor() const override {
				return NodeDescriptor{TypeName<T>(), m_consumes, m_produces, NodeKindOf<T>, PhaseOf<T>};
			}

			std::span<const NodeRelation> Relations() const override { return m_relations; }
		};

		std::unique_ptr<IErased> m_impl;

		explicit NodeHandle(std::unique_ptr<IErased> impl): m_impl(std::move(impl)) {}

	public:
		template <BuilderNode T, typename... Args>
		static NodeHandle Make(Args&&... args) {
			return NodeHandle(std::make_unique<Erased<T>>(std::forward<Args>(args)...));
		}

		template <BuilderNode T>
		static NodeHandle MakeRef(T& node) {
			return NodeHandle(std::make_unique<ErasedRef<T>>(node));
		}

		Recipe Setup(const FrameContext& ctx) { return m_impl->Setup(ctx); }

		void Execute(NodeContext& ctx) { m_impl->Execute(ctx); }

		[[nodiscard]] NodeDescriptor Descriptor() const { return m_impl->Descriptor(); }

		// Valid under the same same-frame-window caveat as Descriptor() -- call after
		// Setup(ctx), before this NodeHandle's next Setup() call.
		[[nodiscard]] std::span<const NodeRelation> Relations() const { return m_impl->Relations(); }
	};

	// An externally-registered resource (e.g. the swapchain) is just a node that produces
	// keys and consumes nothing. This removes the need for a separate
	// InjectExternalResource<K>(Resource*) registration path -- the validator has no notion
	// of "external", only "produced by something in the set". PreviousFrame (Frame.hpp) is a
	// specialized Import.
	template <ResourceRef... Ks>
	struct Import {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Host);
			(r.Create<Ks>(), ...);
		}

		void Execute(NodeContext&) {}
	};

	template <ResourceRef... Ks>
	struct NodeKindOfT<Import<Ks...>> {
		static constexpr NodeKind value = NodeKind::Import;
	};

	template <ResourceRef... Ks>
	struct PhaseOfT<Import<Ks...>> {
		static constexpr Phase value = Phase::PreviousFrame;
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

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Transfer);
			r.Active(everyFrame || dirty);
			r.Create<Key>().Desc(desc);
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

	template <BuilderNode T>
	struct ExecuteOnce {
		T    inner{};
		bool executed{false};

		ExecuteOnce() = default;

		explicit ExecuteOnce(T n): inner(std::move(n)) {}

		void Setup(SetupBuilder& r, const FrameContext& ctx) {
			if (executed) {
				r.Domain(ExecutionDomain::Host);
				r.Active(false);
				return;
			}
			inner.Setup(r, ctx);
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

	template <BuilderNode T>
	struct NodeKindOfT<ExecuteOnce<T>> {
		static constexpr NodeKind value = NodeKindOf<T>;
	};

} // namespace brassica::graph
