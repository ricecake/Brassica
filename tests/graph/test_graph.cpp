// Vulkan-free: builds and runs via `make graph-test` / `make graph-check`, bypassing CMake
// entirely (the root project can't even configure without Vulkan -- see CMakeLists.txt:53).
// Nothing included from here may pull in vulkan/vulkan.hpp, VMA, or external/FrameGraph.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "graph/Declaration.hpp"
#include "graph/Dot.hpp"
#include "graph/Execution.hpp"
#include "graph/Frame.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"
#include "graph/Validation.hpp"

using namespace brassica::graph;

namespace {

	// -- example resource keys -------------------------------------------------------

	struct Swapchain {};

	struct GBufferAlbedo {};

	struct GBufferNormal {};

	struct GTAOData {};

	struct HdrColor {};

	// -- example passes ---------------------------------------------------------------

	struct GBufferPass {
		using Resources = Declares<Create<GBufferAlbedo>, Create<GBufferNormal>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct GtaoPass {
		using Resources = Declares<Read<GBufferNormal>, Read<History<GTAOData>>, Create<GTAOData>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }

		void Execute(NodeContext&) {}
	};

	struct LightingPass {
		using Resources = Declares<Read<GBufferAlbedo>, Read<GTAOData>, Create<HdrColor>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct TonemapPass {
		using Resources = Declares<Transform<HdrColor, Swapchain>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	// Two nodes with disjoint resources on different domains -- nothing connects them, so a
	// Graph containing only these two should place them in the same stage with zero barriers.
	struct KeyP {};

	struct KeyQ {};

	struct NodeP {
		using Resources = Declares<Create<KeyP>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct NodeQ {
		using Resources = Declares<Create<KeyQ>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }

		void Execute(NodeContext&) {}
	};

	using Temporal = TypeList<GTAOData>;

	// -- example passes: node phases and versioned write chains ----------------------

	// Mirrors the water-pass regression exactly: two nodes plain-Modify the same key (no version
	// number, so both consume+produce bare Swapchain) with a phase difference the only thing
	// establishing order between them.
	struct DeferredLikeNode {
		using Resources = Declares<Modify<Swapchain>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct WaterLikeNode {
		static constexpr Phase kPhase = Phase::Late;
		using Resources = Declares<Modify<Swapchain>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct StandaloneKey {};

	// No resource dependency on anything -- must still schedule after an earlier phase purely
	// because Phase says so.
	struct StandaloneLateNode {
		static constexpr Phase kPhase = Phase::Late;
		using Resources = Declares<Create<StandaloneKey>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct GtaoLikeProducer {
		using Resources = Declares<Create<GTAOData>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }

		void Execute(NodeContext&) {}
	};

	// Phase violation: the only producer of ViolationKey runs in a later phase than the reader
	// that needs it -- impossible to satisfy no matter how the schedule is arranged.
	struct ViolationKey {};

	struct LateProducer {
		static constexpr Phase kPhase = Phase::Late;
		using Resources = Declares<Create<ViolationKey>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct EarlyReader {
		static constexpr Phase kPhase = Phase::Early;
		using Resources = Declares<Read<ViolationKey>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	// Version lifting: two Default-phase nodes chain-write Swapchain (version 1, then version 2);
	// a Late-phase node declares a plain Read<Swapchain> (version 0) and must lift to version 2
	// rather than binding against the stale import.
	struct FirstWriter {
		using Resources = Declares<Modify<Swapchain, 1>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct SecondWriter {
		using Resources = Declares<Modify<Swapchain, 2>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct LatePlainReader {
		static constexpr Phase kPhase = Phase::Late;
		using Resources = Declares<Read<Swapchain>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	// Same-phase cycle: each node both reads what the other creates, with no phase difference to
	// break the tie. Compile() must report this rather than silently omit both from every stage
	// (or, when they are the only two nodes registered, index an empty stages vector).
	struct CycleKeyA {};

	struct CycleKeyB {};

	struct CycleNodeA {
		using Resources = Declares<Read<CycleKeyB>, Create<CycleKeyA>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	struct CycleNodeB {
		using Resources = Declares<Read<CycleKeyA>, Create<CycleKeyB>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(NodeContext&) {}
	};

	// -- compile-time checks: TypeList algebra ---------------------------------------

	static_assert(std::is_same_v<Concat<TypeList<int>, TypeList<char, bool>>, TypeList<int, char, bool>>);
	static_assert(std::is_same_v<Dedup<TypeList<int, char, int>>, TypeList<int, char>>);
	static_assert(std::is_same_v<Difference<TypeList<int, char, bool>, TypeList<char>>, TypeList<int, bool>>);
	static_assert(IsSubsetOf<TypeList<char>, TypeList<int, char>>);
	static_assert(!IsSubsetOf<TypeList<char, double>, TypeList<int, char>>);

	// -- compile-time checks: resource keys ------------------------------------------

	static_assert(ResourceKey<GBufferAlbedo> && !ResourceKey<int>);
	static_assert(ResourceRef<History<GTAOData>>);
	static_assert(IsHistory<History<GTAOData>> && !IsHistory<GTAOData>);
	static_assert(std::is_same_v<HistoryTarget<History<GTAOData>>, GTAOData>);
	static_assert(IdOf<GBufferAlbedo>() != IdOf<GBufferNormal>());

	// Runtime unwrap of History<K> -> K's id, used by Dot.hpp to draw the temporal edge from
	// only a pair of ResourceIds (no compile-time type available at render time).
	static_assert(kResourceTypeInfo<History<GTAOData>>.historyTarget == IdOf<GTAOData>());
	static_assert(kResourceTypeInfo<GTAOData>.historyTarget == nullptr);

	// -- compile-time checks: versioned resource keys --------------------------------

	// VersionOf<K, 0> collapses to K itself -- version 0 is whatever Create<K>/an import already
	// produces, not a distinct type. This is what keeps Modify<K> (no version) and Import<K>
	// working untouched.
	static_assert(std::is_same_v<VersionOf<Swapchain, 0>, Swapchain>);
	static_assert(std::is_same_v<VersionOf<Swapchain, 1>, VersionedKey<Swapchain, 1>>);
	static_assert(ResourceRef<VersionedKey<Swapchain, 1>>);
	static_assert(IsVersioned<VersionedKey<Swapchain, 1>> && !IsVersioned<Swapchain>);
	static_assert(std::is_same_v<VersionBase<VersionedKey<Swapchain, 1>>, Swapchain>);
	static_assert(VersionIndexOf<VersionedKey<Swapchain, 2>> == 2 && VersionIndexOf<Swapchain> == 0);
	static_assert(IdOf<VersionedKey<Swapchain, 1>>() != IdOf<VersionedKey<Swapchain, 2>>());

	// Runtime unwrap, the same shape as historyTarget above -- this is what lets
	// PhysicalResourceRegistry::ResolveId (PhysicalRegistry.hpp) map every version back to one
	// physical resource from the id alone, with no per-node registration.
	static_assert(kResourceTypeInfo<VersionedKey<Swapchain, 1>>.versionBase == IdOf<Swapchain>());
	static_assert(kResourceTypeInfo<VersionedKey<Swapchain, 1>>.version == 1);
	static_assert(kResourceTypeInfo<Swapchain>.versionBase == nullptr);
	static_assert(kResourceTypeInfo<Swapchain>.version == 0);

	// -- compile-time checks: declarations -------------------------------------------

	static_assert(std::is_same_v<ConsumesOf<GtaoPass>, TypeList<GBufferNormal, History<GTAOData>>>);
	static_assert(std::is_same_v<ProducesOf<GBufferPass>, TypeList<GBufferAlbedo, GBufferNormal>>);
	static_assert(std::is_same_v<ConsumesOf<TonemapPass>, TypeList<HdrColor>>);
	static_assert(std::is_same_v<ProducesOf<TonemapPass>, TypeList<Swapchain>>);

	// Modify<K> (no version) is unchanged: still a plain self-edge on the bare key.
	static_assert(std::is_same_v<ConsumesOf<DeferredLikeNode>, TypeList<Swapchain>>);
	static_assert(std::is_same_v<ProducesOf<DeferredLikeNode>, TypeList<Swapchain>>);

	// Modify<K, N> for N >= 1: consumes version N-1 (version 0 == bare K), produces version N.
	static_assert(std::is_same_v<ConsumesOf<FirstWriter>, TypeList<Swapchain>>);
	static_assert(std::is_same_v<ProducesOf<FirstWriter>, TypeList<VersionedKey<Swapchain, 1>>>);
	static_assert(std::is_same_v<ConsumesOf<SecondWriter>, TypeList<VersionedKey<Swapchain, 1>>>);
	static_assert(std::is_same_v<ProducesOf<SecondWriter>, TypeList<VersionedKey<Swapchain, 2>>>);

	static_assert(NodeLike<GBufferPass>);
	static_assert(NodeLike<GtaoPass>);
	static_assert(NodeLike<Import<Swapchain>>);
	static_assert(NodeLike<PreviousFrame<Temporal>>);
	static_assert(NodeLike<NextFrame<Temporal>>);

	// -- compile-time checks: node phases ---------------------------------------------

	// Default for any node that doesn't opt in, including ones that predate Phase entirely.
	static_assert(PhaseOf<GBufferPass> == Phase::Default);
	static_assert(PhaseOf<DeferredLikeNode> == Phase::Default);
	// Opts in via `static constexpr graph::Phase kPhase = ...;` rather than specializing PhaseOfT.
	static_assert(PhaseOf<WaterLikeNode> == Phase::Late);
	static_assert(PhaseOf<EarlyReader> == Phase::Early);
	// PreviousFrame/NextFrame bracket every user-defined phase from below/above unconditionally --
	// specialized in Frame.hpp, not opted in via kPhase.
	static_assert(PhaseOf<PreviousFrame<Temporal>> == Phase::PreviousFrame);
	static_assert(PhaseOf<NextFrame<Temporal>> == Phase::NextFrame);
	static_assert(Phase::PreviousFrame < Phase::Early && Phase::Late < Phase::NextFrame);

	// -- compile-time checks: temporal invariant -------------------------------------

	static_assert(TemporalInvariantHolds<PreviousFrame<Temporal>, NextFrame<Temporal>>);
	static_assert(!TemporalInvariantHolds<PreviousFrame<Temporal>, NextFrame<TypeList<HdrColor>>>);

	// -- compile-time checks: renderability, positive and negative -------------------

	using GoodFrame =
		FrameSpec<Import<Swapchain>, PreviousFrame<Temporal>, GBufferPass, GtaoPass, LightingPass, TonemapPass>;
	static_assert(IsRenderable<GoodFrame>);

	// LightingPass reads GTAOData and GtaoPass reads History<GTAOData> -- with no
	// PreviousFrame node, nothing produces History<GTAOData>.
	using BrokenFrame =
		FrameSpec<Import<Swapchain>, GBufferPass, GtaoPass, LightingPass, TonemapPass, NextFrame<Temporal>>;
	static_assert(!IsRenderable<BrokenFrame>);
	static_assert(std::is_same_v<MissingOf<BrokenFrame>, TypeList<History<GTAOData>>>);

	using CompleteFrame = FrameSpec<
		Import<Swapchain>,
		PreviousFrame<Temporal>,
		GBufferPass,
		GtaoPass,
		LightingPass,
		TonemapPass,
		NextFrame<Temporal>>;
	static_assert(IsRenderable<CompleteFrame>);

	// -- compile-time checks: subgraph composition -----------------------------------

	using InnerSpec = FrameSpec<GBufferPass, GtaoPass>;
	using Inner = Subgraph<InnerSpec>;

	static_assert(NodeLike<Inner>);
	static_assert(std::is_same_v<ConsumesOf<Inner>, TypeList<History<GTAOData>>>);
	static_assert(std::is_same_v<ProducesOf<Inner>, TypeList<GBufferAlbedo, GBufferNormal, GTAOData>>);

	using ParentSpec =
		FrameSpec<Import<Swapchain>, PreviousFrame<Temporal>, Inner, LightingPass, TonemapPass, NextFrame<Temporal>>;
	static_assert(IsRenderable<ParentSpec>);

	// -- compile-time checks: NodeKind tagging (Dot.hpp styling/recursion hooks) ----

	static_assert(NodeKindOf<GtaoPass> == NodeKind::Ordinary);
	static_assert(NodeKindOf<Import<Swapchain>> == NodeKind::Import);
	static_assert(NodeKindOf<PreviousFrame<Temporal>> == NodeKind::PreviousFrame);
	static_assert(NodeKindOf<NextFrame<Temporal>> == NodeKind::NextFrame);
	static_assert(NodeKindOf<Inner> == NodeKind::Subgraph);

	// -- root Frame: this instantiates AssertRenderable, so a broken frame here would
	// fail to compile with a diagnostic naming the missing keys.
	using RootFrame = Frame<
		Import<Swapchain>,
		PreviousFrame<Temporal>,
		GBufferPass,
		GtaoPass,
		LightingPass,
		TonemapPass,
		NextFrame<Temporal>>;
	static_assert(RootFrame::kRenderable);

	// Same node set as ParentSpec, as an actual Frame -- used by the DOT rendering test below
	// to exercise recursion into a populated Subgraph.
	using DotFrame =
		Frame<Import<Swapchain>, PreviousFrame<Temporal>, Inner, LightingPass, TonemapPass, NextFrame<Temporal>>;
	static_assert(DotFrame::kRenderable);

} // namespace

TEST_CASE("NodeHandle erasure preserves the declared resource contract") {
	NodeHandle handle = NodeHandle::Make<GtaoPass>();

	const auto& desc = handle.Descriptor();
	CHECK(desc.name.find("GtaoPass") != std::string_view::npos);
	REQUIRE(desc.consumes.size() == 2);
	CHECK(desc.produces.size() == 1);
	CHECK(desc.produces[0] == IdOf<GTAOData>());

	Recipe recipe = handle.Setup(FrameContext{});
	CHECK(recipe.domain == ExecutionDomain::Compute);
}

TEST_CASE("ValidateRuntime reports the missing key by name") {
	std::vector<NodeDescriptor> nodes{
		NodeHandle::Make<GtaoPass>().Descriptor(),
		NodeHandle::Make<LightingPass>().Descriptor(),
	};

	auto result = ValidateRuntime(nodes, {});
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("GTAOData") != std::string::npos);
}

namespace {
	// Which stage a node index landed in, or -1 if it was culled/not scheduled.
	std::ptrdiff_t StageOf(const Schedule& schedule, std::size_t nodeIndex) {
		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			const auto& nodes = schedule.stages[s].nodes;
			if (std::find(nodes.begin(), nodes.end(), nodeIndex) != nodes.end()) {
				return static_cast<std::ptrdiff_t>(s);
			}
		}
		return -1;
	}

	bool BarrierBetween(const BarrierBatch& batch, ResourceId key, ExecutionDomain src, ExecutionDomain dst) {
		for (const auto& barrier : batch.Items()) {
			if (barrier.resource == key && barrier.srcDomain == src && barrier.dstDomain == dst) {
				return true;
			}
		}
		return false;
	}
} // namespace

TEST_CASE("Graph stages nodes by dependency, independent of registration order, with correct transfer barriers") {
	Graph graph;
	// Registered out of order on purpose: LightingPass depends on both GBufferPass and
	// GtaoPass, so a graph that merely executed in registration order (like
	// external/FrameGraph today) would run it too early.
	graph.Register<LightingPass>();
	graph.Register<TonemapPass>();
	graph.Register<GtaoPass>();
	graph.Register<GBufferPass>();
	graph.Register<Import<Swapchain>>();
	graph.Register<PreviousFrame<Temporal>>(); // satisfies GtaoPass's Read<History<GTAOData>>

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();

	// Registration indices: LightingPass=0, TonemapPass=1, GtaoPass=2, GBufferPass=3,
	// Import=4, PreviousFrame=5.
	//
	// PreviousFrame carries Phase::PreviousFrame (Node.hpp/Frame.hpp), which brackets every
	// other node from below regardless of resource dependencies -- so it lands alone in stage
	// 0, one full phase group ahead of everything else, even though nothing here would have
	// forced that ordering through resource flow alone. GBufferPass/Import share no dependency
	// on each other and are both Phase::Default, so they still land in the same stage as one
	// another (just one stage later than before Phase existed): nothing forces an order between
	// the two of them, which is the parallelism proof Phase leaves untouched within one group.
	CHECK(StageOf(schedule, 5) == 0); // PreviousFrame -- its own phase, ahead of everything
	CHECK(StageOf(schedule, 3) == 1); // GBufferPass
	CHECK(StageOf(schedule, 4) == 1); // Import<Swapchain>

	CHECK(StageOf(schedule, 2) == 2); // GtaoPass: depends on GBufferPass + PreviousFrame
	CHECK(StageOf(schedule, 0) == 3); // LightingPass: depends on GBufferPass + GtaoPass
	CHECK(StageOf(schedule, 1) == 4); // TonemapPass: depends on LightingPass

	REQUIRE(schedule.stages.size() == 5);
	CHECK(schedule.stages[0].preBarriers.Empty()); // nothing precedes the first stage

	// GtaoPass (Compute) consumes GBufferNormal from GBufferPass (Graphics, same phase --
	// released at the end of stage 1, acquired at the start of stage 2) and History<GTAOData>
	// from PreviousFrame (Host, an earlier phase -- a forward cross-phase edge, kept and
	// barriered exactly like an ordinary one: released at the end of stage 0, acquired at the
	// start of stage 2).
	CHECK(BarrierBetween(
		schedule.stages[1].postBarriers,
		IdOf<GBufferNormal>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute
	));
	CHECK(BarrierBetween(
		schedule.stages[2].preBarriers,
		IdOf<GBufferNormal>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute
	));
	CHECK(BarrierBetween(
		schedule.stages[0].postBarriers,
		IdOf<History<GTAOData>>(),
		ExecutionDomain::Host,
		ExecutionDomain::Compute
	));
	CHECK(BarrierBetween(
		schedule.stages[2].preBarriers,
		IdOf<History<GTAOData>>(),
		ExecutionDomain::Host,
		ExecutionDomain::Compute
	));

	// LightingPass (Graphics) consumes GBufferAlbedo from GBufferPass (Graphics, same domain
	// -> ordinary barrier, not a transfer) and GTAOData from GtaoPass (Compute -> transfer).
	CHECK(BarrierBetween(
		schedule.stages[3].preBarriers,
		IdOf<GBufferAlbedo>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
	CHECK(BarrierBetween(
		schedule.stages[2].postBarriers,
		IdOf<GTAOData>(),
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics
	));
	CHECK(BarrierBetween(
		schedule.stages[3].preBarriers,
		IdOf<GTAOData>(),
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics
	));

	// TonemapPass (Graphics) consumes HdrColor from LightingPass (Graphics, same domain).
	CHECK(BarrierBetween(
		schedule.stages[4].preBarriers,
		IdOf<HdrColor>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
}

TEST_CASE("Two nodes plain-Modify the same key in different phases, ordered by phase with no version number") {
	// The water-pass regression, reduced to the declaration layer: before Phase existed, both
	// nodes declaring Modify<Swapchain> produced a real two-node cycle (each is simultaneously a
	// producer and a consumer of the other), and Compile() silently dropped both from every
	// stage. Phase breaks the tie without either node needing a version number.
	Graph graph;
	graph.Register<WaterLikeNode>(); // registered first, on purpose -- phase must still win
	graph.Register<DeferredLikeNode>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) == 0); // DeferredLikeNode: Phase::Default, earlier phase
	CHECK(StageOf(schedule, 0) == 1); // WaterLikeNode: Phase::Late, later phase
	REQUIRE(schedule.stages.size() == 2);
}

TEST_CASE("A node with no resource dependency on anything still schedules after an earlier phase") {
	Graph graph;
	graph.Register<StandaloneLateNode>();
	graph.Register<NodeP>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) == 0); // NodeP: Phase::Default
	CHECK(StageOf(schedule, 0) == 1); // StandaloneLateNode: Phase::Late, no edge to NodeP at all
	REQUIRE(schedule.stages.size() == 2);
}

TEST_CASE("PreviousFrame and NextFrame bracket every other node's phase") {
	Graph graph;
	graph.Register<NextFrame<Temporal>>();
	graph.Register<GtaoLikeProducer>();
	graph.Register<PreviousFrame<Temporal>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 2) == 0); // PreviousFrame: Phase::PreviousFrame, first no matter what
	CHECK(StageOf(schedule, 1) == 1); // GtaoLikeProducer: Phase::Default
	CHECK(StageOf(schedule, 0) == 2); // NextFrame: Phase::NextFrame, last no matter what
	REQUIRE(schedule.stages.size() == 3);
}

TEST_CASE(
	"Compile() reports a phase violation when only a later-phase node produces what an earlier "
	"phase needs"
) {
	// Impossible to satisfy no matter how the schedule is arranged -- Phase's ordering guarantee
	// would otherwise be silently broken (EarlyReader would run before ViolationKey exists).
	Graph graph;
	graph.Register<EarlyReader>();
	graph.Register<LateProducer>();

	graph.Setup(FrameContext{});
	auto result = graph.Compile();
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("EarlyReader") != std::string::npos);
	CHECK(result.error().message.find("LateProducer") != std::string::npos);
	CHECK(result.error().message.find("ViolationKey") != std::string::npos);
}

TEST_CASE("A later-phase plain reader lifts to the highest version an earlier phase produced") {
	Graph graph;
	graph.Register<LatePlainReader>();
	graph.Register<SecondWriter>();
	graph.Register<Import<Swapchain>>();
	graph.Register<FirstWriter>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 2) == 0); // Import<Swapchain>: produces version 0
	CHECK(StageOf(schedule, 3) == 1); // FirstWriter: consumes version 0, produces version 1
	CHECK(StageOf(schedule, 1) == 2); // SecondWriter: consumes version 1, produces version 2
	CHECK(StageOf(schedule, 0) == 3); // LatePlainReader: Phase::Late, lifted to version 2
	REQUIRE(schedule.stages.size() == 4);

	// The observable proof lifting happened: a barrier from SecondWriter's *version 2* into
	// LatePlainReader, not from Import's *version 0* -- without lifting, LatePlainReader's bare
	// Read<Swapchain> would still find a (stale) edge straight to Import, since Import really
	// does produce bare Swapchain exactly.
	CHECK(BarrierBetween(
		schedule.stages[3].preBarriers,
		IdOf<VersionedKey<Swapchain, 2>>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
	CHECK_FALSE(BarrierBetween(
		schedule.stages[3].preBarriers,
		IdOf<Swapchain>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
}

TEST_CASE("Compile() reports a cycle instead of silently dropping the nodes involved") {
	// The two nodes registered here are the *entire* graph, so this is also the
	// previously-crashing case: with no cycle detection, LevelNodes returns no levels at all,
	// m_schedule.stages stays empty, and SynthesizeBarrier would index stages[0] out of bounds.
	Graph graph;
	graph.Register<CycleNodeA>();
	graph.Register<CycleNodeB>();

	graph.Setup(FrameContext{});
	auto result = graph.Compile();
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("CycleNodeA") != std::string::npos);
	CHECK(result.error().message.find("CycleNodeB") != std::string::npos);
}

TEST_CASE(
	"A cycle inside one phase group is reported precisely, without implicating an unrelated "
	"node scheduled alongside it"
) {
	Graph graph;
	graph.Register<CycleNodeA>();
	graph.Register<NodeP>(); // same (Default) phase as the cycle, but no edge to either side of it
	graph.Register<CycleNodeB>();

	graph.Setup(FrameContext{});
	auto result = graph.Compile();
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("CycleNodeA") != std::string::npos);
	CHECK(result.error().message.find("CycleNodeB") != std::string::npos);
	CHECK(result.error().message.find("NodeP") == std::string::npos);
}

TEST_CASE("Independent nodes on different domains land in the same stage with zero barriers between them") {
	Graph graph;
	graph.Register<NodeP>();
	graph.Register<NodeQ>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	REQUIRE(schedule.stages.size() == 1);
	CHECK(schedule.stages[0].nodes.size() == 2);
	CHECK(schedule.stages[0].preBarriers.Empty());
	CHECK(schedule.stages[0].postBarriers.Empty());
}

TEST_CASE("BarrierBatch merges repeated reads and records domain transitions") {
	BarrierBatch batch;

	MemoryBarrier readA{
		.resource = IdOf<GBufferAlbedo>(),
		.access = AccessKind::Read,
		.srcStage = 0b0001,
		.dstStage = 0b0010,
		.srcDomain = ExecutionDomain::Graphics,
		.dstDomain = ExecutionDomain::Graphics,
	};
	MemoryBarrier readB = readA;
	readB.srcStage = 0b0100;
	readB.dstStage = 0b1000;

	batch.Add(readA);
	batch.Add(readB);
	REQUIRE(batch.Items().size() == 1);
	CHECK(batch.Items()[0].srcStage == 0b0101);
	CHECK(batch.Items()[0].dstStage == 0b1010);

	MemoryBarrier crossDomain = readA;
	crossDomain.dstDomain = ExecutionDomain::Compute;
	batch.Add(crossDomain);
	CHECK(batch.Items().size() == 2);
}

TEST_CASE("InnerGraphIfAny is non-null only for a Subgraph") {
	CHECK(NodeHandle::Make<GtaoPass>().InnerGraphIfAny() == nullptr);
	CHECK(NodeHandle::Make<Inner>().InnerGraphIfAny() != nullptr);
}

TEST_CASE("ToDot renders nodes, recurses into a populated subgraph, and draws the temporal edge") {
	Inner innerNode;
	innerNode.InnerGraph().Register<GBufferPass>();
	innerNode.InnerGraph().Register<GtaoPass>();

	DotFrame frame;
	frame.Register<Import<Swapchain>>();
	frame.Register<PreviousFrame<Temporal>>();
	frame.Register<Inner>(std::move(innerNode));
	frame.Register<LightingPass>();
	frame.Register<TonemapPass>();
	frame.Register<NextFrame<Temporal>>();

	frame.Setup(FrameContext{});
	REQUIRE(frame.Compile().has_value());

	std::string dot = frame.ToDot("DotFrameTest");

	CHECK(dot.find("digraph") != std::string::npos);
	CHECK(dot.find("GBufferPass") != std::string::npos); // recursed into the subgraph
	CHECK(dot.find("GtaoPass") != std::string::npos);    // recursed into the subgraph
	CHECK(dot.find("LightingPass") != std::string::npos);
	CHECK(dot.find("TonemapPass") != std::string::npos);
	CHECK(dot.find("cluster") != std::string::npos);    // the Subgraph rendered as a cluster
	CHECK(dot.find("[transfer]") != std::string::npos); // a cross-domain edge inside the subgraph
	CHECK(dot.find("gold") != std::string::npos);       // PreviousFrame/NextFrame fill color
	CHECK(dot.find("purple") != std::string::npos);     // the PreviousFrame -> NextFrame temporal edge
}
