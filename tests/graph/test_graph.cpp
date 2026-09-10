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

		void Execute(CommandBuffer&) {}
	};

	struct GtaoPass {
		using Resources = Declares<Read<GBufferNormal>, Read<History<GTAOData>>, Create<GTAOData>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }

		void Execute(CommandBuffer&) {}
	};

	struct LightingPass {
		using Resources = Declares<Read<GBufferAlbedo>, Read<GTAOData>, Create<HdrColor>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(CommandBuffer&) {}
	};

	struct TonemapPass {
		using Resources = Declares<Transform<HdrColor, Swapchain>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(CommandBuffer&) {}
	};

	// Two nodes with disjoint resources on different domains -- nothing connects them, so a
	// Graph containing only these two should place them in the same stage with zero barriers.
	struct KeyP {};

	struct KeyQ {};

	struct NodeP {
		using Resources = Declares<Create<KeyP>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }

		void Execute(CommandBuffer&) {}
	};

	struct NodeQ {
		using Resources = Declares<Create<KeyQ>>;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }

		void Execute(CommandBuffer&) {}
	};

	using Temporal = TypeList<GTAOData>;

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

	// -- compile-time checks: declarations -------------------------------------------

	static_assert(std::is_same_v<ConsumesOf<GtaoPass>, TypeList<GBufferNormal, History<GTAOData>>>);
	static_assert(std::is_same_v<ProducesOf<GBufferPass>, TypeList<GBufferAlbedo, GBufferNormal>>);
	static_assert(std::is_same_v<ConsumesOf<TonemapPass>, TypeList<HdrColor>>);
	static_assert(std::is_same_v<ProducesOf<TonemapPass>, TypeList<Swapchain>>);

	static_assert(NodeLike<GBufferPass>);
	static_assert(NodeLike<GtaoPass>);
	static_assert(NodeLike<Import<Swapchain>>);
	static_assert(NodeLike<PreviousFrame<Temporal>>);
	static_assert(NodeLike<NextFrame<Temporal>>);

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
	// GBufferPass/Import/PreviousFrame have no dependencies on each other -- despite two of
	// them being on different domains (Graphics, Host) -- so they land in the same stage:
	// nothing forces an order between them, which is the whole parallelism proof.
	CHECK(StageOf(schedule, 3) == 0); // GBufferPass
	CHECK(StageOf(schedule, 4) == 0); // Import<Swapchain>
	CHECK(StageOf(schedule, 5) == 0); // PreviousFrame

	CHECK(StageOf(schedule, 2) == 1); // GtaoPass: depends on GBufferPass + PreviousFrame
	CHECK(StageOf(schedule, 0) == 2); // LightingPass: depends on GBufferPass + GtaoPass
	CHECK(StageOf(schedule, 1) == 3); // TonemapPass: depends on LightingPass

	REQUIRE(schedule.stages.size() == 4);
	CHECK(schedule.stages[0].preBarriers.Empty()); // nothing precedes the first stage

	// GtaoPass (Compute) consumes GBufferNormal from GBufferPass (Graphics) and
	// History<GTAOData> from PreviousFrame (Host) -- both cross-domain transfers, released at
	// the end of stage 0 and acquired at the start of stage 1.
	CHECK(BarrierBetween(
		schedule.stages[0].postBarriers,
		IdOf<GBufferNormal>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute
	));
	CHECK(BarrierBetween(
		schedule.stages[1].preBarriers,
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
		schedule.stages[1].preBarriers,
		IdOf<History<GTAOData>>(),
		ExecutionDomain::Host,
		ExecutionDomain::Compute
	));

	// LightingPass (Graphics) consumes GBufferAlbedo from GBufferPass (Graphics, same domain
	// -> ordinary barrier, not a transfer) and GTAOData from GtaoPass (Compute -> transfer).
	CHECK(BarrierBetween(
		schedule.stages[2].preBarriers,
		IdOf<GBufferAlbedo>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
	CHECK(BarrierBetween(
		schedule.stages[1].postBarriers,
		IdOf<GTAOData>(),
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics
	));
	CHECK(BarrierBetween(
		schedule.stages[2].preBarriers,
		IdOf<GTAOData>(),
		ExecutionDomain::Compute,
		ExecutionDomain::Graphics
	));

	// TonemapPass (Graphics) consumes HdrColor from LightingPass (Graphics, same domain).
	CHECK(BarrierBetween(
		schedule.stages[3].preBarriers,
		IdOf<HdrColor>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
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
