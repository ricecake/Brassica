// Vulkan-free: builds and runs via `make graph-test` / `make graph-check`, bypassing CMake
// entirely (the root project can't even configure without Vulkan -- see CMakeLists.txt:53).
// Nothing included from here may pull in vulkan/vulkan.hpp, VMA, or external/FrameGraph.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "graph/Declaration.hpp"
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

TEST_CASE("Graph executes nodes in dependency order regardless of registration order") {
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
	auto compiled = graph.Compile();
	REQUIRE(compiled.has_value());

	auto schedule = graph.Schedule();

	// Registration indices: LightingPass=0, TonemapPass=1, GtaoPass=2, GBufferPass=3,
	// Import=4, PreviousFrame=5.
	auto positionOf = [&](std::size_t nodeIndex) -> std::ptrdiff_t {
		auto it = std::find(schedule.begin(), schedule.end(), nodeIndex);
		return it == schedule.end() ? -1 : std::distance(schedule.begin(), it);
	};

	REQUIRE(positionOf(3) >= 0); // GBufferPass
	REQUIRE(positionOf(2) >= 0); // GtaoPass
	REQUIRE(positionOf(0) >= 0); // LightingPass
	REQUIRE(positionOf(1) >= 0); // TonemapPass

	CHECK(positionOf(3) < positionOf(0)); // GBufferPass before LightingPass
	CHECK(positionOf(2) < positionOf(0)); // GtaoPass before LightingPass
	CHECK(positionOf(0) < positionOf(1)); // LightingPass before TonemapPass
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
