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
#include "ArgparseManager.hpp"
#include "ConfigManager.hpp"
#include "lighting/ILightManager.hpp"
#include "lighting/LightManager.hpp"
#include "lighting/LightningManager.hpp"
#include "passes/ResourceKeys.hpp"
#include "types/ubo/LightingUBO.hpp"

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

	static_assert(std::is_same_v<
				   Declares<Group<Read, GBufferAlbedo, GBufferNormal>>::Consumes,
				   Declares<Read<GBufferAlbedo>, Read<GBufferNormal>>::Consumes>);
	static_assert(std::is_same_v<
				   Declares<Group<Read, GBufferAlbedo, GBufferNormal>>::Produces,
				   Declares<Read<GBufferAlbedo>, Read<GBufferNormal>>::Produces>);

	static_assert(std::is_same_v<
				   Declares<Group<Read, GBufferAlbedo, GBufferNormal>, Modify<Swapchain>>::Consumes,
				   Declares<Read<GBufferAlbedo>, Read<GBufferNormal>, Modify<Swapchain>>::Consumes>);
	static_assert(std::is_same_v<
				   Declares<Group<Create, GBufferAlbedo, GBufferNormal>, Create<GBufferAlbedo>>::Produces,
				   Declares<Create<GBufferAlbedo>, Create<GBufferNormal>>::Produces>);

	static_assert(ResourceKey<GBufferAlbedo> && !ResourceKey<int>);
	static_assert(ResourceRef<History<GTAOData>>);
	static_assert(IsHistory<History<GTAOData>> && !IsHistory<GTAOData>);
	static_assert(std::is_same_v<HistoryTarget<History<GTAOData>>, GTAOData>);
	static_assert(IdOf<GBufferAlbedo>() != IdOf<GBufferNormal>());

	static_assert(kResourceTypeInfo<History<GTAOData>>.historyTarget == IdOf<GTAOData>());
	static_assert(kResourceTypeInfo<GTAOData>.historyTarget == nullptr);

	static_assert(std::is_same_v<VersionOf<Swapchain, 0>, Swapchain>);
	static_assert(std::is_same_v<VersionOf<Swapchain, 1>, VersionedKey<Swapchain, 1>>);
	static_assert(ResourceRef<VersionedKey<Swapchain, 1>>);
	static_assert(IsVersioned<VersionedKey<Swapchain, 1>> && !IsVersioned<Swapchain>);
	static_assert(std::is_same_v<VersionBase<VersionedKey<Swapchain, 1>>, Swapchain>);
	static_assert(VersionIndexOf<VersionedKey<Swapchain, 2>> == 2 && VersionIndexOf<Swapchain> == 0);
	static_assert(IdOf<VersionedKey<Swapchain, 1>>() != IdOf<VersionedKey<Swapchain, 2>>());

	static_assert(kResourceTypeInfo<VersionedKey<Swapchain, 1>>.versionBase == IdOf<Swapchain>());
	static_assert(kResourceTypeInfo<VersionedKey<Swapchain, 1>>.version == 1);
	static_assert(kResourceTypeInfo<Swapchain>.versionBase == nullptr);
	static_assert(kResourceTypeInfo<Swapchain>.version == 0);

	static_assert(std::is_same_v<ConsumesOf<GtaoPass>, TypeList<GBufferNormal, History<GTAOData>>>);
	static_assert(std::is_same_v<ProducesOf<GBufferPass>, TypeList<GBufferAlbedo, GBufferNormal>>);
	static_assert(std::is_same_v<ConsumesOf<TonemapPass>, TypeList<HdrColor>>);
	static_assert(std::is_same_v<ProducesOf<TonemapPass>, TypeList<Swapchain>>);

	static_assert(std::is_same_v<ConsumesOf<DeferredLikeNode>, TypeList<Swapchain>>);
	static_assert(std::is_same_v<ProducesOf<DeferredLikeNode>, TypeList<Swapchain>>);

	static_assert(std::is_same_v<ConsumesOf<FirstWriter>, TypeList<Swapchain>>);
	static_assert(std::is_same_v<ProducesOf<FirstWriter>, TypeList<VersionedKey<Swapchain, 1>>>);
	static_assert(std::is_same_v<ConsumesOf<SecondWriter>, TypeList<VersionedKey<Swapchain, 1>>>);
	static_assert(std::is_same_v<ProducesOf<SecondWriter>, TypeList<VersionedKey<Swapchain, 2>>>);

	static_assert(NodeLike<GBufferPass>);
	static_assert(NodeLike<GtaoPass>);
	static_assert(NodeLike<Import<Swapchain>>);
	static_assert(NodeLike<PreviousFrame<Temporal>>);
	static_assert(NodeLike<NextFrame<Temporal>>);

	static_assert(PhaseOf<GBufferPass> == Phase::Default);
	static_assert(PhaseOf<DeferredLikeNode> == Phase::Default);
	static_assert(PhaseOf<WaterLikeNode> == Phase::Late);
	static_assert(PhaseOf<EarlyReader> == Phase::Early);
	static_assert(PhaseOf<PreviousFrame<Temporal>> == Phase::PreviousFrame);
	static_assert(PhaseOf<NextFrame<Temporal>> == Phase::NextFrame);
	static_assert(Phase::PreviousFrame < Phase::Early && Phase::Late < Phase::NextFrame);

	static_assert(TemporalInvariantHolds<PreviousFrame<Temporal>, NextFrame<Temporal>>);
	static_assert(!TemporalInvariantHolds<PreviousFrame<Temporal>, NextFrame<TypeList<HdrColor>>>);

	using GoodFrame =
		FrameSpec<Import<Swapchain>, PreviousFrame<Temporal>, GBufferPass, GtaoPass, LightingPass, TonemapPass>;
	static_assert(IsRenderable<GoodFrame>);

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

	using InnerSpec = FrameSpec<GBufferPass, GtaoPass>;
	using Inner = Subgraph<InnerSpec>;

	static_assert(NodeLike<Inner>);
	static_assert(std::is_same_v<ConsumesOf<Inner>, TypeList<History<GTAOData>>>);
	static_assert(std::is_same_v<ProducesOf<Inner>, TypeList<GBufferAlbedo, GBufferNormal, GTAOData>>);

	using ParentSpec =
		FrameSpec<Import<Swapchain>, PreviousFrame<Temporal>, Inner, LightingPass, TonemapPass, NextFrame<Temporal>>;
	static_assert(IsRenderable<ParentSpec>);

	static_assert(NodeKindOf<GtaoPass> == NodeKind::Ordinary);
	static_assert(NodeKindOf<Import<Swapchain>> == NodeKind::Import);
	static_assert(NodeKindOf<PreviousFrame<Temporal>> == NodeKind::PreviousFrame);
	static_assert(NodeKindOf<NextFrame<Temporal>> == NodeKind::NextFrame);
	static_assert(NodeKindOf<Inner> == NodeKind::Subgraph);

	using RootFrame = Frame<
		Import<Swapchain>,
		PreviousFrame<Temporal>,
		GBufferPass,
		GtaoPass,
		LightingPass,
		TonemapPass,
		NextFrame<Temporal>>;
	static_assert(RootFrame::kRenderable);

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
	graph.Register<LightingPass>();
	graph.Register<TonemapPass>();
	graph.Register<GtaoPass>();
	graph.Register<GBufferPass>();
	graph.Register<Import<Swapchain>>();
	graph.Register<PreviousFrame<Temporal>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();

	CHECK(StageOf(schedule, 5) == 0);
	CHECK(StageOf(schedule, 3) == 1);
	CHECK(StageOf(schedule, 4) == 1);

	CHECK(StageOf(schedule, 2) == 2);
	CHECK(StageOf(schedule, 0) == 3);
	CHECK(StageOf(schedule, 1) == 4);

	REQUIRE(schedule.stages.size() == 5);
	CHECK(schedule.stages[0].preBarriers.Empty());

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

	CHECK(BarrierBetween(
		schedule.stages[4].preBarriers,
		IdOf<HdrColor>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
}

TEST_CASE("Two nodes plain-Modify the same key in different phases, ordered by phase with no version number") {
	Graph graph;
	graph.Register<WaterLikeNode>();
	graph.Register<DeferredLikeNode>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) == 0);
	CHECK(StageOf(schedule, 0) == 1);
	REQUIRE(schedule.stages.size() == 2);
}

TEST_CASE("A node with no resource dependency on anything still schedules after an earlier phase") {
	Graph graph;
	graph.Register<StandaloneLateNode>();
	graph.Register<NodeP>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) == 0);
	CHECK(StageOf(schedule, 0) == 1);
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
	CHECK(StageOf(schedule, 2) == 0);
	CHECK(StageOf(schedule, 1) == 1);
	CHECK(StageOf(schedule, 0) == 2);
	REQUIRE(schedule.stages.size() == 3);
}

TEST_CASE(
	"Compile() reports a phase violation when only a later-phase node produces what an earlier "
	"phase needs"
) {
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
	CHECK(StageOf(schedule, 2) == 0);
	CHECK(StageOf(schedule, 3) == 1);
	CHECK(StageOf(schedule, 1) == 2);
	CHECK(StageOf(schedule, 0) == 3);
	REQUIRE(schedule.stages.size() == 4);

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
	graph.Register<NodeP>();
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
	CHECK(dot.find("GBufferPass") != std::string::npos);
	CHECK(dot.find("GtaoPass") != std::string::npos);
	CHECK(dot.find("LightingPass") != std::string::npos);
	CHECK(dot.find("TonemapPass") != std::string::npos);
	CHECK(dot.find("cluster") != std::string::npos);
	CHECK(dot.find("[transfer]") != std::string::npos);
	CHECK(dot.find("gold") != std::string::npos);
	CHECK(dot.find("purple") != std::string::npos);
}

TEST_CASE("Lighting structs alignment and size layout") {
	CHECK(sizeof(brassica::LightingUBO) == 1360);
	CHECK(sizeof(brassica::LightGPU) == 64);
	CHECK(sizeof(brassica::ClusterGPU) == 272);
	CHECK(brassica::TOTAL_CLUSTERS == 3457);
}

TEST_CASE("LightManager day/night cycle and behaviors") {
	brassica::LightManager mgr;

	CHECK(mgr.GetDayNightCycle().enabled);
	CHECK(!mgr.GetDayNightCycle().paused);
	CHECK(mgr.GetLights().size() == 2);
	CHECK(mgr.GetLights()[0].type == brassica::DIRECTIONAL_LIGHT);
	CHECK(mgr.GetLights()[1].type == brassica::DIRECTIONAL_LIGHT);
	CHECK(mgr.GetLights()[0].color.r == doctest::Approx(2.5f));
	CHECK(mgr.GetLights()[0].color.g == doctest::Approx(2.3f));
	CHECK(mgr.GetLights()[0].color.b == doctest::Approx(2.0f));

	mgr.GetDayNightCycle().time = 12.0f; // Noon (sun at zenith)
	mgr.Update(0.0f);
	CHECK(mgr.GetLights()[0].intensity == doctest::Approx(1.0f));

	brassica::Light pointLight = brassica::Light::CreatePoint(glm::vec3(10.0f, 5.0f, 0.0f), 20.0f, glm::vec3(1.0f, 0.0f, 0.0f), 50.0f);
	pointLight.SetPulse(2.0f, 1.0f);
	int pointId = mgr.AddLight(pointLight);

	CHECK(mgr.GetLights().size() == 3);
	brassica::Light* addedLight = mgr.GetLight(pointId);
	REQUIRE(addedLight != nullptr);
	CHECK(addedLight->type == brassica::POINT_LIGHT);

	mgr.Update(0.5f);

	brassica::LightingUBO ubo = mgr.GetLightingUBO();
	CHECK(ubo.numLights == 3);
	CHECK(ubo.dayTime >= 0.0f);

	brassica::LightsSSBOData ssbo = mgr.GetLightsSSBOData();
	CHECK(ssbo.count == 3);
	CHECK(ssbo.lights[2].type == brassica::POINT_LIGHT);

	brassica::Light morseLight = brassica::Light::CreatePoint(glm::vec3(0.0f), 10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	morseLight.SetMorse("SOS", 0.1f);
	int morseId = mgr.AddLight(morseLight);

	mgr.Update(0.1f);
	brassica::Light* updatedMorse = mgr.GetLight(morseId);
	REQUIRE(updatedMorse != nullptr);
	CHECK(!updatedMorse->behavior.morseSequence.empty());

	mgr.RemoveLight(pointId);
	CHECK(mgr.GetLights().size() == 3);
	mgr.RemoveLight(morseId);
	CHECK(mgr.GetLights().size() == 2);
}

TEST_CASE("LightningManager strike generation and flash light updates") {
	brassica::LightManager lightMgr;
	brassica::LightningManager lightningMgr;

	CHECK(lightningMgr.GetActiveStrikes().empty());

	lightningMgr.TriggerStrike(
		brassica::LightningType::BOLT,
		glm::vec3(0.0f, 500.0f, 0.0f),
		glm::vec3(10.0f, 0.0f, 10.0f),
		glm::vec3(0.8f, 0.9f, 1.0f),
		lightMgr
	);

	CHECK(lightningMgr.GetActiveStrikes().size() == 1);
	CHECK(lightMgr.GetLights().size() == 3);

	lightningMgr.Update(0.1f, 0.1f, lightMgr);
	CHECK(lightningMgr.GetGlobalPulse() > 0.0f);
}

namespace {
	struct ClusterLightAssignmentLikeNode {
		using Resources = Declares<Create<brassica::ClusteredLighting>>;
		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Compute}; }
		void Execute(NodeContext&) {}
	};

	struct DeferredWithLightingNode {
		using Resources = Declares<Read<brassica::ClusteredLighting>, Modify<Swapchain>>;
		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }
		void Execute(NodeContext&) {}
	};
} // namespace

TEST_CASE("Cluster light assignment node creates dependency before deferred shading pass") {
	Graph graph;
	graph.Register<DeferredWithLightingNode>();
	graph.Register<ClusterLightAssignmentLikeNode>();
	graph.Register<Import<Swapchain>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) < StageOf(schedule, 0));
}

TEST_CASE("ArgparseManager CLI options parsing") {
	brassica::ArgparseManager argMgr("TestApp", "1.0.0");
	argMgr.Initialize();
	CHECK(argMgr.IsInitialized());

	std::vector<std::string> args = {"TestApp", "--headless", "--frames", "15", "--app", "TestSandbox", "--config", "test.ini"};
	bool parseSuccess = argMgr.Parse(args);
	CHECK(parseSuccess);
	CHECK(argMgr.GetHeadless() == true);
	CHECK(argMgr.GetMaxFrames() == 15u);
	CHECK(argMgr.GetAppName() == "TestSandbox");
	CHECK(argMgr.GetConfigFile() == "test.ini");

	// Test CI headless format: --headless 10
	brassica::ArgparseManager ciArgMgr("TestApp", "1.0.0");
	std::vector<std::string> ciArgs = {"TestApp", "--headless", "10"};
	CHECK(ciArgMgr.Parse(ciArgs));
	CHECK(ciArgMgr.GetHeadless() == true);
	CHECK(ciArgMgr.GetMaxFrames() == 10u);
}

TEST_CASE("ConfigManager application and manager scoped configuration") {
	brassica::ConfigManager cfgMgr("Sandbox");
	cfgMgr.Initialize();

	cfgMgr.SetAppSetting<int>("Width", 1920);
	cfgMgr.SetAppSetting<std::string>("Title", "SandboxApp");
	cfgMgr.SetManagerSetting<float>("LightManager", "SunIntensity", 2.5f);
	cfgMgr.SetManagerSetting<bool>("LightManager", "EnableShadows", true);

	CHECK(cfgMgr.GetAppSetting<int>("Width", 1280) == 1920);
	CHECK(cfgMgr.GetAppSetting<std::string>("Title", "Default") == "SandboxApp");
	CHECK(cfgMgr.GetManagerSetting<float>("LightManager", "SunIntensity", 1.0f) == doctest::Approx(2.5f));
	CHECK(cfgMgr.GetManagerSetting<bool>("LightManager", "EnableShadows", false) == true);

	// Scoped app testing: "Editor" app should not get "Sandbox" app settings
	cfgMgr.SetApplicationName("Editor");
	CHECK(cfgMgr.GetAppSetting<int>("Width", 1280) == 1280); // Falls back to default for Editor
	CHECK(cfgMgr.GetManagerSetting<float>("LightManager", "SunIntensity", 1.0f) == doctest::Approx(1.0f)); // Scoped fallback
}

TEST_CASE("ImGuiNode toggles active state in frame graph based on ImGuiManager visibility") {
	struct MockImGuiManager {
		bool visible = false;
		bool IsVisible() const { return visible; }
	};

	MockImGuiManager mockMgr;
	mockMgr.visible = false;

	Recipe r1{.domain = ExecutionDomain::Graphics, .isActive = mockMgr.IsVisible()};
	CHECK(r1.isActive == false);

	mockMgr.visible = true;
	Recipe r2{.domain = ExecutionDomain::Graphics, .isActive = mockMgr.IsVisible()};
	if (mockMgr.IsVisible()) {
		r2.realizations.push_back(ResourceRealization{
			.key = IdOf<Swapchain>(),
			.access = AccessKind::ReadWrite
		});
	}
	CHECK(r2.isActive == true);
	REQUIRE(r2.realizations.size() == 1);
	CHECK(r2.realizations[0].key == IdOf<Swapchain>());
}

namespace {
	struct EntityDataKey {};

	struct HostEntityLogicNode {
		using Resources = Declares<Create<EntityDataKey>>;
		bool* executedFlag = nullptr;

		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Host}; }
		void Execute(NodeContext&) {
			if (executedFlag) {
				*executedFlag = true;
			}
		}
	};

	struct GpuRenderUsingEntityDataNode {
		using Resources = Declares<Read<EntityDataKey>, Modify<Swapchain>>;
		Recipe Setup(const FrameContext&) { return Recipe{.domain = ExecutionDomain::Graphics}; }
		void Execute(NodeContext&) {}
	};
} // namespace

TEST_CASE("Host queue CPU node executes entity logic before GPU consumers") {
	bool entityLogicExecuted = false;
	HostEntityLogicNode hostNode{.executedFlag = &entityLogicExecuted};

	Graph graph;
	graph.Register<GpuRenderUsingEntityDataNode>();
	graph.RegisterRef<HostEntityLogicNode>(hostNode);
	graph.Register<Import<Swapchain>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) < StageOf(schedule, 0));

	NodeContext ctx{};
	graph.Execute(ctx);
	CHECK(entityLogicExecuted == true);
}

TEST_CASE("Multi-queue domain mapping and queue family barrier propagation") {
	CHECK(DomainToLane(ExecutionDomain::Graphics) == 0);
	CHECK(DomainToLane(ExecutionDomain::Compute) == 1);
	CHECK(DomainToLane(ExecutionDomain::Transfer) == 2);
	CHECK(DomainToLane(ExecutionDomain::Host) == 3);

	CHECK(LaneToDomain(0) == ExecutionDomain::Graphics);
	CHECK(LaneToDomain(1) == ExecutionDomain::Compute);
	CHECK(LaneToDomain(2) == ExecutionDomain::Transfer);
	CHECK(LaneToDomain(3) == ExecutionDomain::Host);

	QueueSet qset;
	qset.graphics = QueueInfo{.queue = (void*)0x1, .familyIndex = 0};
	qset.compute = QueueInfo{.queue = (void*)0x2, .familyIndex = 1};
	qset.transfer = QueueInfo{.queue = (void*)0x3, .familyIndex = 2};

	Graph graph;
	graph.SetQueueSet(qset);
	CHECK(graph.QueueFamilyForDomain(ExecutionDomain::Graphics) == 0);
	CHECK(graph.QueueFamilyForDomain(ExecutionDomain::Compute) == 1);
	CHECK(graph.QueueFamilyForDomain(ExecutionDomain::Transfer) == 2);
	CHECK(graph.QueueFamilyForDomain(ExecutionDomain::Host) == kQueueFamilyIgnored);

	graph.Register<GtaoPass>(); // Compute domain
	graph.Register<GBufferPass>(); // Graphics domain
	graph.Register<PreviousFrame<Temporal>>(); // Provides History<GTAOData>

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	REQUIRE(schedule.stages.size() >= 2);

	bool foundCrossQueueBarrier = false;
	for (const auto& stage : schedule.stages) {
		for (const auto& item : stage.preBarriers.Items()) {
			if (item.srcQueueFamily == 0 && item.dstQueueFamily == 1) {
				foundCrossQueueBarrier = true;
			}
		}
		for (const auto& item : stage.postBarriers.Items()) {
			if (item.srcQueueFamily == 0 && item.dstQueueFamily == 1) {
				foundCrossQueueBarrier = true;
			}
		}
	}
	CHECK(foundCrossQueueBarrier == true);
}
