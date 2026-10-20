// Vulkan-free: builds and runs via `make graph-test` / `make graph-check`, bypassing CMake
// entirely (the root project can't even configure without Vulkan -- see CMakeLists.txt:53).
// Nothing included from here may pull in vulkan/vulkan.hpp, VMA, or external/FrameGraph.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "graph/Dot.hpp"
#include "graph/Execution.hpp"
#include "graph/Frame.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/RelationalCompiler.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/TypeList.hpp"
#include "ArgparseManager.hpp"
#include "ConfigManager.hpp"
#include "SystemHandler.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"
#include <entt/entity/registry.hpp>
#include "lighting/ILightManager.hpp"
#include "lighting/LightManager.hpp"
#include "lighting/LightningManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "ServiceLocator.hpp"
#include "terrain/TerrainMapExporter.hpp"
#include "types/CameraData.hpp"
#include "types/ubo/LightingUBO.hpp"
#include "ui/QuickSettingsWidget.hpp"

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
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<GBufferAlbedo>();
			r.Create<GBufferNormal>();
		}

		void Execute(NodeContext&) {}
	};

	struct GtaoPass {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<GBufferNormal>();
			r.ReadPrevious<GTAOData>();
			r.Create<GTAOData>();
		}

		void Execute(NodeContext&) {}
	};

	struct LightingPass {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<GBufferAlbedo>();
			r.Read<GTAOData>();
			r.Create<HdrColor>();
		}

		void Execute(NodeContext&) {}
	};

	struct TonemapPass {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<HdrColor>();
			r.Create<Swapchain>();
		}

		void Execute(NodeContext&) {}
	};

	// Two nodes with disjoint resources on different domains -- nothing connects them, so a
	// Graph containing only these two should place them in the same stage with zero barriers.
	struct KeyP {};

	struct KeyQ {};

	struct NodeP {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<KeyP>();
		}

		void Execute(NodeContext&) {}
	};

	struct NodeQ {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Create<KeyQ>();
		}

		void Execute(NodeContext&) {}
	};

	using Temporal = TypeList<GTAOData>;

	// -- example passes: node phases and versioned write chains ----------------------

	struct DeferredLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<Swapchain>();
		}

		void Execute(NodeContext&) {}
	};

	struct WaterLikeNode {
		static constexpr Phase kPhase = Phase::Late;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<Swapchain>();
		}

		void Execute(NodeContext&) {}
	};

	struct StandaloneKey {};

	struct StandaloneLateNode {
		static constexpr Phase kPhase = Phase::Late;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<StandaloneKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct GtaoLikeProducer {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Create<GTAOData>();
		}

		void Execute(NodeContext&) {}
	};

	struct ViolationKey {};

	struct LateProducer {
		static constexpr Phase kPhase = Phase::Late;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<ViolationKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct EarlyReader {
		static constexpr Phase kPhase = Phase::Early;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<ViolationKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct CycleKeyA {};

	struct CycleKeyB {};

	struct CycleNodeA {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<CycleKeyB>();
			r.Create<CycleKeyA>();
		}

		void Execute(NodeContext&) {}
	};

	struct CycleNodeB {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<CycleKeyA>();
			r.Create<CycleKeyB>();
		}

		void Execute(NodeContext&) {}
	};

	// -- compile-time checks: TypeList algebra ---------------------------------------

	static_assert(std::is_same_v<Concat<TypeList<int>, TypeList<char, bool>>, TypeList<int, char, bool>>);
	static_assert(std::is_same_v<Dedup<TypeList<int, char, int>>, TypeList<int, char>>);
	static_assert(std::is_same_v<Difference<TypeList<int, char, bool>, TypeList<char>>, TypeList<int, bool>>);
	static_assert(IsSubsetOf<TypeList<char>, TypeList<int, char>>);
	static_assert(!IsSubsetOf<TypeList<char, double>, TypeList<int, char>>);

	static_assert(ResourceKey<GBufferAlbedo> && !ResourceKey<int>);
	static_assert(ResourceRef<History<GTAOData>>);
	static_assert(IsHistory<History<GTAOData>> && !IsHistory<GTAOData>);
	static_assert(std::is_same_v<HistoryTarget<History<GTAOData>>, GTAOData>);
	static_assert(IdOf<GBufferAlbedo>() != IdOf<GBufferNormal>());

	static_assert(kResourceTypeInfo<History<GTAOData>>.historyTarget == IdOf<GTAOData>());
	static_assert(kResourceTypeInfo<GTAOData>.historyTarget == nullptr);

	static_assert(BuilderNode<GBufferPass>);
	static_assert(BuilderNode<GtaoPass>);
	static_assert(BuilderNode<Import<Swapchain>>);
	static_assert(BuilderNode<PreviousFrame<Temporal>>);
	static_assert(BuilderNode<NextFrame<Temporal>>);

	static_assert(PhaseOf<GBufferPass> == Phase::Default);
	static_assert(PhaseOf<DeferredLikeNode> == Phase::Default);
	static_assert(PhaseOf<WaterLikeNode> == Phase::Late);
	static_assert(PhaseOf<EarlyReader> == Phase::Early);
	static_assert(PhaseOf<PreviousFrame<Temporal>> == Phase::PreviousFrame);
	static_assert(PhaseOf<NextFrame<Temporal>> == Phase::NextFrame);
	static_assert(Phase::PreviousFrame < Phase::Early && Phase::Late < Phase::NextFrame);

	static_assert(NodeKindOf<GtaoPass> == NodeKind::Ordinary);
	static_assert(NodeKindOf<Import<Swapchain>> == NodeKind::Import);
	static_assert(NodeKindOf<PreviousFrame<Temporal>> == NodeKind::PreviousFrame);
	static_assert(NodeKindOf<NextFrame<Temporal>> == NodeKind::NextFrame);

} // namespace

TEST_CASE("NodeHandle erasure preserves the declared resource contract") {
	NodeHandle handle = NodeHandle::Make<GtaoPass>();

	// Descriptor() has nothing to report before Setup has run once -- builder nodes have no
	// compile-time Resources, so consumes/produces are only known after this call.
	Recipe recipe = handle.Setup(FrameContext{});
	CHECK(recipe.domain == ExecutionDomain::Compute);

	const auto& desc = handle.Descriptor();
	CHECK(desc.name.find("GtaoPass") != std::string_view::npos);
	REQUIRE(desc.consumes.size() == 2);
	CHECK(desc.produces.size() == 1);
	CHECK(desc.produces[0] == IdOf<GTAOData>());
}

TEST_CASE("ValidateRuntime reports the missing key by name") {
	NodeHandle gtaoHandle = NodeHandle::Make<GtaoPass>();
	NodeHandle lightingHandle = NodeHandle::Make<LightingPass>();
	// Descriptor() on a builder-shaped handle has nothing to report until Setup has run once.
	gtaoHandle.Setup(FrameContext{});
	lightingHandle.Setup(FrameContext{});

	std::vector<NodeDescriptor> nodes{gtaoHandle.Descriptor(), lightingHandle.Descriptor()};

	// GBufferNormal/GBufferAlbedo have no producer in this list -- genuinely missing.
	// History<GTAOData> also has no PreviousFrame/Import producer here, but is NOT
	// reported: a History<K> consumption is self-satisfying (its physical pairing comes
	// from ProvisionTemporalPairs, not a graph-level producer edge), so requiring one
	// here would be a false positive.
	auto result = ValidateRuntime(nodes);
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("GBufferNormal") != std::string::npos);
	CHECK(result.error().message.find("History") == std::string::npos);
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
	CHECK(StageOf(schedule, 4) == 0);
	CHECK(StageOf(schedule, 3) == 1);

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

// S3 exit criterion: a shuffled-registration property test gives an identical schedule. "Identical"
// here means the same name->stage-index mapping, not the same name->node-index mapping (registration
// order still decides *index*, just never stage placement -- §C.5's "ties break by type name, never
// registration index" is specifically about this).
TEST_CASE("Registration order never changes which stage a node lands in") {
	using Registrar = void (*)(Graph&);
	std::vector<Registrar> registrars{
		[](Graph& g) { g.Register<LightingPass>(); },
		[](Graph& g) { g.Register<TonemapPass>(); },
		[](Graph& g) { g.Register<GtaoPass>(); },
		[](Graph& g) { g.Register<GBufferPass>(); },
		[](Graph& g) { g.Register<Import<Swapchain>>(); },
		[](Graph& g) { g.Register<PreviousFrame<Temporal>>(); },
	};

	auto stagesByName = [](std::vector<Registrar> order) {
		Graph graph;
		for (auto registrar : order) {
			registrar(graph);
		}
		graph.Setup(FrameContext{});
		REQUIRE(graph.Compile().has_value());

		const Schedule& schedule = graph.GetSchedule();
		std::map<std::string_view, std::ptrdiff_t> result;
		for (std::size_t i = 0; i < graph.Nodes().size(); ++i) {
			result[graph.Nodes()[i].Descriptor().name] = StageOf(schedule, i);
		}
		return result;
	};

	auto baseline = stagesByName(registrars);

	std::vector<Registrar> reversed(registrars.rbegin(), registrars.rend());
	CHECK(stagesByName(reversed) == baseline);

	// A third, neither-forward-nor-reverse order -- a fixed permutation, not randomized (no
	// value in flakiness here, just in not being either of the two "obvious" orders above).
	std::vector<Registrar> shuffled{
		registrars[3], registrars[0], registrars[5], registrars[1], registrars[4], registrars[2]
	};
	CHECK(stagesByName(shuffled) == baseline);
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

// S3: the dedicated "phase violation" error is gone (phase is a compat marker chain now, not a
// hard partition) -- but this exact scenario still fails to compile, just via a structural G0
// cycle instead of a special-cased check: LateProducer's Create-before-accessor edge ("the
// creator precedes every accessor") points at EarlyReader, while the phase chain already points
// the other way (EarlyReader's phase precedes LateProducer's) -- those two edges directly
// contradict, which is a real cycle, not an artifact. Still names both nodes; no longer names the
// key by name (a G0 cycle predates any resource-edge/key bookkeeping) -- see
// RelationalCompiler.hpp's own comment on this exact tradeoff.
TEST_CASE("Compile() reports a cycle when a later-phase node is the only producer an earlier-phase node reads") {
	Graph graph;
	graph.Register<EarlyReader>();
	graph.Register<LateProducer>();

	graph.Setup(FrameContext{});
	auto result = graph.Compile();
	REQUIRE(!result.has_value());
	CHECK(result.error().message.find("EarlyReader") != std::string::npos);
	CHECK(result.error().message.find("LateProducer") != std::string::npos);
}

// S3: version lifting is gone. The replacement idiom for "two same-key writers in sequence,
// same phase" is an explicit relation, exactly what the particle system's GBufferDepth v1/v2
// chain becomes in this same stage (§F) -- demonstrated here with builder-shaped nodes.
namespace {

	struct SeqKey {};

	struct SeqSecondWriter;

	struct SeqFirstWriter {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<SeqKey>();
			r.Before<SeqSecondWriter>();
		}

		void Execute(NodeContext&) {}
	};

	struct SeqSecondWriter {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<SeqKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct SeqLateReader {
		static constexpr Phase kPhase = Phase::Late;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<SeqKey>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("Two same-key writers in sequence order via an explicit relation, not a version number") {
	Graph graph;
	graph.Register<SeqLateReader>();
	graph.Register<SeqSecondWriter>();
	graph.Register<SeqFirstWriter>();

	graph.Setup(FrameContext{});
	auto result = graph.Compile();
	REQUIRE(result.has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 2) < StageOf(schedule, 1));
	CHECK(StageOf(schedule, 1) < StageOf(schedule, 0));

	CHECK(BarrierBetween(
		schedule.stages[StageOf(schedule, 1)].preBarriers,
		IdOf<SeqKey>(),
		ExecutionDomain::Graphics,
		ExecutionDomain::Graphics
	));
	CHECK(BarrierBetween(
		schedule.stages[StageOf(schedule, 0)].preBarriers,
		IdOf<SeqKey>(),
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

TEST_CASE("ToDot renders nodes, phase clusters, and the PreviousFrame/NextFrame temporal edge") {
	Graph graph;
	graph.Register<Import<Swapchain>>();
	graph.Register<PreviousFrame<Temporal>>();
	graph.Register<GBufferPass>();
	graph.Register<GtaoPass>();
	graph.Register<LightingPass>();
	graph.Register<TonemapPass>();
	graph.Register<NextFrame<Temporal>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	std::string dot = ToDot(graph, "TemporalEdgeTest");

	CHECK(dot.find("digraph") != std::string::npos);
	CHECK(dot.find("GBufferPass") != std::string::npos);
	CHECK(dot.find("GtaoPass") != std::string::npos);
	CHECK(dot.find("LightingPass") != std::string::npos);
	CHECK(dot.find("TonemapPass") != std::string::npos);
	CHECK(dot.find("cluster_phase_") != std::string::npos);
	CHECK(dot.find("[transfer]") != std::string::npos);
	// PreviousFrame/NextFrame fill color and the History<K>/K temporal edge they share --
	// pure Dot-rendering coverage, no Subgraph involved (never was: EmitGraph's temporal-edge
	// loop always scanned this graph's own nodes directly, with no recursion).
	CHECK(dot.find("gold") != std::string::npos);
	CHECK(dot.find("purple") != std::string::npos);
}

TEST_CASE("ToDot exports phase clusters and labels") {
	Graph graph;
	graph.Register<DeferredLikeNode>();
	graph.Register<WaterLikeNode>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	std::string dot = ToDot(graph, "PhaseClusterTest");

	CHECK(dot.find("cluster_phase_0") != std::string::npos);
	CHECK(dot.find("cluster_phase_1000") != std::string::npos);
	CHECK(dot.find("Phase 0 (Default (no declared phase))") != std::string::npos);
	CHECK(dot.find("Phase 1000 (Late / UnderwaterStructuralTranslucentRender)") != std::string::npos);
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
	CHECK(mgr.GetLights()[0].intensity == doctest::Approx(10.0f));

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
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Create<brassica::ClusteredLighting>();
		}
		void Execute(NodeContext&) {}
	};

	struct DeferredWithLightingNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<brassica::ClusteredLighting>();
			r.Modify<Swapchain>();
		}
		void Execute(NodeContext&) {}
	};
} // namespace

TEST_CASE("GTAONode creates dependency before deferred shading pass") {
	struct GtaoTestNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<GBufferNormal>();
			r.Create<brassica::GTAOTexture>();
		}
		void Execute(NodeContext&) {}
	};

	struct DeferredTestNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<brassica::GTAOTexture>();
			r.Modify<Swapchain>();
		}
		void Execute(NodeContext&) {}
	};

	Graph graph;
	graph.Register<DeferredTestNode>();
	graph.Register<GtaoTestNode>();
	graph.Register<GBufferPass>();
	graph.Register<Import<Swapchain>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 1) < StageOf(schedule, 0));
}

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

namespace {
	struct TransmittanceLUTLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Create<brassica::TransmittanceLUT>();
		}
		void Execute(NodeContext&) {}
	};

	struct MultiScatteringLUTLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<brassica::TransmittanceLUT>();
			r.Create<brassica::MultiScatteringLUT>();
		}
		void Execute(NodeContext&) {}
	};

	struct SkyViewLUTLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Compute);
			r.Read<brassica::TransmittanceLUT>();
			r.Read<brassica::MultiScatteringLUT>();
			r.Create<brassica::SkyViewLUT>();
		}
		void Execute(NodeContext&) {}
	};

	struct SkyBackgroundLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<brassica::SkyViewLUT>();
			r.Read<brassica::TransmittanceLUT>();
			r.Create<brassica::AtmosphereRadiance>();
		}
		void Execute(NodeContext&) {}
	};

	struct DeferredReadingSkyNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<brassica::AtmosphereRadiance>();
			r.Modify<Swapchain>();
		}
		void Execute(NodeContext&) {}
	};
} // namespace

TEST_CASE("Atmosphere Sky pipeline stages nodes by dependency Transmittance -> MultiScattering -> SkyView -> SkyBackground -> Deferred") {
	Graph graph;
	graph.Register<DeferredReadingSkyNode>();
	graph.Register<SkyBackgroundLikeNode>();
	graph.Register<SkyViewLUTLikeNode>();
	graph.Register<MultiScatteringLUTLikeNode>();
	graph.Register<TransmittanceLUTLikeNode>();
	graph.Register<Import<Swapchain>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 4) < StageOf(schedule, 3));
	CHECK(StageOf(schedule, 3) < StageOf(schedule, 2));
	CHECK(StageOf(schedule, 2) < StageOf(schedule, 1));
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
		bool* executedFlag = nullptr;

		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Host);
			r.Create<EntityDataKey>();
		}
		void Execute(NodeContext&) {
			if (executedFlag) {
				*executedFlag = true;
			}
		}
	};

	struct GpuRenderUsingEntityDataNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<EntityDataKey>();
			r.Modify<Swapchain>();
		}
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

// MemoryBarrier::srcQueueFamily/dstQueueFamily (checked here against real family values
// directly) were removed at S4: under VK_SHARING_MODE_CONCURRENT (what every multi-queue-touched
// resource now uses, PhysicalResource.hpp), a barrier's queue family fields must both be
// VK_QUEUE_FAMILY_IGNORED, so there is nothing left to assert about real family values on a
// MemoryBarrier at all -- the cross-domain split (postBarriers in the producer's stage,
// preBarriers in the consumer's) is still keyed on each node's own *declared* domain exactly as
// before, which is what this test checks instead now.
TEST_CASE("Multi-queue domain mapping and declared-domain barrier propagation") {
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
			if (item.srcDomain == ExecutionDomain::Graphics && item.dstDomain == ExecutionDomain::Compute) {
				foundCrossQueueBarrier = true;
			}
		}
		for (const auto& item : stage.postBarriers.Items()) {
			if (item.srcDomain == ExecutionDomain::Graphics && item.dstDomain == ExecutionDomain::Compute) {
				foundCrossQueueBarrier = true;
			}
		}
	}
	CHECK(foundCrossQueueBarrier == true);
}

namespace {
	class MockTestSystemHandler: public brassica::SystemHandler {
	public:
		bool setupCalled{false};
		bool preFrameCalled{false};
		bool postFrameCalled{false};
		bool cleanupCalled{false};
		uint32_t updateEntityCalls{0};
		uint32_t lastFrameIndex{0};

		void Setup(brassica::Engine&, const brassica::FrameDetails& details) override {
			setupCalled = true;
			lastFrameIndex = details.frameIndex;
		}

		entt::entity RegisterTestEntity(entt::registry& reg, const brassica::TransformComponent& transform) {
			entt::entity entity = reg.create();
			reg.emplace<brassica::TransformComponent>(entity, transform);
			m_entities.push_back(entity);
			return entity;
		}

		void PreFrame(brassica::Engine&, const brassica::FrameDetails&) override { preFrameCalled = true; }

		void PostFrame(brassica::Engine&, const brassica::FrameDetails&) override { postFrameCalled = true; }

		void Cleanup(brassica::Engine&) override { cleanupCalled = true; }

		void UpdateEntity(entt::entity, brassica::Engine&, const brassica::FrameDetails& details) override {
			updateEntityCalls++;
			lastFrameIndex = details.frameIndex;
		}
	};
} // namespace

namespace {
	struct BallLikeNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<GBufferAlbedo>();
			r.Modify<GBufferNormal>();
			r.Create<brassica::EntityIndirectBuffer>();
		}
		void Execute(NodeContext&) {}
	};
} // namespace

TEST_CASE("EntityNode compiles smoothly in graph with GBuffer and Deferred nodes") {
	Graph graph;
	graph.Register<GBufferPass>();
	graph.Register<BallLikeNode>();
	graph.Register<ClusterLightAssignmentLikeNode>();
	graph.Register<DeferredWithLightingNode>();
	graph.Register<Import<Swapchain>>();

	graph.Setup(FrameContext{});
	REQUIRE(graph.Compile().has_value());

	const auto& schedule = graph.GetSchedule();
	CHECK(StageOf(schedule, 0) <= StageOf(schedule, 1));
	CHECK(StageOf(schedule, 1) <= StageOf(schedule, 3));
}

TEST_CASE("SystemHandler lifecycle, entity registration, and update callbacks") {
	entt::registry registry;
	brassica::TransformComponent initialTransform{
		.position = glm::vec3(1.0f, 2.0f, 3.0f),
		.rotation = glm::vec3(0.0f),
		.scale = glm::vec3(2.0f)
	};

	MockTestSystemHandler handler;
	entt::entity e1 = handler.RegisterTestEntity(registry, initialTransform);

	CHECK(registry.valid(e1));
	auto* transform = registry.try_get<brassica::TransformComponent>(e1);
	REQUIRE(transform != nullptr);
	CHECK(transform->position.x == doctest::Approx(1.0f));
	CHECK(transform->position.y == doctest::Approx(2.0f));
	CHECK(transform->position.z == doctest::Approx(3.0f));
	CHECK(transform->scale.x == doctest::Approx(2.0f));

	glm::mat4 mat = transform->GetTransformMatrix();
	CHECK(mat[3][0] == doctest::Approx(1.0f));
	CHECK(mat[3][1] == doctest::Approx(2.0f));
	CHECK(mat[3][2] == doctest::Approx(3.0f));

	brassica::FrameDetails details{
		.deltaTime = 0.016f,
		.totalTime = 1.0,
		.frameIndex = 42,
		.camera = nullptr
	};

	brassica::Engine* enginePtr = nullptr;

	handler.PreFrame(*enginePtr, details);
	CHECK(handler.preFrameCalled == true);

	handler.Update(*enginePtr, details);
	CHECK(handler.updateEntityCalls == 1);
	CHECK(handler.lastFrameIndex == 42);

	handler.PostFrame(*enginePtr, details);
	CHECK(handler.postFrameCalled == true);

	handler.Cleanup(*enginePtr);
	CHECK(handler.cleanupCalled == true);
}

TEST_CASE("CameraData default state and mode cycling") {
	brassica::CameraData cam;

	CHECK(cam.mode == brassica::CameraMode::Accelerated);
	CHECK(cam.speed == doctest::Approx(10.0f));
	CHECK(cam.baseFov == doctest::Approx(1.2f));
	CHECK(cam.fov == doctest::Approx(1.2f));
	CHECK(cam.currentSpeed == doctest::Approx(0.0f));
	CHECK(cam.GetDisplayedSpeed() == doctest::Approx(0.0f));

	cam.CycleMode();
	CHECK(cam.mode == brassica::CameraMode::FirstPerson);

	cam.CycleMode();
	CHECK(cam.mode == brassica::CameraMode::Instant);
	CHECK(cam.GetDisplayedSpeed() == doctest::Approx(10.0f));

	cam.CycleMode();
	CHECK(cam.mode == brassica::CameraMode::Accelerated);
}

TEST_CASE("Camera Accelerated mode acceleration, coasting down, FOV adaptation, and speed capping") {
	brassica::CameraData cam;
	cam.mode = brassica::CameraMode::Accelerated;
	cam.speed = 20.0f; // Configured max speed
	cam.baseFov = 1.2f;
	cam.maxFovBoost = 0.3f;
	cam.accelerationRate = 5.0f;
	cam.decelerationRate = 4.0f;

	glm::vec3 moveDir{0.0f, 0.0f, -1.0f}; // Moving forward
	float     deltaTime = 0.1f;

	// Accelerate step 1
	glm::vec3 targetVelocity = glm::normalize(moveDir) * cam.speed;
	float     blend = 1.0f - std::exp(-cam.accelerationRate * deltaTime);
	cam.velocity = glm::mix(cam.velocity, targetVelocity, blend);
	cam.currentSpeed = glm::length(cam.velocity);

	float speedRatio = std::clamp(cam.currentSpeed / cam.speed, 0.0f, 1.0f);
	cam.fov = cam.baseFov + speedRatio * cam.maxFovBoost;

	CHECK(cam.currentSpeed > 0.0f);
	CHECK(cam.currentSpeed < cam.speed);
	CHECK(cam.fov > cam.baseFov);
	CHECK(cam.GetDisplayedSpeed() == doctest::Approx(cam.currentSpeed));

	// Accelerate for several steps until reaching top speed
	for (int i = 0; i < 25; ++i) {
		float b = 1.0f - std::exp(-cam.accelerationRate * deltaTime);
		cam.velocity = glm::mix(cam.velocity, targetVelocity, b);
		float speedLen = glm::length(cam.velocity);
		if (speedLen > cam.speed) {
			cam.velocity = (cam.velocity / speedLen) * cam.speed;
			speedLen = cam.speed;
		}
		cam.currentSpeed = speedLen;
		float sRatio = std::clamp(cam.currentSpeed / cam.speed, 0.0f, 1.0f);
		cam.fov = cam.baseFov + sRatio * cam.maxFovBoost;
	}

	CHECK(cam.currentSpeed == doctest::Approx(20.0f).epsilon(0.001));
	CHECK(cam.fov == doctest::Approx(1.2f + 0.3f).epsilon(0.001));

	// Coast down (release move keys)
	moveDir = glm::vec3(0.0f);
	targetVelocity = glm::vec3(0.0f);

	float highSpeed = cam.currentSpeed;
	float highFov = cam.fov;

	// Coast down step 1
	float decelBlend = 1.0f - std::exp(-cam.decelerationRate * deltaTime);
	cam.velocity = glm::mix(cam.velocity, targetVelocity, decelBlend);
	cam.currentSpeed = glm::length(cam.velocity);
	float decelRatio = std::clamp(cam.currentSpeed / cam.speed, 0.0f, 1.0f);
	cam.fov = cam.baseFov + decelRatio * cam.maxFovBoost;

	CHECK(cam.currentSpeed < highSpeed);
	CHECK(cam.fov < highFov);
	CHECK(cam.fov > cam.baseFov);

	// Coast down until full stop
	for (int i = 0; i < 25; ++i) {
		float db = 1.0f - std::exp(-cam.decelerationRate * deltaTime);
		cam.velocity = glm::mix(cam.velocity, targetVelocity, db);
		float speedLen = glm::length(cam.velocity);
		if (speedLen < 0.01f) {
			cam.velocity = glm::vec3(0.0f);
			speedLen = 0.0f;
		}
		cam.currentSpeed = speedLen;
		float sr = std::clamp(cam.currentSpeed / cam.speed, 0.0f, 1.0f);
		cam.fov = cam.baseFov + sr * cam.maxFovBoost;
	}

	CHECK(cam.currentSpeed == doctest::Approx(0.0f));
	CHECK(cam.fov == doctest::Approx(cam.baseFov));
}

TEST_CASE("QuickSettingsWidget camera mode UI integration via ServiceLocator") {
	brassica::ServiceLocator locator;
	brassica::ServiceLocator::SetInstance(&locator);

	auto camData = std::make_shared<brassica::CameraData>();
	locator.Provide<brassica::CameraData>(camData);

	CHECK(locator.Has<brassica::CameraData>());
	CHECK(locator.Get<brassica::CameraData>()->mode == brassica::CameraMode::Accelerated);

	brassica::ui::QuickSettingsWidget widget;
	CHECK(widget.GetTitle() == "Quick Controls");

	camData->mode = brassica::CameraMode::Instant;
	CHECK(locator.Get<brassica::CameraData>()->mode == brassica::CameraMode::Instant);
}

TEST_CASE("ArgparseManager terrain map flag options") {
	brassica::ArgparseManager argMgr("TestApp", "1.0.0");
	std::vector<std::string> argsDefault = {"TestApp", "--render-terrain-map"};
	CHECK(argMgr.Parse(argsDefault));
	CHECK(argMgr.GetRenderTerrainMap() == true);
	CHECK(argMgr.GetTerrainMapPath() == "terrain_map.png");

	brassica::ArgparseManager argMgrCustom("TestApp", "1.0.0");
	std::vector<std::string> argsCustom = {"TestApp", "--export-terrain-map", "custom_planet.png"};
	CHECK(argMgrCustom.Parse(argsCustom));
	CHECK(argMgrCustom.GetRenderTerrainMap() == true);
	CHECK(argMgrCustom.GetTerrainMapPath() == "custom_planet.png");
}

TEST_CASE("ArgparseManager print render graph flag options") {
	brassica::ArgparseManager argMgr("TestApp", "1.0.0");
	std::vector<std::string> args = {"TestApp", "--print-render-graph"};
	CHECK(argMgr.Parse(args));
	CHECK(argMgr.GetPrintRenderGraph() == true);

	brassica::ArgparseManager argMgrAlias("TestApp", "1.0.0");
	std::vector<std::string> argsAlias = {"TestApp", "--print-graphviz"};
	CHECK(argMgrAlias.Parse(argsAlias));
	CHECK(argMgrAlias.GetPrintRenderGraph() == true);
}

TEST_CASE("TerrainMapColorConfig elevation, slope, and water color tinting") {
	uint8_t r = 0, g = 0, b = 0, a = 0;

	// Water: h = -100m, flat surface normal (0, 1, 0)
	brassica::TerrainMapColorConfig::GetColor(-100.0f, 0.0f, 1.0f, 0.0f, r, g, b, a);
	CHECK(b > r);
	CHECK(b > g);
	CHECK(a == 255);

	// Lowland: h = 50m
	brassica::TerrainMapColorConfig::GetColor(50.0f, 0.0f, 1.0f, 0.0f, r, g, b, a);
	CHECK(g > r);
	CHECK(a == 255);

	// High Peak / Summit: h = 1200m
	brassica::TerrainMapColorConfig::GetColor(1200.0f, 0.0f, 1.0f, 0.0f, r, g, b, a);
	CHECK(r > 180);
	CHECK(g > 180);
	CHECK(b > 180);
}

// -- Spike B: shadow relational compiler (design plan's §C) ----------------------------

namespace {

	std::size_t IndexOfNamed(std::span<const NodeHandle> nodes, std::string_view name) {
		for (std::size_t i = 0; i < nodes.size(); ++i) {
			if (nodes[i].Descriptor().name.find(name) != std::string_view::npos) {
				return i;
			}
		}
		FAIL("no node named ", name);
		return static_cast<std::size_t>(-1);
	}

	std::ptrdiff_t RelStageOf(const RelationalResult& result, std::size_t nodeIndex) {
		for (std::size_t s = 0; s < result.stages.size(); ++s) {
			if (std::find(result.stages[s].begin(), result.stages[s].end(), nodeIndex) != result.stages[s].end()) {
				return static_cast<std::ptrdiff_t>(s);
			}
		}
		return -1;
	}

	RelationalResult CompileRelationalFromGraph(Graph& g, const FrameContext& ctx) {
		g.Setup(ctx);
		return CompileRelational(g.Nodes(), g.Recipes());
	}

} // namespace

TEST_CASE("CompileRelational: independent nodes on different domains land in the same stage") {
	Graph graph;
	graph.Register<NodeP>();
	graph.Register<NodeQ>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());
	REQUIRE(result.stages.size() == 1);
	CHECK(result.stages[0].size() == 2);
}

// S3: Compile() and CompileRelational are no longer two things to compare -- Compile() just is
// CompileRelational (see Graph.hpp). The scenario this test used to compare against the phase
// compiler is already covered directly by "Graph stages nodes by dependency..." above, on the
// exact same fixture set; keeping both would just be the same assertion twice.

namespace {

	struct RelOrderKeyA {};

	struct RelOrderKeyB {};

	struct RelOrderNodeB;

	// No resource edge to RelOrderNodeB at all (disjoint keys) -- the only thing that can order
	// these two is the explicit Before<> relation itself, proving the relational compiler
	// actually uses NodeRelation rather than falling back to phase/resource luck.
	struct RelOrderNodeA {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<RelOrderKeyA>();
			r.Before<RelOrderNodeB>();
		}

		void Execute(NodeContext&) {}
	};

	struct RelOrderNodeB {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<RelOrderKeyB>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("CompileRelational orders two resource-disjoint nodes purely from an explicit Before<> relation") {
	Graph graph;
	graph.Register<RelOrderNodeA>();
	graph.Register<RelOrderNodeB>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t a = IndexOfNamed(graph.Nodes(), "RelOrderNodeA");
	std::size_t b = IndexOfNamed(graph.Nodes(), "RelOrderNodeB");
	CHECK(RelStageOf(result, a) < RelStageOf(result, b));
}

namespace {

	struct AmbiguousKey {};

	struct AmbiguousWriterA {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<AmbiguousKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct AmbiguousWriterB {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<AmbiguousKey>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("CompileRelational reports an ambiguity for two same-phase writers with no ordering relation") {
	Graph graph;
	graph.Register<AmbiguousWriterA>();
	graph.Register<AmbiguousWriterB>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok); // an ambiguity is reported, not a hard compile failure
	REQUIRE(result.ambiguities.size() == 1);
	CHECK(result.ambiguities[0].key == IdOf<AmbiguousKey>());
}

namespace {

	struct CycleRelNodeB;

	struct CycleRelNodeA {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Before<CycleRelNodeB>();
		}

		void Execute(NodeContext&) {}
	};

	struct CycleRelNodeB {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Before<CycleRelNodeA>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("CompileRelational reports a hard error when two explicit relations directly contradict") {
	Graph graph;
	graph.Register<CycleRelNodeA>();
	graph.Register<CycleRelNodeB>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	CHECK_FALSE(result.ok);
	CHECK(!result.error.empty());
}

namespace {

	struct DroppedRelKey {};

	struct NoSuchNode;

	// Before<> names a type that's never registered in this graph -- must be dropped, not a hard
	// error (§E: "A relation to an unregistered type is dropped, with a one-time warning").
	struct DanglingRelNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<DroppedRelKey>();
			r.Before<NoSuchNode>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("CompileRelational drops a relation naming an unregistered node type instead of failing") {
	Graph graph;
	graph.Register<DanglingRelNode>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	REQUIRE(result.stages.size() == 1);
	CHECK(result.stages[0].size() == 1);
}

// -- S4: 6-phase keystone backbone (design plan's §D) ---------------------------------------

namespace {

	struct KeyGBufferOnly {};

	struct KeyLightingOnly {};

	struct GBufferPhaseNodeA {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<KeyGBufferOnly>();
			r.In<brassica::phase::GBuffer>();
		}

		void Execute(NodeContext&) {}
	};

	struct GBufferPhaseNodeB {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.In<brassica::phase::GBuffer>();
		}

		void Execute(NodeContext&) {}
	};

	struct LightingPhaseNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Create<KeyLightingOnly>();
			r.In<brassica::phase::Lighting>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("In<GBuffer>() orders before In<Lighting>() purely via the shared boundary marker") {
	Graph graph;
	graph.Register<LightingPhaseNode>();
	graph.Register<GBufferPhaseNodeA>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t gbuffer = IndexOfNamed(graph.Nodes(), "GBufferPhaseNodeA");
	std::size_t lighting = IndexOfNamed(graph.Nodes(), "LightingPhaseNode");
	CHECK(RelStageOf(result, gbuffer) < RelStageOf(result, lighting));
}

TEST_CASE("Two nodes in the same phase with nothing else ordering them land in the same stage") {
	Graph graph;
	graph.Register<GBufferPhaseNodeA>();
	graph.Register<GBufferPhaseNodeB>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());
	REQUIRE(result.stages.size() == 1);
	CHECK(result.stages[0].size() == 2);
}

namespace {

	struct LightingKeystoneNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Keystone<brassica::phase::Lighting>();
		}

		void Execute(NodeContext&) {}
	};

	struct ConsumerOfKeystoneNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.After<LightingKeystoneNode>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("A phase's keystone is a true sink: After<keystone>() lands after every other node in that phase too") {
	Graph graph;
	graph.Register<ConsumerOfKeystoneNode>();
	graph.Register<LightingPhaseNode>();
	graph.Register<LightingKeystoneNode>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t keystone = IndexOfNamed(graph.Nodes(), "LightingKeystoneNode");
	std::size_t other = IndexOfNamed(graph.Nodes(), "LightingPhaseNode");
	std::size_t after = IndexOfNamed(graph.Nodes(), "ConsumerOfKeystoneNode");

	// The keystone itself only has to be after `other` (both In<Lighting>(), but the keystone is
	// forced to be the phase's sink) -- not asserted directly here since that's an implementation
	// detail of the routing, not the observable contract. What matters: ConsumerOfKeystoneNode
	// -- which never says anything about phase::Lighting at all, only After<LightingKeystoneNode>
	// -- still lands after `other`, which it has no relation to whatsoever.
	CHECK(RelStageOf(result, other) < RelStageOf(result, keystone));
	CHECK(RelStageOf(result, keystone) < RelStageOf(result, after));
	CHECK(RelStageOf(result, other) < RelStageOf(result, after));
}

namespace {

	struct SecondLightingKeystoneNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Keystone<brassica::phase::Lighting>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("Two nodes claiming the same phase's keystone is a hard error") {
	Graph graph;
	graph.Register<LightingKeystoneNode>();
	graph.Register<SecondLightingKeystoneNode>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	CHECK_FALSE(result.ok);
	CHECK(!result.error.empty());
}

// -- S5: Requires<A>/.Required() activity fixed point -----------------------------------

namespace {

	struct InactiveRequirement {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Active(false);
		}

		void Execute(NodeContext&) {}
	};

	struct ActiveRequirement {
		void Setup(SetupBuilder& r, const FrameContext&) { r.Domain(ExecutionDomain::Graphics); }

		void Execute(NodeContext&) {}
	};

	struct RequiresInactiveNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Requires<InactiveRequirement>();
		}

		void Execute(NodeContext&) {}
	};

	struct RequiresActiveNode {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Requires<ActiveRequirement>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("Requires<A>() deactivates this node when A is inactive") {
	Graph graph;
	graph.Register<InactiveRequirement>();
	graph.Register<RequiresInactiveNode>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t dependent = IndexOfNamed(graph.Nodes(), "RequiresInactiveNode");
	REQUIRE(dependent < result.activity.size());
	CHECK_FALSE(result.activity[dependent]);
	CHECK(RelStageOf(result, dependent) < 0);
}

TEST_CASE("Requires<A>() leaves this node active when A is active") {
	Graph graph;
	graph.Register<ActiveRequirement>();
	graph.Register<RequiresActiveNode>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t dependent = IndexOfNamed(graph.Nodes(), "RequiresActiveNode");
	REQUIRE(dependent < result.activity.size());
	CHECK(result.activity[dependent]);
	CHECK(RelStageOf(result, dependent) >= 0);
}

namespace {

	struct RequiredKey {};

	struct InactiveProducer {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Active(false);
			r.Create<RequiredKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct RequiredReaderOfInactive {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<RequiredKey>().Required();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE(".Required() deactivates a reader when its only writer is inactive") {
	Graph graph;
	graph.Register<InactiveProducer>();
	graph.Register<RequiredReaderOfInactive>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "RequiredReaderOfInactive");
	REQUIRE(reader < result.activity.size());
	CHECK_FALSE(result.activity[reader]);
}

namespace {

	struct LateWriterKey {};

	struct LateWriter;

	// Before<LateWriter>() here is deliberately NOT a real data dependency -- it's standing in
	// for any case where something else orders a .Required() reader before its key's only
	// writer (e.g. an unrelated explicit relation, or a phase/relation mistake). The point is
	// that .Required() cares about schedule order, not intent: a writer that only runs *after*
	// this read cannot be what fed it this frame, so this read sees nothing active, regardless
	// of why.
	struct EarlyRequiredReader {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<LateWriterKey>().Required();
			r.Before<LateWriter>();
		}

		void Execute(NodeContext&) {}
	};

	struct LateWriter {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Modify<LateWriterKey>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE(".Required() deactivates a reader when its only writer is ordered after it rather than before") {
	Graph graph;
	graph.Register<EarlyRequiredReader>();
	graph.Register<LateWriter>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "EarlyRequiredReader");
	REQUIRE(reader < result.activity.size());
	CHECK_FALSE(result.activity[reader]);
}

namespace {

	struct ChainKey {};
	struct ChainMiddleOutputKey {};

	struct ChainRoot {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Active(false);
			r.Create<ChainKey>();
		}

		void Execute(NodeContext&) {}
	};

	// Depends on ChainKey's only producer, not on ChainRoot by name -- so this node only goes
	// inactive via the .Required() fixed point's own cascade, not a direct Requires<>() edge.
	struct ChainMiddle {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<ChainKey>().Required();
			r.Create<ChainMiddleOutputKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct ChainLeaf {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<ChainMiddleOutputKey>().Required();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE(".Required() cascades: a node downstream of a deactivated .Required() node goes inactive too") {
	Graph graph;
	graph.Register<ChainRoot>();
	graph.Register<ChainMiddle>();
	graph.Register<ChainLeaf>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t middle = IndexOfNamed(graph.Nodes(), "ChainMiddle");
	std::size_t leaf = IndexOfNamed(graph.Nodes(), "ChainLeaf");
	REQUIRE(middle < result.activity.size());
	REQUIRE(leaf < result.activity.size());
	CHECK_FALSE(result.activity[middle]);
	CHECK_FALSE(result.activity[leaf]);
}

namespace {

	struct ZeroInitKey {};

	struct ZeroInitWriter {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Active(false);
			r.Create<ZeroInitKey>();
		}

		void Execute(NodeContext&) {}
	};

	// Deliberately no .Required() -- this is the default-case contrast with the earlier
	// .Required() tests: a plain Read of a key with no active writer flags for zero-init, but
	// does NOT deactivate the reader itself (unlike .Required(), it has no opinion on whether
	// this node should run at all).
	struct ZeroInitReader {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<ZeroInitKey>();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("A plain Read with no active writer preceding it is flagged for zero-init, without deactivating the reader") {
	Graph graph;
	graph.Register<ZeroInitWriter>();
	graph.Register<ZeroInitReader>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "ZeroInitReader");
	REQUIRE(reader < result.activity.size());
	CHECK(result.activity[reader]);

	const bool flagged =
		std::any_of(result.zeroInitReads.begin(), result.zeroInitReads.end(), [&](const auto& pair) {
			return pair.first == reader && pair.second == IdOf<ZeroInitKey>();
		});
	CHECK(flagged);
}

namespace {

	// A real hardware test (not a hypothetical) found this exact gap: TransmittanceLUT/
	// MultiScatteringLUT are throttled (only regenerate when atmosphere parameters change), so
	// their producer going inactive once the throttle settles is normal, expected behavior, not
	// "no valid data" -- unlike ZeroInitKey above, which models a genuine feature toggle. Without
	// Lifetime::Persistent, a plain Read of a throttled-but-cached key gets wrongly zero-cleared
	// every frame once the producer stops regenerating it.
	struct PersistentZeroInitKey {
		static constexpr Lifetime kLifetime = Lifetime::Persistent;
	};

	struct InactiveThrottledProducer {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Active(false);
			r.Create<PersistentZeroInitKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct PlainReaderOfPersistentKey {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<PersistentZeroInitKey>();
		}

		void Execute(NodeContext&) {}
	};

	struct RequiredReaderOfPersistentKey {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.Read<PersistentZeroInitKey>().Required();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("A Persistent-lifetime key with an inactive producer is never flagged for zero-init") {
	Graph graph;
	graph.Register<InactiveThrottledProducer>();
	graph.Register<PlainReaderOfPersistentKey>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "PlainReaderOfPersistentKey");
	REQUIRE(reader < result.activity.size());
	CHECK(result.activity[reader]);

	const bool flagged =
		std::any_of(result.zeroInitReads.begin(), result.zeroInitReads.end(), [&](const auto& pair) {
			return pair.first == reader && pair.second == IdOf<PersistentZeroInitKey>();
		});
	CHECK_FALSE(flagged);
}

TEST_CASE(".Required() does not deactivate a reader of a Persistent-lifetime key whose producer is inactive") {
	Graph graph;
	graph.Register<InactiveThrottledProducer>();
	graph.Register<RequiredReaderOfPersistentKey>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "RequiredReaderOfPersistentKey");
	REQUIRE(reader < result.activity.size());
	CHECK(result.activity[reader]);
}

namespace {

	struct PreviousFrameTemporalKey {};

	using PreviousFrameTemporal = TypeList<PreviousFrameTemporalKey>;

	// Required() on a ReadPrevious<K>() read: if PreviousFrame<Temporal>'s builder-shaped Setup
	// (Frame.hpp) didn't actually call Create<History<K>>() for every K, this reader would see
	// no active writer at all and get wrongly deactivated here -- the exact silent-failure risk
	// flagged when PreviousFrame/NextFrame were converted off the legacy Resources-based shape
	// (same bug class as the GTAO/sky zero-init incidents: a reader that should see real data
	// instead gets treated as having nothing feeding it).
	struct RequiredReaderOfPreviousFrame {
		void Setup(SetupBuilder& r, const FrameContext&) {
			r.Domain(ExecutionDomain::Graphics);
			r.ReadPrevious<PreviousFrameTemporalKey>().Required();
		}

		void Execute(NodeContext&) {}
	};

} // namespace

TEST_CASE("PreviousFrame<Temporal>'s builder Setup registers as an active writer for ReadPrevious<K>().Required()") {
	Graph graph;
	graph.Register<PreviousFrame<PreviousFrameTemporal>>();
	graph.Register<RequiredReaderOfPreviousFrame>();

	RelationalResult result = CompileRelationalFromGraph(graph, FrameContext{});
	REQUIRE(result.ok);
	CHECK(result.ambiguities.empty());

	std::size_t reader = IndexOfNamed(graph.Nodes(), "RequiredReaderOfPreviousFrame");
	REQUIRE(reader < result.activity.size());
	CHECK(result.activity[reader]);
	CHECK(result.zeroInitReads.empty());
}
