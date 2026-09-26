#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

// 14 before EntityNode replaced BallNode: BallNode used to be a plain auto-registered node like
// every other, but EntityNode<Tag> is instantiated per-Tag and registered dynamically per frame
// via SystemHandler::GetEntityNode().RegisterInto (Engine::DrawFrame), not through this static
// list -- see include/passes/AllNodes.hpp's own header comment for why that list exists at all.
// 13 -> 15: UnderwaterParticleRenderNode/AboveWaterParticleRenderNode were promoted out of
// ParticleSystemNode's Subgraph to be independent top-level nodes (ParticleSystemNode.hpp), so
// their SubPhase::UnderwaterParticleRender/ParticleRender phases actually reach the outer
// scheduler relative to WaterNode's SubPhase::WaterRender -- a Subgraph child's own kPhase only
// orders it against its Subgraph siblings, never an outer sibling.
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 16);
}
