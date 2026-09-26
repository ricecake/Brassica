#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

// 14 before EntityNode replaced BallNode: BallNode used to be a plain auto-registered node like
// every other, but EntityNode<Tag> is instantiated per-Tag and registered dynamically per frame
// via SystemHandler::GetEntityNode().RegisterInto (Engine::DrawFrame), not through this static
// list -- see include/passes/AllNodes.hpp's own header comment for why that list exists at all.
// 13 -> 15: UnderwaterParticleRenderNode/AboveWaterParticleRenderNode were promoted out of
// ParticleSystemNode's Subgraph to be independent top-level nodes (ParticleSystemNode.hpp)...
// 15 -> 22: Added 7 cloud layer nodes (CloudBakeNode, CloudBoundingNode, CloudShadowBakeNode,
// CloudTileSchedulerNode, CloudRenderNode, CloudTemporalNode, CloudSpatialFilterNode) in CloudNodes.hpp.
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 22);
}
