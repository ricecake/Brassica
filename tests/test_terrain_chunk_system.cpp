#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "resources.hpp"
#include "passes/terrain/nodes/TerrainChunkManagerNode.hpp"

using namespace brassica;

TEST_CASE("Terrain Chunk System Buffer Descriptors and Constants") {
	constexpr std::uint32_t dim = constants::Class::Terrain::ChunkManager::ChunkDimension;
	constexpr std::uint32_t vertsPerChunk = constants::Class::Terrain::ChunkManager::VerticesPerChunk;
	constexpr std::uint32_t quadsPerChunk = constants::Class::Terrain::ChunkManager::QuadsPerChunk;
	constexpr std::uint32_t trisPerChunk = constants::Class::Terrain::ChunkManager::TrianglesPerChunk;
	constexpr std::uint32_t indicesPerChunk = constants::Class::Terrain::ChunkManager::IndicesPerChunk;
	constexpr std::uint32_t maxVirtual = constants::Class::Terrain::ChunkManager::MaxVirtualChunks;
	constexpr std::uint32_t maxAlloc = constants::Class::Terrain::ChunkManager::MaxAllocatedChunks;

	CHECK(dim == 16);
	CHECK(vertsPerChunk == 256);
	CHECK(quadsPerChunk == 225);
	CHECK(trisPerChunk == 450);
	CHECK(indicesPerChunk == 1350);
	CHECK(maxVirtual == 3072);
	CHECK(maxAlloc == 4096);

	graph::FrameContext ctx{.width = 1280, .height = 720};

	TerrainPageTableBuffer         ptKey;
	TerrainIndirectionBuffer       indKey;
	TerrainChunkLodBuffer          lodKey;
	TerrainChunkRequirementsBuffer reqKey;
	TerrainChunkVertexBuffer       vertKey;

	auto ptDesc = ptKey.Desc(ctx);
	auto indDesc = indKey.Desc(ctx);
	auto lodDesc = lodKey.Desc(ctx);
	auto reqDesc = reqKey.Desc(ctx);
	auto vertDesc = vertKey.Desc(ctx);

	CHECK(ptDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(ptDesc.byteSize == maxVirtual * sizeof(TerrainPageTableEntryGPU));
	CHECK(ptDesc.persistent);

	CHECK(lodDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(lodDesc.byteSize == maxVirtual * sizeof(std::uint32_t));
	CHECK(lodDesc.persistent);

	CHECK(indDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(indDesc.byteSize == 64 + maxAlloc * sizeof(std::uint32_t) * 2);
	CHECK(indDesc.persistent);

	CHECK(reqDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(reqDesc.byteSize == 16 + maxVirtual * sizeof(std::uint32_t));
	CHECK(reqDesc.persistent);

	CHECK(vertDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(vertDesc.byteSize == maxAlloc * vertsPerChunk * sizeof(TerrainChunkVertexGPU));
	CHECK(vertDesc.persistent);
}

TEST_CASE("Terrain Chunk Pipeline Passes Register and Schedule Acyclically") {
	TerrainChunkResetNode        resetNode;
	TerrainChunkLODNode          lodNode;
	TerrainChunkRequirementsNode reqNode;
	TerrainChunkAllocNode        allocNode;
	TerrainChunkGenerateNode     genNode;
	TerrainChunkEvictNode        evictNode;
	TerrainChunkRenderNode       renderNode;

	graph::Graph g;
	g.RegisterRef(resetNode);
	g.RegisterRef(lodNode);
	g.RegisterRef(reqNode);
	g.RegisterRef(allocNode);
	g.RegisterRef(genNode);
	g.RegisterRef(evictNode);
	g.RegisterRef(renderNode);

	// GBuffer attachments and HiZ history read by renderNode
	g.Register<graph::Import<GBufferPosition>>();
	g.Register<graph::Import<GBufferNormal>>();
	g.Register<graph::Import<GBufferAlbedo>>();
	g.Register<graph::Import<GBufferMaterial>>();
	g.Register<graph::Import<GBufferDepth>>();
	g.Register<graph::Import<graph::History<HiZTexture>>>();

	graph::FrameContext ctx{.width = 1280, .height = 720};
	g.Setup(ctx);
	auto compileResult = g.Compile();
	REQUIRE(compileResult.has_value());

	const auto& schedule = g.GetSchedule();

	auto stageOf = [&](std::size_t nodeIndex) -> std::optional<std::size_t> {
		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			for (std::size_t n : schedule.stages[s].nodes) {
				if (n == nodeIndex) {
					return s;
				}
			}
		}
		return std::nullopt;
	};

	std::optional<std::size_t> resetStage = stageOf(0);
	std::optional<std::size_t> lodStage = stageOf(1);
	std::optional<std::size_t> reqStage = stageOf(2);
	std::optional<std::size_t> allocStage = stageOf(3);
	std::optional<std::size_t> genStage = stageOf(4);
	std::optional<std::size_t> evictStage = stageOf(5);
	std::optional<std::size_t> renderStage = stageOf(6);

	REQUIRE(resetStage.has_value());
	REQUIRE(lodStage.has_value());
	REQUIRE(reqStage.has_value());
	REQUIRE(allocStage.has_value());
	REQUIRE(genStage.has_value());
	REQUIRE(evictStage.has_value());
	REQUIRE(renderStage.has_value());

	// Verify sequential dependency order across passes:
	// Reset -> LOD -> Evict -> Requirements -> Alloc -> Generate -> Render
	CHECK(*lodStage >= *resetStage);
	CHECK(*evictStage >= *lodStage);
	CHECK(*reqStage >= *evictStage);
	CHECK(*allocStage >= *reqStage);
	CHECK(*genStage >= *allocStage);
	CHECK(*renderStage >= *genStage);
}
