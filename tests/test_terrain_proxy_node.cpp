#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainMinMaxDownsampleNode.hpp"
#include "passes/TerrainProxyNode.hpp"

using namespace brassica;

TEST_CASE("TerrainProxyNode registers, compiles graph dependencies, and orders after TerrainGenNode") {
	TerrainGenNode              genNode;
	TerrainMinMaxDownsampleNode downsampleNode;
	TerrainProxyNode            proxyNode;

	graph::Graph g;
	g.RegisterRef(genNode);
	g.RegisterRef(downsampleNode);
	g.RegisterRef(proxyNode);

	graph::FrameContext ctx{.width = 1024, .height = 1024};
	g.Setup(ctx);
	REQUIRE(g.Compile().has_value());

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

	std::optional<std::size_t> genStage = stageOf(0);
	std::optional<std::size_t> downsampleStage = stageOf(1);
	std::optional<std::size_t> proxyStage = stageOf(2);

	REQUIRE(genStage.has_value());
	REQUIRE(downsampleStage.has_value());
	REQUIRE(proxyStage.has_value());

	// TerrainProxyNode reads TerrainClipmapTexture and TerrainMinMaxTexture,
	// both created by TerrainGenNode and modified by TerrainMinMaxDownsampleNode.
	// Therefore, proxyStage must execute after genStage and downsampleStage.
	CHECK(*proxyStage >= *genStage);
	CHECK(*proxyStage >= *downsampleStage);

	// Verify ResourceIds for proxy geometry buffers are distinct and valid
	const graph::ResourceId ptId = graph::IdOf<TerrainProxyPageTableBuffer>();
	const graph::ResourceId indId = graph::IdOf<TerrainProxyIndirectionBuffer>();
	const graph::ResourceId vertId = graph::IdOf<TerrainProxyVertexBuffer>();
	const graph::ResourceId idxId = graph::IdOf<TerrainProxyIndexBuffer>();

	CHECK(ptId != indId);
	CHECK(vertId != idxId);
	CHECK(ptId == graph::IdOf<TerrainProxyPageTableBuffer>());
	CHECK(indId == graph::IdOf<TerrainProxyIndirectionBuffer>());
}

TEST_CASE("Terrain Proxy Geometry Chunk Layout Math") {
	constexpr std::uint32_t dim = constants::Class::Terrain::Proxy::ChunkDimension;
	constexpr std::uint32_t vertsPerChunk = constants::Class::Terrain::Proxy::VerticesPerChunk;
	constexpr std::uint32_t quadsPerChunk = constants::Class::Terrain::Proxy::QuadsPerChunk;
	constexpr std::uint32_t indicesPerChunk = constants::Class::Terrain::Proxy::IndicesPerChunk;

	CHECK(dim == 5);
	CHECK(vertsPerChunk == 25);
	CHECK(quadsPerChunk == 16);
	CHECK(indicesPerChunk == 96);

	// Verify buffer descriptor sizing
	graph::FrameContext ctx{.width = 1024, .height = 1024};
	TerrainProxyPageTableBuffer   ptKey;
	TerrainProxyIndirectionBuffer indKey;
	TerrainProxyVertexBuffer      vertKey;
	TerrainProxyIndexBuffer       idxKey;

	auto ptDesc = ptKey.Desc(ctx);
	auto indDesc = indKey.Desc(ctx);
	auto vertDesc = vertKey.Desc(ctx);
	auto idxDesc = idxKey.Desc(ctx);

	CHECK(ptDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(ptDesc.width == constants::Class::Terrain::Proxy::MaxVirtualChunks * sizeof(std::uint32_t));
	CHECK(ptDesc.persistent);

	CHECK(indDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(indDesc.persistent);

	CHECK(vertDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(vertDesc.width == constants::Class::Terrain::Proxy::MaxAllocatedChunks * vertsPerChunk * sizeof(TerrainProxyVertexGPU));
	CHECK(vertDesc.persistent);

	CHECK(idxDesc.kind == graph::ResourceDesc::Kind::Buffer);
	CHECK(idxDesc.byteSize == indicesPerChunk * sizeof(std::uint32_t));
	CHECK(idxDesc.persistent);
}
