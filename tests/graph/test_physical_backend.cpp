#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceKey.hpp"

using namespace brassica::graph;

namespace {

	struct TestColorTarget {};

	struct TestDepthTarget {};

	struct TestBufferTarget {};

	struct PassA {
		using Resources = Declares<Create<TestColorTarget>, Create<TestDepthTarget>>;

		Recipe Setup(const FrameContext& ctx) {
			Recipe r{.domain = ExecutionDomain::Graphics};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestColorTarget>(),
					.access = AccessKind::Write,
					.desc = ResourceDesc{
						.kind = ResourceDesc::Kind::Image2D,
						.width = ctx.width,
						.height = ctx.height,
						.formatCode = 44,        // VK_FORMAT_R8G8B8A8_UNORM
						.usageMask = 0x10 | 0x04 // ColorAttachment | Sampled
					}
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestDepthTarget>(),
					.access = AccessKind::Write,
					.desc = ResourceDesc{
						.kind = ResourceDesc::Kind::Image2D,
						.width = ctx.width,
						.height = ctx.height,
						.formatCode = 126, // VK_FORMAT_D32_SFLOAT
						.usageMask = 0x20  // DepthStencilAttachment
					}
				}
			);
			return r;
		}

		void Execute(CommandBuffer&) {}
	};

	struct PassB {
		using Resources = Declares<Read<TestColorTarget>, Create<TestBufferTarget>>;

		Recipe Setup(const FrameContext&) {
			Recipe r{.domain = ExecutionDomain::Compute};
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestColorTarget>(),
					.access = AccessKind::Read,
					.desc = ResourceDesc{
						.kind = ResourceDesc::Kind::Image2D,
						.width = 1920,
						.height = 1080,
						.formatCode = 44
					}
				}
			);
			r.realizations.push_back(
				ResourceRealization{
					.key = IdOf<TestBufferTarget>(),
					.access = AccessKind::Write,
					.desc = ResourceDesc{
						.kind = ResourceDesc::Kind::Buffer,
						.usageMask = 0x20, // StorageBuffer
						.byteSize = 1024
					}
				}
			);
			return r;
		}

		void Execute(CommandBuffer&) {}
	};

} // namespace

TEST_CASE("PhysicalResourceRegistry provisions resources and tracks bindless indices") {
	PhysicalResourceRegistry registry;

	Graph graph;
	graph.Register<PassA>();
	graph.Register<PassB>();

	FrameContext ctx{.width = 1920, .height = 1080};
	graph.Setup(ctx);
	REQUIRE(graph.Compile().has_value());

	registry.Provision(graph.GetSchedule(), graph.Recipes(), true);

	const auto* colorTex = registry.GetTexture<TestColorTarget>();
	REQUIRE(colorTex != nullptr);
	CHECK(colorTex->desc.width == 1920);
	CHECK(colorTex->desc.height == 1080);
	CHECK(colorTex->desc.formatCode == 44);

	const auto* depthTex = registry.GetTexture<TestDepthTarget>();
	REQUIRE(depthTex != nullptr);
	CHECK(depthTex->desc.formatCode == 126);

	const auto* buf = registry.GetBuffer<TestBufferTarget>();
	REQUIRE(buf != nullptr);
	CHECK(buf->desc.byteSize == 1024);
}

TEST_CASE("PhysicalExecutionBackend runs full physical pipeline cleanly") {
	PhysicalResourceRegistry registry;
	PhysicalExecutionBackend backend(registry);

	Graph graph;
	graph.Register<PassA>();
	graph.Register<PassB>();

	FrameContext  ctx{.width = 1280, .height = 720};
	CommandBuffer cmd{};

	CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, true));

	CHECK(registry.GetTexture<TestColorTarget>() != nullptr);
	CHECK(registry.GetTexture<TestDepthTarget>() != nullptr);
	CHECK(registry.GetBuffer<TestBufferTarget>() != nullptr);
}
