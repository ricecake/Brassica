#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <memory>

#include "Engine.hpp"
#include "animation/OzzModel.hpp"
#include "../apps/sandbox/CowSystemHandler.hpp"

TEST_CASE("OzzModel loads cow.glb and animates motions via OzzModelInstance") {
	std::filesystem::path cowPath = "assets/cow.glb";
	if (!std::filesystem::exists(cowPath)) {
		cowPath = "../assets/cow.glb";
	}
	REQUIRE(std::filesystem::exists(cowPath));

	brassica::OzzModel model(cowPath);
	REQUIRE(model.GetSkeleton() != nullptr);
	CHECK(model.GetSkeleton()->num_joints() == 42);
	CHECK(model.GetAnimationCount() > 0);

	std::size_t vertexCount = model.GetVertexCount();
	std::size_t triangleCount = model.GetTriangleCount();
	CHECK(vertexCount > 0);
	CHECK(triangleCount > 0);

	auto inst1 = model.CreateInstance();
	auto inst2 = model.CreateInstance();

	REQUIRE(inst1.skinnedPositions.size() == vertexCount);
	REQUIRE(inst2.skinnedPositions.size() == vertexCount);

	// Update inst1 withWalk and inst2 with Gallop independently
	int walkIdx = model.FindAnimationIndex("Walk");
	int gallopIdx = model.FindAnimationIndex("Gallop");
	CHECK(walkIdx >= 0);
	CHECK(gallopIdx >= 0);

	model.UpdateInstance(inst1, 0.0f, static_cast<std::size_t>(walkIdx));
	model.UpdateInstance(inst2, 0.0f, static_cast<std::size_t>(gallopIdx));
	model.UpdateInstance(inst1, 0.2f, static_cast<std::size_t>(walkIdx));
	model.UpdateInstance(inst2, 0.3f, static_cast<std::size_t>(gallopIdx));

	CHECK(inst1.playbackTime == 0.2f);
	CHECK(inst2.playbackTime == 0.3f);
	CHECK(inst1.currentAnimIndex == static_cast<std::size_t>(walkIdx));
	CHECK(inst2.currentAnimIndex == static_cast<std::size_t>(gallopIdx));
}

TEST_CASE("CowSystemHandler registers and manages cow entities in Engine") {
	brassica::EngineOptions options{};
	options.headless = true;

	brassica::Engine engine;
	auto cowHandler = engine.AddSystemHandler<brassica::CowSystemHandler>();
	REQUIRE(cowHandler != nullptr);

	brassica::FrameDetails details{};
	details.totalTime = 0.0;
	details.deltaTime = 0.016f;

	cowHandler->Setup(engine, details);

	CHECK(cowHandler->GetEntities().size() == 6);
	CHECK(cowHandler->GetCowModel() != nullptr);

	// Update frame
	details.totalTime = 1.0;
	cowHandler->Update(engine, details);

	CHECK(cowHandler->GetEntities().size() == 6);

	// Cleanup
	cowHandler->ClearEntities(engine);
	CHECK(cowHandler->GetEntities().empty());
}
