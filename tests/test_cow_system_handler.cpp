#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <memory>

#include "Engine.hpp"
#include "animation/OzzModel.hpp"
#include "../apps/sandbox/CowSystemHandler.hpp"

TEST_CASE("OzzModel loads cow.glb and animates motions") {
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

	auto initPositions = model.GetPositions();
	REQUIRE(initPositions.size() == vertexCount);

	// Update animation
	model.Update(0.5f, 0); // First animation clip (e.g. Walk / Attack)
	auto animatedPositions = model.GetPositions();
	REQUIRE(animatedPositions.size() == vertexCount);

	// Test animation switching
	int walkIdx = model.FindAnimationIndex("Walk");
	CHECK(walkIdx >= 0);
	model.Update(0.0f, static_cast<std::size_t>(walkIdx)); // Switches animation and resets time
	model.Update(0.5f, static_cast<std::size_t>(walkIdx)); // Advances playback time
	CHECK(model.GetPlaybackTime() > 0.0f);
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
