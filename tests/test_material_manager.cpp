#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include <doctest/doctest.h>
#include <glm/glm.hpp>

#include "graph/Graph.hpp"
#include "render/MaterialManager.hpp"

TEST_CASE("MaterialData - std430 Layout and Sentinel Values") {
	CHECK(sizeof(brassica::MaterialData) == 128);
	CHECK(brassica::UNBOUND_TEXTURE_ID == 0xFFFFFFFFu);

	brassica::MaterialData mat;
	CHECK(mat.textureIds.x == brassica::UNBOUND_TEXTURE_ID);
	CHECK(mat.textureIds.y == brassica::UNBOUND_TEXTURE_ID);
	CHECK(mat.textureIds.z == brassica::UNBOUND_TEXTURE_ID);
	CHECK(mat.textureIds.w == brassica::UNBOUND_TEXTURE_ID);

	CHECK(mat.colorTintOffset == glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
	CHECK(mat.shadingBitmask == 0u);
	CHECK(mat.uvScaleOffset == glm::vec4(1.0f, 1.0f, 0.0f, 0.0f));

	CHECK(mat.palette[0] == glm::vec4(0.5f, 0.5f, 0.5f, 0.0f));
	CHECK(mat.palette[1] == glm::vec4(0.5f, 0.5f, 0.5f, 0.0f));
	CHECK(mat.palette[2] == glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
	CHECK(mat.palette[3] == glm::vec4(0.0f, 0.33f, 0.67f, 0.0f));
}

TEST_CASE("MaterialManager - Material Registration, Retrieval, and Updates") {
	brassica::MaterialManager mgr;
	mgr.Initialize();

	// Default material exists at index 0
	CHECK(mgr.GetMaterialCount() == 1);
	const auto* defaultMat = mgr.GetMaterial(0);
	REQUIRE(defaultMat != nullptr);
	CHECK(defaultMat->textureIds.x == brassica::UNBOUND_TEXTURE_ID);

	// Register a custom model material
	brassica::MaterialData customMat{};
	customMat.textureIds = glm::uvec4(10, 11, 12, 13);
	customMat.colorTintOffset = glm::vec4(0.1f, 0.8f, 0.2f, 5.0f);
	customMat.shadingBitmask = 0b00000001;
	customMat.uvScaleOffset = glm::vec4(2.0f, 2.0f, 0.5f, 0.5f);
	customMat.palette[0] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

	std::uint32_t id1 = mgr.RegisterMaterial(customMat);
	CHECK(id1 == 1);
	CHECK(mgr.GetMaterialCount() == 2);

	const auto* retrievedMat = mgr.GetMaterial(id1);
	REQUIRE(retrievedMat != nullptr);
	CHECK(retrievedMat->textureIds == glm::uvec4(10, 11, 12, 13));
	CHECK(retrievedMat->colorTintOffset.x == doctest::Approx(0.1f));
	CHECK(retrievedMat->colorTintOffset.y == doctest::Approx(0.8f));
	CHECK(retrievedMat->colorTintOffset.z == doctest::Approx(0.2f));
	CHECK(retrievedMat->colorTintOffset.w == doctest::Approx(5.0f));
	CHECK(retrievedMat->shadingBitmask == 1u);
	CHECK(retrievedMat->uvScaleOffset == glm::vec4(2.0f, 2.0f, 0.5f, 0.5f));

	// Update existing material
	customMat.shadingBitmask = 0b00000011;
	bool updated = mgr.UpdateMaterial(id1, customMat);
	CHECK(updated == true);
	CHECK(mgr.GetMaterial(id1)->shadingBitmask == 3u);

	// Update invalid ID fails
	bool invalidUpdate = mgr.UpdateMaterial(999, customMat);
	CHECK(invalidUpdate == false);
	CHECK(mgr.GetMaterial(999) == nullptr);
}

TEST_CASE("MaterialManager - Persistence Across Simulated Frame Cycles") {
	brassica::MaterialManager mgr;
	mgr.Initialize();

	brassica::MaterialData mat1{};
	mat1.textureIds.x = 42;
	std::uint32_t id1 = mgr.RegisterMaterial(mat1);

	// Simulate frame graph builds across multiple frames
	for (int frame = 0; frame < 5; ++frame) {
		brassica::graph::Graph frameGraph;
		mgr.RegisterBufferNode(frameGraph);

		const auto& bufferNode = mgr.GetBufferNode();
		REQUIRE(bufferNode.data.size() == mgr.GetMaterialCount());
		CHECK(bufferNode.data[id1].textureIds.x == 42);
	}

	// Register another material later during frame execution
	brassica::MaterialData mat2{};
	mat2.textureIds.x = 100;
	std::uint32_t id2 = mgr.RegisterMaterial(mat2);

	CHECK(mgr.GetMaterialCount() == 3);
	CHECK(mgr.GetMaterial(id2)->textureIds.x == 100);

	// Both materials persist
	CHECK(mgr.GetMaterial(id1)->textureIds.x == 42);
	CHECK(mgr.GetMaterial(id2)->textureIds.x == 100);

	// Reset materials back to default
	mgr.ClearMaterials();
	CHECK(mgr.GetMaterialCount() == 1);
	CHECK(mgr.GetMaterial(0)->textureIds.x == brassica::UNBOUND_TEXTURE_ID);
}

TEST_CASE("MaterialData - Cosine Palette Evaluation") {
	brassica::MaterialData mat{};
	mat.palette[0] = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f); // a
	mat.palette[1] = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f); // b
	mat.palette[2] = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f); // c
	mat.palette[3] = glm::vec4(0.0f, 0.33f, 0.67f, 0.0f); // d

	auto evaluatePalette = [](const brassica::MaterialData& m, float t) {
		constexpr float TWO_PI = 6.283185307179586f;
		glm::vec3 a = glm::vec3(m.palette[0]);
		glm::vec3 b = glm::vec3(m.palette[1]);
		glm::vec3 c = glm::vec3(m.palette[2]);
		glm::vec3 d = glm::vec3(m.palette[3]);
		return a + b * glm::cos(TWO_PI * (c * t + d));
	};

	glm::vec3 col0 = evaluatePalette(mat, 0.0f);
	CHECK(col0.x == doctest::Approx(1.0f));

	glm::vec3 col05 = evaluatePalette(mat, 0.5f);
	CHECK(col05.x == doctest::Approx(0.0f));
}
