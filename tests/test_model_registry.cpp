#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "model/ModelRegistry.hpp"
#include "model/ProceduralTreeAsset.hpp"

using namespace brassica;

namespace {

	// A tiny synthetic asset -- just enough shape to exercise ModelRegistry's own concatenation
	// bookkeeping (base offsets), not a real renderable model.
	ModelAsset MakeTinyAsset(std::uint32_t vertexCount, std::uint32_t meshletVertexCount, std::uint32_t triangleWordCount, std::uint32_t meshletCount, std::uint32_t jointCount, std::uint32_t clipCount) {
		ModelAsset asset;
		asset.vertices.resize(vertexCount);
		asset.meshletVertices.resize(meshletVertexCount);
		asset.meshletTrianglesPacked.resize(triangleWordCount);
		asset.meshlets.resize(meshletCount);
		asset.jointCount = jointCount;
		asset.palette.resize(jointCount); // rest pose only
		for (std::uint32_t i = 0; i < clipCount; ++i) {
			ModelClipEntry clip{};
			clip.paletteBase = static_cast<std::uint32_t>(asset.palette.size());
			clip.frameCount = 2;
			clip.durationMs = 1000;
			clip.frameDtMs = 1000.0f;
			asset.clipTable.push_back(clip);
			asset.palette.resize(asset.palette.size() + jointCount * clip.frameCount);
		}
		asset.restBoundsCenter = glm::vec3(0.0f);
		asset.restBoundsRadius = 1.0f;
		return asset;
	}

} // namespace

TEST_CASE("ModelRegistry concatenates multiple models with correct base offsets") {
	ModelRegistry registry;

	ModelAsset first = MakeTinyAsset(10, 20, 5, 2, 3, 1);
	ModelAsset second = MakeTinyAsset(7, 14, 3, 1, 4, 2);

	std::uint32_t firstIndex = registry.RegisterAsset(first);
	std::uint32_t secondIndex = registry.RegisterAsset(second);

	CHECK(firstIndex == 0);
	CHECK(secondIndex == 1);
	CHECK(registry.ModelCount() == 2);

	CHECK(registry.GetMeshletCount(firstIndex) == 2);
	CHECK(registry.GetMeshletCount(secondIndex) == 1);
	CHECK(registry.GetJointCount(firstIndex) == 3);
	CHECK(registry.GetJointCount(secondIndex) == 4);
	CHECK(registry.GetClipCount(firstIndex) == 1);
	CHECK(registry.GetClipCount(secondIndex) == 2);

	// First model's bases all start at 0 (nothing registered before it).
	CHECK(registry.GetTotalMeshletCount() == 3);

	// Out-of-range queries degrade to zero rather than throwing/asserting -- the shader-side
	// equivalent (ModelDescEntry lookup) has no bounds check of its own, so the CPU-side
	// accessors staying defensive here is deliberate.
	CHECK(registry.GetMeshletCount(99) == 0);
	CHECK(registry.GetJointCount(99) == 0);
	CHECK(registry.GetClipCount(99) == 0);
}

TEST_CASE("BuildProceduralTreeAsset produces internally consistent geometry") {
	ModelAsset tree = BuildProceduralTreeAsset();

	// 3 segments: 34 vertices (16 side + 9 bottom cap + 9 top cap) and 32 triangles (16 side + 8
	// bottom + 8 top) each -- see ProceduralTreeAsset.cpp's own AppendCylinderSegment.
	CHECK(tree.vertices.size() == 34 * 3);
	CHECK(tree.meshletVertices.size() == 34 * 3);
	CHECK(tree.meshlets.size() == 3);
	CHECK(tree.jointCount == 3);

	for (const GltfVertex& v : tree.vertices) {
		float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
		CHECK(weightSum == doctest::Approx(1.0f)); // rigidly skinned, single joint
		CHECK(v.joints.x < tree.jointCount);
	}

	// Rest pose occupies palette[0 .. jointCount), one "Sway" clip follows.
	CHECK(tree.clipTable.size() == 1);
	CHECK(tree.clipTable[0].paletteBase == tree.jointCount);
	CHECK(tree.clipTable[0].frameCount >= 2);
	CHECK(tree.clipTable[0].durationMs == 2000);
	CHECK(tree.palette.size() == tree.jointCount + static_cast<std::size_t>(tree.clipTable[0].frameCount) * tree.jointCount);

	// Bounds must be real (non-degenerate) and at least as tall as the trunk's own total height.
	CHECK(tree.restBoundsRadius > 0.0f);
	CHECK(tree.clipTable[0].boundsRadius > 0.0f);
	CHECK(tree.restBoundsRadius >= 1.5f); // >= the trunk segment's own height alone

	// Stage 4: skeleton in topological order (parentIndex < this joint's own index), rigidity
	// tapering from the rigid trunk base to the flexible tip.
	REQUIRE(tree.skeleton.size() == tree.jointCount);
	CHECK(tree.skeleton[0].parentIndex == kModelNoSkeletonSentinel);
	CHECK(tree.skeleton[1].parentIndex == 0);
	CHECK(tree.skeleton[2].parentIndex == 1);
	for (const ModelSkeletonJoint& joint : tree.skeleton) {
		CHECK(joint.rigidity >= 0.0f);
		CHECK(joint.rigidity <= 1.0f);
	}
	CHECK(tree.skeleton[0].rigidity > tree.skeleton[1].rigidity);
	CHECK(tree.skeleton[1].rigidity > tree.skeleton[2].rigidity);
}
