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

	// Same shape, plus a real (if synthetic) skeleton -- MakeTinyAsset alone never populates
	// asset.skeleton, so every model it builds reads as jointless to the registry
	// (skeletonBase == kModelNoSkeletonSentinel) regardless of jointCount; this variant exercises
	// the actual skeleton-bookkeeping path ReplaceAsset also has to get right.
	ModelAsset MakeTinyAssetWithSkeleton(
		std::uint32_t vertexCount, std::uint32_t meshletVertexCount, std::uint32_t triangleWordCount,
		std::uint32_t meshletCount, std::uint32_t jointCount, std::uint32_t clipCount
	) {
		ModelAsset asset = MakeTinyAsset(vertexCount, meshletVertexCount, triangleWordCount, meshletCount, jointCount, clipCount);
		asset.skeleton.resize(jointCount);
		return asset;
	}

} // namespace

TEST_CASE("ModelRegistry::ReplaceAsset overwrites one model in place, leaving siblings untouched") {
	ModelRegistry registry;
	std::uint32_t first = registry.RegisterAsset(MakeTinyAsset(10, 20, 5, 2, 0, 0));
	std::uint32_t middle = registry.RegisterAsset(MakeTinyAsset(7, 14, 3, 1, 0, 1));
	std::uint32_t last = registry.RegisterAsset(MakeTinyAsset(5, 10, 2, 1, 0, 0));

	registry.ReplaceAsset(middle, MakeTinyAsset(20, 40, 9, 4, 0, 2));

	CHECK(registry.ModelCount() == 3);
	CHECK(registry.GetMeshletCount(first) == 2);
	CHECK(registry.GetMeshletCount(middle) == 4); // the new asset's own count
	CHECK(registry.GetClipCount(middle) == 2);
	CHECK(registry.GetMeshletCount(last) == 1);
	CHECK(registry.GetClipCount(last) == 0);
	CHECK(registry.GetTotalMeshletCount() == 2 + 4 + 1);
}

TEST_CASE("ModelRegistry::ReplaceAsset shifts a later model's skeleton correctly when joint counts differ") {
	ModelRegistry registry;
	std::uint32_t first = registry.RegisterAsset(MakeTinyAssetWithSkeleton(4, 8, 2, 1, 3, 0));
	std::uint32_t second = registry.RegisterAsset(MakeTinyAssetWithSkeleton(4, 8, 2, 1, 5, 0));

	registry.ReplaceAsset(first, MakeTinyAssetWithSkeleton(4, 8, 2, 1, 9, 0));

	CHECK(registry.GetJointCount(first) == 9);
	CHECK(registry.GetJointCount(second) == 5); // untouched by the replace, just shifted
}

TEST_CASE("ModelRegistry::ReplaceAsset handles a jointless model becoming skeletal, and back") {
	ModelRegistry registry;
	std::uint32_t jointless = registry.RegisterAsset(MakeTinyAsset(4, 8, 2, 1, 0, 0));
	std::uint32_t skeletal = registry.RegisterAsset(MakeTinyAssetWithSkeleton(4, 8, 2, 1, 6, 0));

	// Was jointless -- gains a real skeleton.
	registry.ReplaceAsset(jointless, MakeTinyAssetWithSkeleton(4, 8, 2, 1, 3, 0));
	CHECK(registry.GetJointCount(jointless) == 3);
	CHECK(registry.GetJointCount(skeletal) == 6); // still correct after the earlier model's skeleton insert

	// Now goes back to jointless.
	registry.ReplaceAsset(jointless, MakeTinyAsset(4, 8, 2, 1, 0, 0));
	CHECK(registry.GetJointCount(jointless) == 0);
	CHECK(registry.GetJointCount(skeletal) == 6); // still correct after the removal too
}

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
