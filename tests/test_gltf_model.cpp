#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "model/GltfModel.hpp"

// Pure CPU-side coverage (fastgltf parse + meshopt_buildMeshlets), no Vulkan device needed --
// runs under ctest's WORKING_DIRECTORY = CMAKE_SOURCE_DIR, so "assets/..." resolves to the repo's
// real test assets (same ones GltfModelManager::LoadFromFile loads at engine startup).

using namespace brassica;

TEST_CASE("LoadFromFile succeeds for a real test asset and produces consistent buffers") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));
	CHECK(model.IsLoaded());

	CHECK(!model.Vertices().empty());
	CHECK(!model.Meshlets().empty());
	CHECK(!model.MeshletVertices().empty());
	CHECK(!model.MeshletTrianglesPacked().empty());

	// Every meshlet's vertex/triangle ranges must stay inside the buffers they index into --
	// this is exactly the kind of offset-math mistake the triangleOffset unit conversion
	// (meshopt's byte offset -> GltfMeshlet's triangle-unit offset, see GltfModel.cpp) could get
	// wrong without anything else catching it, since a wrong but in-bounds read wouldn't crash.
	for (const GltfMeshlet& meshlet : model.Meshlets()) {
		CHECK(meshlet.vertexCount > 0);
		CHECK(meshlet.vertexCount <= 128);
		CHECK(meshlet.triangleCount > 0);
		CHECK(meshlet.triangleCount <= 128);

		CHECK(meshlet.vertexOffset + meshlet.vertexCount <= model.MeshletVertices().size());

		std::size_t lastTriangleCornerByte =
			static_cast<std::size_t>(meshlet.triangleOffset) * 3 + static_cast<std::size_t>(meshlet.triangleCount) * 3;
		CHECK(lastTriangleCornerByte <= model.MeshletTrianglesPacked().size() * 4);

		for (std::uint32_t i = 0; i < meshlet.vertexCount; ++i) {
			CHECK(model.MeshletVertices()[meshlet.vertexOffset + i] < model.Vertices().size());
		}

		// Before GltfModelManager::RemapMaterialIndices runs, every meshlet's materialIndex is
		// either the "no material" sentinel or a real index into Materials() -- cow.glb's own
		// primitives all reference a real material (confirmed via its own glTF JSON), so the
		// sentinel branch isn't exercised by this asset, but both are checked for either case.
		CHECK((meshlet.materialIndex == 0xFFFFFFFFu || meshlet.materialIndex < model.Materials().size()));
	}

	// cow.glb has 7 materials and zero images/textures (confirmed by inspecting its own glTF JSON
	// chunk directly) -- every material here is flat-factor-only, so this doesn't exercise image
	// decoding (stb_image) or the sRGB/UNORM role split at all. That path compiles and the shader
	// side is wired up, but is unverified against a real textured asset -- flagged, not silently
	// assumed correct.
	REQUIRE(model.Materials().size() == 7);
	CHECK(model.Images().empty());
	for (const GltfMaterial& material : model.Materials()) {
		CHECK(material.baseColorFactor.a > 0.0f);
		CHECK(material.metallicFactor >= 0.0f);
		CHECK(material.metallicFactor <= 1.0f);
		CHECK(material.roughnessFactor >= 0.0f);
		CHECK(material.roughnessFactor <= 1.0f);
		CHECK(material.baseColorImageIndex == -1);
	}
}

TEST_CASE("RemapMaterialIndices rewrites meshlet material indices and maps the sentinel to 0") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));

	std::vector<std::uint32_t> localToGlobal(model.Materials().size());
	for (std::size_t i = 0; i < localToGlobal.size(); ++i) {
		localToGlobal[i] = static_cast<std::uint32_t>(100 + i); // arbitrary, distinguishable offset
	}
	model.RemapMaterialIndices(localToGlobal);

	// Every meshlet lands in [100, 100+count) (a real local->global remap) except a primitive
	// with no material at all, which maps to 0 (MaterialManager's own default material) --
	// either is a valid outcome, not asserting cow.glb's primitives are all one or the other.
	for (const GltfMeshlet& meshlet : model.Meshlets()) {
		bool remapped = meshlet.materialIndex >= 100 && meshlet.materialIndex < 100 + localToGlobal.size();
		bool defaulted = meshlet.materialIndex == 0;
		CHECK((remapped || defaulted));
	}
}

TEST_CASE("Skeleton/animation parsing and pose evaluation against a real skinned asset") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));

	// Confirmed directly from cow.glb's own glTF JSON chunk: 1 skin with 42 joints, 25 animation
	// clips, every sampler LINEAR, only translation/rotation channels (no scale, no CUBICSPLINE) --
	// this test is grounded in that real data, not assumed shape.
	REQUIRE(model.JointCount() == 42);
	REQUIRE(model.Animations().size() == 25);
	for (const GltfAnimation& anim : model.Animations()) {
		CHECK(anim.duration > 0.0f);
		CHECK(!anim.channels.empty());
	}

	// Stage 1's JOINTS_0/WEIGHTS_0 decoding was written but never actually verified against a
	// real skinned asset's real weight data until now -- a skinned mesh's weights should sum to
	// ~1.0 per vertex (glTF's own normalization contract), not stay at the zero-filled default a
	// non-skinned primitive would have.
	bool anyVertexHasRealWeights = false;
	for (const GltfVertex& v : model.Vertices()) {
		float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
		if (weightSum > 0.9f) {
			anyVertexHasRealWeights = true;
			CHECK(weightSum < 1.1f);
		}
	}
	CHECK(anyVertexHasRealWeights);

	std::vector<glm::mat4> poseAtStart;
	model.EvaluatePose(0, 0.0f, poseAtStart);
	REQUIRE(poseAtStart.size() == 42);
	for (const glm::mat4& m : poseAtStart) {
		// Every joint matrix must be a valid affine transform (bottom row (0,0,0,1)) and free of
		// NaN/Inf -- a real correctness check on EvaluatePose's matrix math, not just "ran without
		// crashing."
		CHECK(m[0][3] == doctest::Approx(0.0f));
		CHECK(m[1][3] == doctest::Approx(0.0f));
		CHECK(m[2][3] == doctest::Approx(0.0f));
		CHECK(m[3][3] == doctest::Approx(1.0f));
		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				CHECK(std::isfinite(m[col][row]));
			}
		}
	}

	// A real correctness check that sampling actually moves the pose over time, not just
	// returning the rest pose regardless of `time` -- animation 0 ("Attack_Headbutt") has a
	// nonzero duration (checked above), so sampling partway through it must differ from t=0 for
	// at least one joint.
	std::vector<glm::mat4> poseAtMid;
	model.EvaluatePose(0, model.Animations()[0].duration * 0.5f, poseAtMid);
	REQUIRE(poseAtMid.size() == 42);
	bool anyJointMoved = false;
	for (std::size_t i = 0; i < poseAtStart.size(); ++i) {
		if (poseAtStart[i] != poseAtMid[i]) {
			anyJointMoved = true;
			break;
		}
	}
	CHECK(anyJointMoved);

	// An out-of-range animation index (including "no animation at all") must fall back to the
	// rest pose cleanly, never crash or produce garbage -- EvaluatePose's own documented contract.
	std::vector<glm::mat4> restPose;
	model.EvaluatePose(9999, 0.0f, restPose);
	REQUIRE(restPose.size() == 42);
	for (const glm::mat4& m : restPose) {
		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				CHECK(std::isfinite(m[col][row]));
			}
		}
	}
}

TEST_CASE("LoadFromFile fails cleanly for a nonexistent file") {
	GltfModel model;
	CHECK_FALSE(model.LoadFromFile("assets/does_not_exist.glb"));
	CHECK_FALSE(model.IsLoaded());
}
