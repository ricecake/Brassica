#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/epsilon.hpp>
#include <vector>

#include "gltf/GltfModel.hpp"
#include "gltf/GltfAnimation.hpp"
#include "gltf/GltfTextureManager.hpp"
#include "passes/GltfComputePass.hpp"
#include "passes/GltfRenderPass.hpp"

TEST_CASE("glTF Animation Sampling and Hierarchy Evaluation") {
	std::vector<brassica::GltfNode> nodes(3);

	// Root Node 0
	nodes[0].name = "Root";
	nodes[0].parent = -1;
	nodes[0].children = {1};
	nodes[0].translation = glm::vec3(0.0f, 10.0f, 0.0f);
	nodes[0].localMatrix = glm::translate(glm::mat4(1.0f), nodes[0].translation);

	// Child Node 1
	nodes[1].name = "Bone1";
	nodes[1].parent = 0;
	nodes[1].children = {2};
	nodes[1].translation = glm::vec3(0.0f, 5.0f, 0.0f);
	nodes[1].localMatrix = glm::translate(glm::mat4(1.0f), nodes[1].translation);

	// Child Node 2
	nodes[2].name = "Bone2";
	nodes[2].parent = 1;
	nodes[2].children = {};
	nodes[2].translation = glm::vec3(0.0f, 2.0f, 0.0f);
	nodes[2].localMatrix = glm::translate(glm::mat4(1.0f), nodes[2].translation);

	brassica::GltfAnimation anim;
	anim.name = "TestAnim";
	anim.duration = 2.0f;

	brassica::GltfAnimationChannel chan;
	chan.targetNode = 1;
	chan.path = brassica::GltfTargetPath::Translation;
	chan.interpolation = brassica::GltfInterpolationType::Linear;
	chan.samplersInput = {0.0f, 2.0f};
	chan.samplersOutput = {glm::vec4(0.0f, 5.0f, 0.0f, 0.0f), glm::vec4(0.0f, 15.0f, 0.0f, 0.0f)};
	anim.channels.push_back(chan);

	// Evaluate at t = 1.0 (midway)
	brassica::GltfAnimationEvaluator::EvaluateAnimation(anim, 1.0f, nodes);
	CHECK(nodes[1].translation.y == doctest::Approx(10.0f));

	// Compute global transforms
	brassica::GltfAnimationEvaluator::ComputeGlobalTransforms(nodes);

	CHECK(nodes[0].globalMatrix[3].y == doctest::Approx(10.0f));
	CHECK(nodes[1].globalMatrix[3].y == doctest::Approx(20.0f));
	CHECK(nodes[2].globalMatrix[3].y == doctest::Approx(22.0f));
}

TEST_CASE("glTF Skinning Matrices Computation") {
	std::vector<brassica::GltfNode> nodes(2);
	nodes[0].parent = -1;
	nodes[0].localMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f));
	nodes[0].globalMatrix = nodes[0].localMatrix;

	nodes[1].parent = 0;
	nodes[1].localMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 5.0f, 0.0f));
	nodes[1].globalMatrix = nodes[0].globalMatrix * nodes[1].localMatrix;

	brassica::GltfSkin skin;
	skin.name = "Armature";
	skin.joints = {0, 1};
	skin.inverseBindMatrices = {
		glm::translate(glm::mat4(1.0f), glm::vec3(-10.0f, 0.0f, 0.0f)),
		glm::translate(glm::mat4(1.0f), glm::vec3(-10.0f, -5.0f, 0.0f))
	};

	auto skinningMats = brassica::GltfAnimationEvaluator::ComputeSkinningMatrices(skin, nodes);
	REQUIRE(skinningMats.size() == 2);

	// Joint 0: global (10, 0, 0) * invBind (-10, 0, 0) = Identity
	CHECK(skinningMats[0] == glm::mat4(1.0f));

	// Joint 1: global (10, 5, 0) * invBind (-10, -5, 0) = Identity
	CHECK(skinningMats[1] == glm::mat4(1.0f));
}

TEST_CASE("glTF Model Instance Data Verification") {
	brassica::GltfModel model;
	CHECK(model.GetTotalVertices() == 0);
	CHECK(model.GetTotalMeshlets() == 0);
	CHECK(model.GetNodes().empty());
	CHECK(model.GetPrimitives().empty());
	CHECK(!model.GetVertexBuffer());
}
