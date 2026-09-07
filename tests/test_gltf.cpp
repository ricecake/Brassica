#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/epsilon.hpp>
#include <vector>
#include <memory>
#include <entt/entt.hpp>

#include "ecs/Components.hpp"
#include "gltf/GltfModel.hpp"
#include "gltf/GltfAnimation.hpp"
#include "gltf/GltfTextureManager.hpp"
#include "gltf/GltfModelSystem.hpp"
#include "passes/GltfComputePass.hpp"
#include "passes/GltfRenderPass.hpp"
#include "types/CameraData.hpp"

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

TEST_CASE("TransformComponent World Matrix Calculation") {
	brassica::TransformComponent transform;
	transform.position = glm::vec3(5.0f, -2.0f, 10.0f);
	transform.scale = glm::vec3(2.0f, 2.0f, 2.0f);
	transform.SetEulerAngles(glm::vec3(0.0f, 0.0f, 0.0f));

	glm::mat4 worldMatrix = transform.GetWorldMatrix();

	CHECK(worldMatrix[3].x == doctest::Approx(5.0f));
	CHECK(worldMatrix[3].y == doctest::Approx(-2.0f));
	CHECK(worldMatrix[3].z == doctest::Approx(10.0f));

	CHECK(worldMatrix[0][0] == doctest::Approx(2.0f));
	CHECK(worldMatrix[1][1] == doctest::Approx(2.0f));
	CHECK(worldMatrix[2][2] == doctest::Approx(2.0f));
}

TEST_CASE("EnTT Registry Entity Model and Avatar Tracking") {
	entt::registry registry;

	auto entity = registry.create();

	brassica::TransformComponent transform{};
	transform.position = glm::vec3(0.0f, 1.0f, 0.0f);

	brassica::GltfModelComponent modelComp{};
	modelComp.model = std::make_shared<brassica::GltfModel>();
	modelComp.visible = true;

	brassica::AvatarComponent avatarComp{};
	avatarComp.offset = glm::vec3(0.0f, -0.5f, 2.0f);

	registry.emplace<brassica::TransformComponent>(entity, transform);
	registry.emplace<brassica::GltfModelComponent>(entity, modelComp);
	registry.emplace<brassica::AvatarComponent>(entity, avatarComp);

	CHECK(registry.valid(entity));
	CHECK(registry.all_of<brassica::TransformComponent, brassica::GltfModelComponent, brassica::AvatarComponent>(entity));

	auto view = registry.view<brassica::AvatarComponent, brassica::TransformComponent>();
	size_t count = 0;
	for (auto e : view) {
		count++;
		const auto& avatar = view.get<brassica::AvatarComponent>(e);
		CHECK(avatar.offset.z == doctest::Approx(2.0f));
	}
	CHECK(count == 1);
}

TEST_CASE("Avatar Position Calculation Relative to Camera") {
	brassica::CameraData camera;
	camera.position = glm::vec3(10.0f, 0.0f, 0.0f);
	camera.yaw = -1.5707963f; // ~ -90 deg looking along +X in Vulkan camera orientation
	camera.pitch = 0.0f;
	camera.roll = 0.0f;
	camera.UpdateMatrices(16.0f / 9.0f);

	brassica::AvatarComponent avatarComp;
	avatarComp.offset = glm::vec3(0.0f, 0.0f, 2.0f); // 2 units in front of camera

	glm::vec3 forward = camera.GetForward();
	glm::vec3 targetPos = camera.position + forward * avatarComp.offset.z;

	CHECK(targetPos.x == doctest::Approx(12.0f));
	CHECK(targetPos.y == doctest::Approx(0.0f));
	CHECK(targetPos.z == doctest::Approx(0.0f));
}
