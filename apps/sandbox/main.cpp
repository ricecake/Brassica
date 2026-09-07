#include "Engine.hpp"
#include <brassica.hpp>
#include <filesystem>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
	brassica::InitializeCore();

	brassica::EngineOptions options = brassica::EngineOptions::FromArgs(argc, argv);

	brassica::Engine engine;
	engine.Init(options);

	std::string modelPath = "assets/models/avatar.gltf";
	if (!std::filesystem::exists(modelPath)) {
		if (std::filesystem::exists("bin/assets/models/avatar.gltf")) {
			modelPath = "bin/assets/models/avatar.gltf";
		} else if (std::filesystem::exists(std::string(BRASSICA_BUILD_DIR) + "/bin/assets/models/avatar.gltf")) {
			modelPath = std::string(BRASSICA_BUILD_DIR) + "/bin/assets/models/avatar.gltf";
		}
	}

	auto avatarModel = engine.GetGltfModelSystem().LoadModel(modelPath);
	if (avatarModel) {
		auto avatarEntity = engine.GetRegistry().create();

		brassica::TransformComponent transform{};
		transform.scale = glm::vec3(0.5f);

		brassica::GltfModelComponent modelComp{};
		modelComp.model = avatarModel;
		modelComp.visible = true;

		brassica::AvatarComponent avatarComp{};
		avatarComp.offset = glm::vec3(0.0f, -0.4f, 1.8f);
		avatarComp.rotationOffset = glm::vec3(0.0f, 180.0f, 0.0f);

		engine.GetRegistry().emplace<brassica::TransformComponent>(avatarEntity, transform);
		engine.GetRegistry().emplace<brassica::GltfModelComponent>(avatarEntity, modelComp);
		engine.GetRegistry().emplace<brassica::AvatarComponent>(avatarEntity, avatarComp);

		spdlog::info("Spawned avatar entity in EnTT registry with model {}", modelPath);
	} else {
		spdlog::warn("Failed to load avatar model from {}", modelPath);
	}

	engine.Run();
	engine.Cleanup();

	return 0;
}
