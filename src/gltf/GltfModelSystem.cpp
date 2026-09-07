#include "gltf/GltfModelSystem.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace brassica {

	GltfModelSystem::~GltfModelSystem() {
		Cleanup();
	}

	void GltfModelSystem::Init(
		vk::Instance instance,
		vk::Device dev,
		VmaAllocator alloc,
		vk::DescriptorSetLayout globalSet0Layout,
		GltfTextureManager* texMgr,
		ShaderWatcher* watcher
	) {
		device = dev;
		allocator = alloc;
		textureManager = texMgr;

		computePass = std::make_unique<GltfComputePass>(device, watcher);

		vk::DescriptorSetLayout texSetLayout = textureManager ? textureManager->GetDescriptorSetLayout() : nullptr;
		renderPass = std::make_unique<GltfRenderPass>(instance, device, globalSet0Layout, nullptr, texSetLayout, watcher);

		CreateDescriptorPool();
	}

	void GltfModelSystem::CreateDescriptorPool() {
		std::array<vk::DescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(128);
		poolSizes[1].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(128);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		poolInfo.setMaxSets(128);
		poolInfo.setPoolSizes(poolSizes);

		descriptorPool = device.createDescriptorPool(poolInfo);
	}

	void GltfModelSystem::Cleanup() {
		if (device) {
			device.waitIdle();

			for (auto& [model, resArray] : modelResources) {
				for (auto& res : resArray) {
					if (res.instanceBuffer) vmaDestroyBuffer(allocator, res.instanceBuffer, res.instanceAllocation);
					if (res.jointMatrixBuffer) vmaDestroyBuffer(allocator, res.jointMatrixBuffer, res.jointMatrixAllocation);
					if (res.indirectDrawBuffer) vmaDestroyBuffer(allocator, res.indirectDrawBuffer, res.indirectDrawAllocation);
					if (res.visibleInstanceBuffer) vmaDestroyBuffer(allocator, res.visibleInstanceBuffer, res.visibleInstanceAllocation);
					if (res.drawCountBuffer) vmaDestroyBuffer(allocator, res.drawCountBuffer, res.drawCountAllocation);

					res = PerModelFrameResources{};
				}
			}
			modelResources.clear();

			if (descriptorPool) {
				device.destroyDescriptorPool(descriptorPool);
				descriptorPool = nullptr;
			}
		}

		modelCache.clear();
		computePass.reset();
		renderPass.reset();
	}

	std::shared_ptr<GltfModel> GltfModelSystem::LoadModel(const std::string& filepath) {
		auto it = modelCache.find(filepath);
		if (it != modelCache.end()) {
			return it->second;
		}

		auto model = std::make_shared<GltfModel>();
		if (model->LoadFromFile(device, allocator, textureManager, filepath)) {
			modelCache[filepath] = model;
			return model;
		}

		return nullptr;
	}

	void GltfModelSystem::Update(entt::registry& registry, const CameraData& camera, float deltaTime) {
		// 1. Update Avatar position and orientation in front of camera
		auto avatarView = registry.view<AvatarComponent, TransformComponent>();
		for (auto entity : avatarView) {
			auto& avatar = avatarView.get<AvatarComponent>(entity);
			auto& transform = avatarView.get<TransformComponent>(entity);

			glm::vec3 forward = camera.GetForward();
			glm::vec3 right = camera.GetRight();
			glm::vec3 up = camera.GetUp();

			glm::vec3 targetPos = camera.position + forward * avatar.offset.z + right * avatar.offset.x + up * avatar.offset.y;
			transform.position = targetPos;

			glm::quat camOrientation = camera.orientation;
			glm::quat rotOffset = glm::quat(glm::radians(avatar.rotationOffset));
			transform.rotation = camOrientation * rotOffset;
		}

		// 2. Update glTF Animations
		auto animView = registry.view<GltfAnimationComponent, GltfModelComponent>();
		for (auto entity : animView) {
			auto& animComp = animView.get<GltfAnimationComponent>(entity);
			auto& modelComp = animView.get<GltfModelComponent>(entity);

			if (animComp.playing && modelComp.model && modelComp.visible) {
				animComp.time += deltaTime * animComp.speed;
				const auto& animations = modelComp.model->GetAnimations();

				if (animComp.animationIndex >= 0 && animComp.animationIndex < static_cast<int32_t>(animations.size())) {
					const auto& anim = animations[animComp.animationIndex];
					if (anim.duration > 0.0f) {
						if (animComp.loop) {
							animComp.time = std::fmod(animComp.time, anim.duration);
						} else {
							animComp.time = std::min(animComp.time, anim.duration);
						}
					}
					GltfAnimationEvaluator::EvaluateAnimation(anim, animComp.time, modelComp.model->GetNodes());
					GltfAnimationEvaluator::ComputeGlobalTransforms(modelComp.model->GetNodes());
				}
			}
		}
	}

	void GltfModelSystem::EnsureBufferCapacity(
		PerModelFrameResources& res,
		size_t instanceCount,
		size_t jointCount
	) {
		auto createOrRealloc = [this](
			size_t requiredBytes,
			vk::BufferUsageFlags usage,
			vk::Buffer& outBuffer,
			VmaAllocation& outAlloc,
			size_t& outCapacity
		) {
			if (requiredBytes == 0) requiredBytes = 64;

			if (outBuffer && outCapacity >= requiredBytes) {
				return;
			}

			if (outBuffer) {
				vmaDestroyBuffer(allocator, outBuffer, outAlloc);
				outBuffer = nullptr;
				outAlloc = VK_NULL_HANDLE;
			}

			size_t newCapacity = std::max(requiredBytes, outCapacity * 2);
			if (newCapacity < 256) newCapacity = 256;

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = newCapacity;
			bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer rawBuf = VK_NULL_HANDLE;
			if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &rawBuf, &outAlloc, nullptr) == VK_SUCCESS) {
				outBuffer = rawBuf;
				outCapacity = newCapacity;
			} else {
				spdlog::error("GltfModelSystem failed to allocate GPU storage buffer of size {}", newCapacity);
			}
		};

		vk::BufferUsageFlags ssboUsage = vk::BufferUsageFlagBits::eStorageBuffer;

		createOrRealloc(instanceCount * sizeof(InstanceGPUData), ssboUsage, res.instanceBuffer, res.instanceAllocation, res.instanceBufferSize);
		createOrRealloc(std::max(size_t(1), jointCount) * sizeof(glm::mat4), ssboUsage, res.jointMatrixBuffer, res.jointMatrixAllocation, res.jointMatrixBufferSize);
		createOrRealloc(instanceCount * sizeof(VkDrawMeshTasksIndirectCommandEXT), ssboUsage | vk::BufferUsageFlagBits::eIndirectBuffer, res.indirectDrawBuffer, res.indirectDrawAllocation, res.indirectDrawBufferSize);
		createOrRealloc(instanceCount * sizeof(VisibleInstanceGPUData), ssboUsage, res.visibleInstanceBuffer, res.visibleInstanceAllocation, res.visibleInstanceBufferSize);

		if (!res.drawCountBuffer) {
			size_t dummyCap = 0;
			createOrRealloc(sizeof(uint32_t), ssboUsage | vk::BufferUsageFlagBits::eTransferDst, res.drawCountBuffer, res.drawCountAllocation, dummyCap);
		}
	}

	void GltfModelSystem::Render(
		entt::registry& registry,
		FrameGraph& fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D extent,
		vk::DescriptorSet globalDescriptorSet,
		const CameraData& camera,
		uint32_t activeFrame
	) {
		std::unordered_map<GltfModel*, std::vector<entt::entity>> modelEntities;
		auto view = registry.view<TransformComponent, GltfModelComponent>();

		for (auto entity : view) {
			const auto& modelComp = view.get<GltfModelComponent>(entity);
			if (modelComp.visible && modelComp.model) {
				modelEntities[modelComp.model.get()].push_back(entity);
			}
		}

		for (auto& [model, entities] : modelEntities) {
			if (!model || entities.empty()) continue;

			PerModelFrameResources& res = modelResources[model][activeFrame % FRAME_OVERLAP];

			std::vector<InstanceGPUData> instanceData;
			std::vector<VisibleInstanceGPUData> visibleData;
			std::vector<glm::mat4> jointMatricesData;

			for (auto entity : entities) {
				const auto& transformComp = registry.get<TransformComponent>(entity);
				glm::mat4 entityWorld = transformComp.GetWorldMatrix();

				uint32_t jointOffset = static_cast<uint32_t>(jointMatricesData.size());
				uint32_t jointCount = 0;

				if (!model->GetSkins().empty()) {
					const auto& skin = model->GetSkins()[0];
					auto skinMats = GltfAnimationEvaluator::ComputeSkinningMatrices(skin, model->GetNodes());
					jointMatricesData.insert(jointMatricesData.end(), skinMats.begin(), skinMats.end());
					jointCount = static_cast<uint32_t>(skinMats.size());
				}

				const auto& primitives = model->GetPrimitives();
				for (size_t i = 0; i < primitives.size(); ++i) {
					InstanceGPUData inst{};
					inst.worldMatrix = entityWorld;
					inst.modelIndex = 0;
					inst.primitiveIndex = static_cast<uint32_t>(i);
					inst.jointMatrixOffset = jointOffset;
					inst.jointCount = jointCount;

					instanceData.push_back(inst);

					VisibleInstanceGPUData vis{};
					vis.worldMatrix = entityWorld;
					vis.primitiveIndex = static_cast<uint32_t>(i);
					vis.materialIndex = primitives[i].materialIndex;
					vis.jointMatrixOffset = jointOffset;
					vis.jointCount = jointCount;

					visibleData.push_back(vis);
				}
			}

			if (instanceData.empty()) continue;

			EnsureBufferCapacity(res, instanceData.size(), jointMatricesData.size());

			// Upload Instance Data
			VmaAllocationInfo instAllocInfo{};
			vmaGetAllocationInfo(allocator, res.instanceAllocation, &instAllocInfo);
			if (instAllocInfo.pMappedData) {
				std::memcpy(instAllocInfo.pMappedData, instanceData.data(), instanceData.size() * sizeof(InstanceGPUData));
			}

			// Upload Visible Instance Data
			VmaAllocationInfo visAllocInfo{};
			vmaGetAllocationInfo(allocator, res.visibleInstanceAllocation, &visAllocInfo);
			if (visAllocInfo.pMappedData) {
				std::memcpy(visAllocInfo.pMappedData, visibleData.data(), visibleData.size() * sizeof(VisibleInstanceGPUData));
			}

			// Upload Joint Matrices Data
			if (!jointMatricesData.empty()) {
				VmaAllocationInfo jointAllocInfo{};
				vmaGetAllocationInfo(allocator, res.jointMatrixAllocation, &jointAllocInfo);
				if (jointAllocInfo.pMappedData) {
					std::memcpy(jointAllocInfo.pMappedData, jointMatricesData.data(), jointMatricesData.size() * sizeof(glm::mat4));
				}
			}

			// Allocate/Update Model Descriptor Set (Set 1 in Render Pass)
			if (!res.modelDescriptorSet) {
				vk::DescriptorSetLayout layout = renderPass->GetModelSetLayout();
				vk::DescriptorSetAllocateInfo allocInfo{};
				allocInfo.setDescriptorPool(descriptorPool);
				allocInfo.setSetLayouts(layout);
				res.modelDescriptorSet = device.allocateDescriptorSets(allocInfo).front();
			}

			std::array<vk::DescriptorBufferInfo, 8> renderBufferInfos{};
			renderBufferInfos[0].setBuffer(model->GetVertexBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[1].setBuffer(model->GetMeshletBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[2].setBuffer(model->GetMeshletVerticesBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[3].setBuffer(model->GetMeshletTrianglesBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[4].setBuffer(model->GetPrimitiveBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[5].setBuffer(model->GetMaterialBuffer()).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[6].setBuffer(res.visibleInstanceBuffer).setOffset(0).setRange(VK_WHOLE_SIZE);
			renderBufferInfos[7].setBuffer(res.jointMatrixBuffer).setOffset(0).setRange(VK_WHOLE_SIZE);

			std::array<vk::WriteDescriptorSet, 8> renderWrites{};
			for (uint32_t b = 0; b < 8; ++b) {
				renderWrites[b].setDstSet(res.modelDescriptorSet)
					.setDstBinding(b)
					.setDescriptorType(vk::DescriptorType::eStorageBuffer)
					.setBufferInfo(renderBufferInfos[b]);
			}
			device.updateDescriptorSets(renderWrites, nullptr);

			// Register Render Pass
			GltfRenderPushConstants renderPush{};
			renderPush.viewProj = camera.viewProjMatrix;

			vk::DescriptorSet texSet = textureManager ? textureManager->GetDescriptorSet() : nullptr;

			renderPass->RegisterPass(
				fg,
				blackboard,
				extent,
				globalDescriptorSet,
				res.modelDescriptorSet,
				texSet,
				res.indirectDrawBuffer,
				res.drawCountBuffer,
				static_cast<uint32_t>(instanceData.size()),
				renderPush
			);
		}
	}

} // namespace brassica
