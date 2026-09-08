#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/AtmosphereLUTPass.hpp"
#include "passes/RenderPass.hpp"
#include "Shader.hpp"
#include "types/Light.hpp"

namespace brassica {

	class ShaderWatcher;

	struct AtmosphereSkyPushConstants {
		alignas(16) glm::mat4 invViewProj{1.0f};
		alignas(16) glm::vec4 cameraPosAndScale{0.0f, 0.0f, 0.0f, 1.0f};     // xyz = cameraPos, w = worldScale
		alignas(16) glm::vec4 sunDirAndAureole{0.0f, 1.0f, 0.0f, 0.5f};      // xyz = sunDir, w = sunAureole
		alignas(16) glm::vec4 moonDirAndCirrus{0.0f, -1.0f, 0.0f, 0.3f};     // xyz = moonDir, w = cirrus
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f}; // xyz = sunRadiance, w = skyExposure
	};

	static_assert(sizeof(AtmosphereSkyPushConstants) == 128, "AtmosphereSkyPushConstants must be 128 bytes");

	struct AtmosphereSkyPassData {
		FrameGraphResource background;
	};

	class AtmosphereSkyPass: public RenderPass {
	public:
		AtmosphereSkyPass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~AtmosphereSkyPass() override;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		AtmosphereSkyPassData RegisterPass(
			FrameGraph&                      fg,
			FrameGraphBlackboard&            blackboard,
			vk::Extent2D                     extent,
			vk::DescriptorSet                globalDescriptorSet,
			uint32_t                         activeFrame,
			const AtmosphereSkyPushConstants& push
		);

		void DestroyPipeline();

	private:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		vk::DispatchLoaderDynamic dls;

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		vk::DescriptorSetLayout skySetLayout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};
		vk::DescriptorSet       skySets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::Sampler             sampler{nullptr};

		void CreateDescriptorResources(vk::Device dev);
		void CleanupDescriptorResources();
	};

} // namespace brassica
