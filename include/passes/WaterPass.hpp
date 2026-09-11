#pragma once

#include <array>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPass.hpp"
#include "passes/ResourceKeys.hpp"
#include "passes/TerrainPass.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	class WaterPass: public RenderPass {
	public:
		WaterPass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~WaterPass() override;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		[[nodiscard]] vk::DescriptorSet GetWaterDescriptorSet(uint32_t activeFrame) const {
			return waterDescriptorSets[activeFrame % FRAME_OVERLAP];
		}

		[[nodiscard]] vk::Sampler GetSampler() const { return sampler; }

		[[nodiscard]] const DispatchLoaderDynamic& GetDls() const { return dls; }

	private:
		DispatchLoaderDynamic dls;

		MeshShader     meshShader;
		FragmentShader fragShader;

		static constexpr uint32_t FRAME_OVERLAP = 2;
		vk::DescriptorSetLayout   waterSetLayout{nullptr};
		vk::DescriptorPool        descriptorPool{nullptr};
		vk::DescriptorSet         waterDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::Sampler               sampler{nullptr};

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
	};

	struct WaterNode {
		using Resources = graph::Declares<
			graph::Read<GBufferPosition>,
			graph::Read<GBufferDepth>,
			graph::Read<GBufferAlbedo>,
			graph::Modify<LitSwapchain>>;

		WaterPass*                       pass;
		graph::PhysicalResourceRegistry* registry;
		vk::Extent2D                     extent;
		vk::Format                       swapchainFormat;
		vk::DescriptorSet                globalDescriptorSet;
		uint32_t                         activeFrame;
		vk::ImageView                    clipmapImageView;
		vk::Sampler                      clipmapSampler;
		TerrainPushConstants             pushConstants;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<LitSwapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(extent.width, extent.height, swapchainFormat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Read,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			auto posTex = registry->GetTexture<GBufferPosition>();
			auto depthTex = registry->GetTexture<GBufferDepth>();
			auto albTex = registry->GetTexture<GBufferAlbedo>();

			vk::DescriptorSet currentWaterSet = pass->GetWaterDescriptorSet(activeFrame);
			vk::Sampler       sampler = pass->GetSampler();

			std::array<vk::DescriptorImageInfo, 4> imageInfos{};
			imageInfos[0]
				.setSampler(sampler)
				.setImageView(posTex ? posTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imageInfos[1]
				.setSampler(sampler)
				.setImageView(depthTex ? depthTex->GetView() : nullptr)
				.setImageLayout(depthTex ? depthTex->GetCurrentLayout() : vk::ImageLayout::eShaderReadOnlyOptimal);
			imageInfos[2]
				.setSampler(sampler)
				.setImageView(albTex ? albTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

			vk::Sampler   clipSampler = clipmapSampler ? clipmapSampler : sampler;
			vk::ImageView clipView = clipmapImageView ? clipmapImageView : (posTex ? posTex->GetView() : nullptr);
			imageInfos[3]
				.setSampler(clipSampler)
				.setImageView(clipView)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

			std::vector<vk::WriteDescriptorSet> descriptorWrites(4);
			for (uint32_t i = 0; i < 4; ++i) {
				descriptorWrites[i].setDstSet(currentWaterSet);
				descriptorWrites[i].setDstBinding(i);
				descriptorWrites[i].setDstArrayElement(0);
				descriptorWrites[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[i].setImageInfo(imageInfos[i]);
			}

			pass->GetDevice().updateDescriptorSets(descriptorWrites, nullptr);

			pass->BindForDraw(vkCmd, extent);

			if (globalDescriptorSet) {
				vkCmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pass->GetPipelineLayout(),
					0,
					globalDescriptorSet,
					nullptr
				);
			}
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pass->GetPipelineLayout(),
				1,
				currentWaterSet,
				nullptr
			);

			vkCmd.pushConstants(
				pass->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(TerrainPushConstants),
				&pushConstants
			);

			pass->DrawMeshTasksEXT(vkCmd, 32, 32, 1, pass->GetDls());
		}
	};

} // namespace brassica
