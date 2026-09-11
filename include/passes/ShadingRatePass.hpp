#pragma once

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ComputePass.hpp"
#include "passes/ResourceKeys.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	class ShadingRatePass : public ComputePass {
	public:
		ShadingRatePass(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~ShadingRatePass() override;

		void UpdateDescriptors(
			vk::ImageView gBufferNormalView,
			vk::Sampler   gBufferNormalSampler,
			vk::ImageView gBufferDepthView,
			vk::Sampler   gBufferDepthSampler,
			vk::ImageView shadingRateMapView
		);

		[[nodiscard]] vk::DescriptorSet GetDescriptorSet() const { return descriptorSet; }

	private:
		ComputeShader compShader;

		vk::DescriptorSetLayout set1Layout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};
		vk::DescriptorSet       descriptorSet{nullptr};
	};

	struct ShadingRateNode {
		using Resources = graph::Declares<
			graph::Read<GBufferNormal>,
			graph::Read<GBufferDepth>,
			graph::Modify<ShadingRateMap>>;

		ShadingRatePass*                 pass;
		graph::PhysicalResourceRegistry* registry;
		vk::Extent2D                     extent;
		vk::DescriptorSet                globalDescriptorSet;
		vk::Sampler                      sampler;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};

			// Shading rate attachment texel size is 16x16, so map size is (width + 15)/16 x (height + 15)/16
			uint32_t mapWidth = (ctx.width + 15) / 16;
			uint32_t mapHeight = (ctx.height + 15) / 16;

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
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
					.key = graph::IdOf<ShadingRateMap>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ShadingRateAttachmentDesc(mapWidth, mapHeight, vk::Format::eR8Uint),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			auto normTex = registry->GetTexture<GBufferNormal>();
			auto depthTex = registry->GetTexture<GBufferDepth>();
			auto rateTex = registry->GetTexture<ShadingRateMap>();

			if (normTex && depthTex && rateTex) {
				pass->UpdateDescriptors(
					normTex->GetView(),
					sampler,
					depthTex->GetView(),
					sampler,
					rateTex->GetView()
				);
			}

			uint32_t mapWidth = (extent.width + 15) / 16;
			uint32_t mapHeight = (extent.height + 15) / 16;

			if (globalDescriptorSet) {
				vkCmd.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute,
					pass->GetPipelineLayout(),
					0,
					globalDescriptorSet,
					nullptr
				);
			}
			if (pass->GetDescriptorSet()) {
				vkCmd.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute,
					pass->GetPipelineLayout(),
					1,
					pass->GetDescriptorSet(),
					nullptr
				);
			}

			uint32_t groupX = (mapWidth + 15) / 16;
			uint32_t groupY = (mapHeight + 15) / 16;
			pass->Dispatch(vkCmd, groupX, groupY, 1);
		}
	};

} // namespace brassica
