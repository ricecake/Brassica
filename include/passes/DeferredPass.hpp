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

	class DeferredPass: public RenderPass {
	public:
		DeferredPass(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~DeferredPass() override;

		void InitPipeline(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		[[nodiscard]] vk::DescriptorSet GetGBufferDescriptorSet(uint32_t activeFrame) const {
			return gbufferDescriptorSets[activeFrame % FRAME_OVERLAP];
		}

		[[nodiscard]] vk::Sampler GetSampler() const { return sampler; }

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;

		static constexpr uint32_t FRAME_OVERLAP = 2;
		vk::DescriptorSetLayout   gbufferSetLayout{nullptr};
		vk::DescriptorPool        descriptorPool{nullptr};
		vk::DescriptorSet         gbufferDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::Sampler               sampler{nullptr};

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
	};

	// Reads its four sampled inputs and the TLAS by key from the registry rather than through
	// the old fg::FrameGraphPassResources lookup -- see the registry pointer below. Modify<Swapchain>
	// (not Read+separate-Write) matches the real access: the fullscreen triangle overwrites every
	// pixel, so this is a read-modify-write of a single physical image, not a produce-a-new-one.
	struct DeferredNode {
		using Resources = graph::Declares<
			graph::Read<GBufferPosition>,
			graph::Read<GBufferNormal>,
			graph::Read<GBufferAlbedo>,
			graph::Read<GradientBackground>,
			graph::Read<TerrainTLAS>,
			graph::Transform<Swapchain, LitSwapchain>>;

		DeferredPass*                    pass;
		graph::PhysicalResourceRegistry* registry;
		vk::Extent2D                     extent;
		vk::Format                       swapchainFormat;
		vk::DescriptorSet                globalDescriptorSet;
		uint32_t                         activeFrame;
		vk::ImageView                    clipmapImageView;
		vk::Sampler                      clipmapSampler;
		TerrainPushConstants             pushConstants;

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(extent.width, extent.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));

			auto posTex = registry->GetTexture<GBufferPosition>();
			auto normTex = registry->GetTexture<GBufferNormal>();
			auto albTex = registry->GetTexture<GBufferAlbedo>();
			auto bgTex = registry->GetTexture<GradientBackground>();
			auto tlasHandle = registry->GetAccelerationStructure<TerrainTLAS>();

			vk::DescriptorSet currentGbufferSet = pass->GetGBufferDescriptorSet(activeFrame);
			vk::Sampler       sampler = pass->GetSampler();

			std::array<vk::DescriptorImageInfo, 5> imageInfos{};
			imageInfos[0]
				.setSampler(sampler)
				.setImageView(posTex ? posTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imageInfos[1]
				.setSampler(sampler)
				.setImageView(normTex ? normTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imageInfos[2]
				.setSampler(sampler)
				.setImageView(albTex ? albTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imageInfos[3]
				.setSampler(sampler)
				.setImageView(bgTex ? bgTex->GetView() : nullptr)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

			// Same fallback the old code used: if no clipmap view/sampler was supplied, bind the
			// position texture as a harmless placeholder rather than writing a null descriptor.
			vk::Sampler   clipSampler = clipmapSampler ? clipmapSampler : sampler;
			vk::ImageView clipView = clipmapImageView ? clipmapImageView : (posTex ? posTex->GetView() : nullptr);
			imageInfos[4]
				.setSampler(clipSampler)
				.setImageView(clipView)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

			std::vector<vk::WriteDescriptorSet> descriptorWrites(5);
			for (uint32_t i = 0; i < 5; ++i) {
				descriptorWrites[i].setDstSet(currentGbufferSet);
				descriptorWrites[i].setDstBinding(i);
				descriptorWrites[i].setDstArrayElement(0);
				descriptorWrites[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[i].setImageInfo(imageInfos[i]);
			}

			// Fixes a real bug from the old fg-based path: it always wrote descriptorCount=1 for
			// the AS binding even when tlas was null, which is a VUID violation
			// (VkWriteDescriptorSet-descriptorType-02382 requires accelerationStructureCount to
			// match). Skip the write entirely until a TLAS actually exists (e.g. the very first
			// frame, before TerrainPass has built one).
			vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
			vk::AccelerationStructureKHR                   rawTlas = tlasHandle ? tlasHandle->Get() : nullptr;
			if (rawTlas) {
				// setAccelerationStructures takes an ArrayProxyNoTemporaries -- it needs a named
				// lvalue it can point at, not a temporary, hence rawTlas rather than inlining
				// tlasHandle->Get() here.
				asInfo.setAccelerationStructures(rawTlas);

				vk::WriteDescriptorSet asWrite{};
				asWrite.setDstSet(currentGbufferSet);
				asWrite.setDstBinding(5);
				asWrite.setDstArrayElement(0);
				asWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
				asWrite.setDescriptorCount(1);
				asWrite.setPNext(&asInfo);
				descriptorWrites.push_back(asWrite);
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
				currentGbufferSet,
				nullptr
			);

			vkCmd.pushConstants(
				pass->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(TerrainPushConstants),
				&pushConstants
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

} // namespace brassica
