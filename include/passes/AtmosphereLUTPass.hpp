#pragma once

#include <array>
#include <cstring>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ComputePass.hpp"
#include "passes/ResourceKeys.hpp"
#include "Shader.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	class AtmosphereLUTPass: public Pass {
	public:
		AtmosphereLUTPass(vk::Device device, ShaderWatcher* watcher = nullptr);
		~AtmosphereLUTPass() override;

		void InitPipeline(vk::Device device, ShaderWatcher* watcher = nullptr);

		void DestroyPipeline();

		// Regeneration throttle: both LUTs are only rebuilt when the atmosphere parameters
		// actually change, mirroring TerrainPass's camera-movement throttle for the TLAS. The
		// push-constant struct is a tightly packed POD used directly for GPU upload (see its
		// offsetof/size assertions in tests/test_atmosphere_lut.cpp), so a byte comparison is a
		// safe and exact "did anything change" check.
		[[nodiscard]] bool ShouldRegenerate(const AtmospherePushConstants& push) const {
			return !hasGenerated || std::memcmp(&push, &lastPush, sizeof(AtmospherePushConstants)) != 0;
		}

		void MarkRegenerated(const AtmospherePushConstants& push) {
			lastPush = push;
			hasGenerated = true;
		}

		[[nodiscard]] vk::Pipeline GetTransmittancePipeline() const { return transmittancePipeline; }

		[[nodiscard]] vk::PipelineLayout GetTransmittancePipelineLayout() const { return transmittancePipelineLayout; }

		[[nodiscard]] vk::DescriptorSet GetTransmittanceSet(uint32_t activeFrame) const {
			return transmittanceSets[activeFrame % FRAME_OVERLAP];
		}

		[[nodiscard]] vk::Pipeline GetMultiScatteringPipeline() const { return multiScatteringPipeline; }

		[[nodiscard]] vk::PipelineLayout GetMultiScatteringPipelineLayout() const {
			return multiScatteringPipelineLayout;
		}

		[[nodiscard]] vk::DescriptorSet GetMultiScatteringSet(uint32_t activeFrame) const {
			return multiScatteringSets[activeFrame % FRAME_OVERLAP];
		}

		[[nodiscard]] vk::Sampler GetSampler() const { return sampler; }

	private:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		ComputeShader transmittanceShader;
		ComputeShader multiScatteringShader;

		vk::DescriptorSetLayout transmittanceSetLayout{nullptr};
		vk::DescriptorSetLayout multiScatteringSetLayout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};

		vk::DescriptorSet transmittanceSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet multiScatteringSets[FRAME_OVERLAP]{nullptr, nullptr};

		vk::PipelineLayout transmittancePipelineLayout{nullptr};
		vk::PipelineLayout multiScatteringPipelineLayout{nullptr};

		vk::Pipeline transmittancePipeline{nullptr};
		vk::Pipeline multiScatteringPipeline{nullptr};

		vk::Sampler sampler{nullptr};

		AtmospherePushConstants lastPush{};
		bool                    hasGenerated{false};

		void CreateDescriptorResources(vk::Device dev);
		void CleanupDescriptorResources();
	};

	// Not wired into the live Frame -- nothing today consumes AtmosphereLUTPass's output, and
	// deciding to light-integrate it into DeferredNode is a separate feature decision, not part
	// of this migration. Ported and tested (tests/test_atmosphere_lut.cpp) so it compiles and
	// behaves correctly against the new model, ready for whoever wires it up next.
	struct TransmittanceLUTNode {
		using Resources = graph::Declares<graph::Create<TransmittanceLUT>>;

		AtmosphereLUTPass*               pass;
		graph::PhysicalResourceRegistry* registry;
		uint32_t                         activeFrame;
		AtmospherePushConstants          push;

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = pass->ShouldRegenerate(push)};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
			auto              tex = registry->GetTexture<TransmittanceLUT>();
			if (!tex) {
				return;
			}

			vk::DescriptorSet currentSet = pass->GetTransmittanceSet(activeFrame);

			vk::DescriptorImageInfo imageInfo{};
			imageInfo.setImageView(tex->GetView());
			imageInfo.setImageLayout(vk::ImageLayout::eGeneral);

			vk::WriteDescriptorSet descriptorWrite{};
			descriptorWrite.setDstSet(currentSet);
			descriptorWrite.setDstBinding(0);
			descriptorWrite.setDstArrayElement(0);
			descriptorWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
			descriptorWrite.setImageInfo(imageInfo);

			pass->GetDevice().updateDescriptorSets(descriptorWrite, nullptr);

			vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, pass->GetTransmittancePipeline());
			vkCmd.pushConstants(
				pass->GetTransmittancePipelineLayout(),
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(AtmospherePushConstants),
				&push
			);
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute,
				pass->GetTransmittancePipelineLayout(),
				0,
				currentSet,
				nullptr
			);

			vkCmd.dispatch((256 + 7) / 8, (64 + 7) / 8, 1);

			// Runs only when isActive (Graph::Compile culls inactive nodes before Execute), so
			// this unconditionally marks the throttle as caught up -- see
			// AtmosphereLUTPass::ShouldRegenerate.
			pass->MarkRegenerated(push);
		}
	};

	struct MultiScatteringLUTNode {
		using Resources = graph::Declares<graph::Read<TransmittanceLUT>, graph::Create<MultiScatteringLUT>>;

		AtmosphereLUTPass*               pass;
		graph::PhysicalResourceRegistry* registry;
		uint32_t                         activeFrame;
		AtmospherePushConstants          push;

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = pass->ShouldRegenerate(push)};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<MultiScatteringLUT>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(32, 32, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
			auto              transTex = registry->GetTexture<TransmittanceLUT>();
			auto              multiTex = registry->GetTexture<MultiScatteringLUT>();
			if (!transTex || !multiTex) {
				return;
			}

			vk::DescriptorSet currentSet = pass->GetMultiScatteringSet(activeFrame);

			std::array<vk::DescriptorImageInfo, 2> imageInfos{};
			imageInfos[0].setImageView(multiTex->GetView());
			imageInfos[0].setImageLayout(vk::ImageLayout::eGeneral);

			imageInfos[1].setSampler(pass->GetSampler());
			imageInfos[1].setImageView(transTex->GetView());
			imageInfos[1].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

			std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
			descriptorWrites[0].setDstSet(currentSet);
			descriptorWrites[0].setDstBinding(0);
			descriptorWrites[0].setDstArrayElement(0);
			descriptorWrites[0].setDescriptorType(vk::DescriptorType::eStorageImage);
			descriptorWrites[0].setImageInfo(imageInfos[0]);

			descriptorWrites[1].setDstSet(currentSet);
			descriptorWrites[1].setDstBinding(1);
			descriptorWrites[1].setDstArrayElement(0);
			descriptorWrites[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
			descriptorWrites[1].setImageInfo(imageInfos[1]);

			pass->GetDevice().updateDescriptorSets(descriptorWrites, nullptr);

			vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, pass->GetMultiScatteringPipeline());
			vkCmd.pushConstants(
				pass->GetMultiScatteringPipelineLayout(),
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(AtmospherePushConstants),
				&push
			);
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute,
				pass->GetMultiScatteringPipelineLayout(),
				0,
				currentSet,
				nullptr
			);

			vkCmd.dispatch(1, 1, 1);
		}
	};

} // namespace brassica
