#pragma once

#include <array>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct CloudWeatherBakePushConstants {
		std::uint32_t outWeatherMapIdx{0};
		std::uint32_t outMinMaxMapIdx{0};
		float uWorldScale{1.0f};
		float uCloudCoverage{0.35f};
		float uCloudThickness{1500.0f};
		float uTime{0.0f};
	};

	struct Cloud3DVolumeBakePushConstants {
		std::uint32_t outVolumeIdx{0};
	};

	struct CloudBoundingPushConstants {
		std::uint32_t depthTextureIdx{0};
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t outBoundingMapIdx{0};
		float uCloudMaxRayDistance{175000.0f};
		float uRenderScale{1.0f};
		std::int32_t uFrameIndex{0};
	};

	struct CloudShadowBakePushConstants {
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t outShadowMapIdx{0};
		alignas(16) glm::mat4 u_invLightSpaceMatrix{1.0f};
		alignas(16) glm::vec3 u_primaryLightDir{0.0f, 1.0f, 0.0f};
		alignas(4)  std::int32_t u_frameIndex{0};
	};

	struct CloudTileSchedulerPushConstants {
		std::uint32_t boundingMapIdx{0};
		std::uint32_t outErrorMapIdx{0};
		std::int32_t uPass{0};
		float uPriorityErrorWeight{2.5f};
		float uPriorityGradWeight{2.0f};
		float uPriorityAgeWeight{0.05f};
		float uPriorityNeighborErrorWeight{1.5f};
		float uPriorityNeighborGradWeight{1.0f};
		float uPriorityThreshold{0.05f};
	};

	struct CloudRenderPushConstants {
		std::uint32_t depthTextureIdx{0};
		std::uint32_t boundingTextureIdx{0};
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t weatherTextureIdx{0};
		std::uint32_t volume3DIdx{0};
		std::uint32_t outPackedColorIdx{0};
		std::uint32_t outPackedDepthIdx{0};
		std::uint32_t outPackedVelocityIdx{0};
		std::uint32_t outErrorMapIdx{0};
		float uDeltaTime{0.016f};
		float uCloudMaxRayDistance{175000.0f};
		float uRenderScale{1.0f};
		std::int32_t uCloudMinSamples{32};
		std::int32_t uCloudMaxSamples{96};
		float uCloudExtinction{0.372f};
		alignas(16) glm::vec3 uCloudExtinctionColor{1.0f, 1.0f, 1.0f};
		alignas(16) glm::vec3 uCloudAlbedo{0.85f, 0.85f, 0.85f};
		alignas(16) glm::vec3 cloudColorUniform{1.0f, 1.0f, 1.0f};
	};

	struct CloudTemporalPushConstants {
		std::uint32_t packedFrameIdx{0};
		std::uint32_t packedDepthIdx{0};
		std::uint32_t packedVelocityIdx{0};
		std::uint32_t historyFrameIdx{0};
		std::uint32_t historyCloudDepthIdx{0};
		std::uint32_t historyMomentsIdx{0};
		std::uint32_t boundingMapIdx{0};
		std::uint32_t outColorIdx{0};
		std::uint32_t outDepthIdx{0};
		std::uint32_t outMomentsIdx{0};
		std::uint32_t outErrorMapIdx{0};
		float uCloudTemporalGamma{1.1f};
		float uCloudMaxHistoryLength{32.0f};
		float uCloudMaxRayDistance{175000.0f};
		float uRenderScale{1.0f};
		std::int32_t uEnableTemporal{1};
		std::int32_t uHasHistory{0};
		std::int32_t uUseTileQueue{1};
		float uDeltaTime{0.016f};
	};

	struct CloudSpatialFilterPushConstants {
		std::uint32_t cloudColorIdx{0};
		std::uint32_t cloudDepthIdx{0};
		std::uint32_t cloudMomentsIdx{0};
		std::uint32_t errorMapIdx{0};
		std::uint32_t outFilteredColorIdx{0};
		std::int32_t uStepSize{1};
		std::int32_t uPassIndex{0};
		float uCloudPhiLuma{20.0f};
		float uCloudPhiDensity{0.05f};
		float uCloudPhiDepth{1.5f};
		float uCloudSvgfHistoryBoost{4.0f};
		float uCloudSvgfHistoryThreshold{10.0f};
	};

	// Boidish Cloud System Nodes
	struct CloudBakeNode : render::NodeRegistrar<CloudBakeNode> {
		using Resources = graph::Declares<
			graph::Create<CloudWeatherTexture>,
			graph::Create<CloudWeatherMinMaxTexture>,
			graph::Create<Cloud3DVolumeTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader weatherShader;
		ComputeShader volumeShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			weatherShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_weather_bake.comp");
			volumeShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_3d_volume_bake.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&weatherShader);
			watcher.RegisterShader(&volumeShader);
		}

		void Destroy(vk::Device device) {
			weatherShader.Destroy(device);
			volumeShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudWeatherTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(1024, 1024, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudWeatherMinMaxTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(1024, 1024, vk::Format::eR16G16Sfloat),
				}
			);
			auto volumeDesc = graph::ComputeStorageImageDesc(128, 128, vk::Format::eR16G16B16A16Sfloat);
			volumeDesc.kind = graph::ResourceDesc::Kind::Image3D;
			volumeDesc.depth = 128;
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Cloud3DVolumeTexture>(),
					.access = graph::AccessKind::Write,
					.desc = volumeDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			CloudWeatherBakePushConstants pushW{
				.outWeatherMapIdx = ctx.Index<CloudWeatherTexture>(),
				.outMinMaxMapIdx = ctx.Index<CloudWeatherMinMaxTexture>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesW{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudWeatherBakePushConstants)}
			};

			render::ComputePipelineRequest reqW{
				.shader = &weatherShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesW
			};
			render::ResolvedPipeline resW = pipelineLibrary->ResolveCached(reqW);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resW.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resW.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resW.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resW.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudWeatherBakePushConstants), &pushW);
			vkCmd.dispatch(64, 64, 1);

			Cloud3DVolumeBakePushConstants pushV{
				.outVolumeIdx = ctx.Index<Cloud3DVolumeTexture>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesV{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(Cloud3DVolumeBakePushConstants)}
			};

			render::ComputePipelineRequest reqV{
				.shader = &volumeShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesV
			};
			render::ResolvedPipeline resV = pipelineLibrary->ResolveCached(reqV);
			if (resV.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resV.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resV.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(resV.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(Cloud3DVolumeBakePushConstants), &pushV);
			vkCmd.dispatch(32, 32, 32);
		}
	};

	struct CloudBoundingNode : render::NodeRegistrar<CloudBoundingNode> {
		using Resources = graph::Declares<
			graph::Read<GBufferDepth>,
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Create<CloudBoundingTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_bounding.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudBoundingTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			CloudBoundingPushConstants pushB{
				.depthTextureIdx = ctx.Index<GBufferDepth>(),
				.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>(),
				.outBoundingMapIdx = ctx.Index<CloudBoundingTexture>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesB{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudBoundingPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesB
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudBoundingPushConstants), &pushB);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
		}
	};

	struct CloudShadowBakeNode : render::NodeRegistrar<CloudShadowBakeNode> {
		using Resources = graph::Declares<
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Create<CloudShadowMap>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_shadow_bake.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& /*ctx*/) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			auto shadowDesc = graph::ComputeStorageImageDesc(512, 512, vk::Format::eR16Sfloat);
			shadowDesc.layers = 8;
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudShadowMap>(),
					.access = graph::AccessKind::Write,
					.desc = shadowDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			CloudShadowBakePushConstants pushS{
				.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>(),
				.outShadowMapIdx = ctx.Index<CloudShadowMap>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesS{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudShadowBakePushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesS
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudShadowBakePushConstants), &pushS);
			vkCmd.dispatch(64, 64, 1);
		}
	};

	struct CloudTileSchedulerNode : render::NodeRegistrar<CloudTileSchedulerNode> {
		using Resources = graph::Declares<
			graph::Read<CloudBoundingTexture>,
			graph::Create<CloudErrorMap>,
			graph::Create<CloudTileQueueSSBO>,
			graph::Create<CloudIndirectDispatchSSBO>>;

		static constexpr graph::Phase kPhase = SubPhase::GIComput;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;
		vk::DescriptorSetLayout cloudSsboLayout{nullptr};
		vk::DescriptorPool      cloudSsboPool{nullptr};
		vk::DescriptorSet       cloudSsboSet{nullptr};
		vk::Device              m_device{nullptr};

		void Init(const render::NodeServices& services) {
			m_device = services.device;
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_tile_scheduler.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			std::array<vk::DescriptorSetLayoutBinding, 2> ssboBindings{};
			ssboBindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
			ssboBindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(ssboBindings);
			cloudSsboLayout = services.device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(1);
			cloudSsboPool = services.device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(cloudSsboPool);
			allocInfo.setSetLayouts(cloudSsboLayout);
			cloudSsboSet = services.device.allocateDescriptorSets(allocInfo).front();
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
			if (cloudSsboPool) device.destroyDescriptorPool(cloudSsboPool);
			if (cloudSsboLayout) device.destroyDescriptorSetLayout(cloudSsboLayout);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudErrorMap>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(tileCols, tileRows, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTileQueueSSBO>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc((tileCols * tileRows + 16) * sizeof(uint32_t) * 2),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(3 * sizeof(uint32_t));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudIndirectDispatchSSBO>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					auto tileQueueBuf = registry->GetBuffer<CloudTileQueueSSBO>();
					auto indirectBuf = registry->GetBuffer<CloudIndirectDispatchSSBO>();
					if (tileQueueBuf && indirectBuf) {
						vk::DescriptorBufferInfo b0{tileQueueBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
						vk::DescriptorBufferInfo b1{indirectBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
						std::array<vk::WriteDescriptorSet, 2> writes{};
						writes[0].setDstSet(cloudSsboSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b0);
						writes[1].setDstSet(cloudSsboSet).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b1);
						m_device.updateDescriptorSets(writes, nullptr);
					}
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				cloudSsboLayout
			};

			CloudTileSchedulerPushConstants pushT{
				.boundingMapIdx = ctx.Index<CloudBoundingTexture>(),
				.outErrorMapIdx = ctx.Index<CloudErrorMap>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesT{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTileSchedulerPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesT
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 3> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet),
				cloudSsboSet
			};
			if (boundSets[0] && boundSets[1] && boundSets[2]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTileSchedulerPushConstants), &pushT);

			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;
			uint32_t gx = (tileCols + 7) / 8;
			uint32_t gy = (tileRows + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
		}
	};

	struct CloudRenderNode : render::NodeRegistrar<CloudRenderNode> {
		using Resources = graph::Declares<
			graph::Read<GBufferDepth>,
			graph::Read<CloudBoundingTexture>,
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Read<CloudTileQueueSSBO>,
			graph::Read<CloudIndirectDispatchSSBO>,
			graph::Create<CloudPackedColor>,
			graph::Create<CloudPackedDepth>,
			graph::Create<CloudPackedVelocity>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;
		vk::DescriptorSetLayout cloudSsboLayout{nullptr};
		vk::DescriptorPool      cloudSsboPool{nullptr};
		vk::DescriptorSet       cloudSsboSet{nullptr};
		vk::Device              m_device{nullptr};

		void Init(const render::NodeServices& services) {
			m_device = services.device;
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_render.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			std::array<vk::DescriptorSetLayoutBinding, 1> ssboBindings{};
			ssboBindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(ssboBindings);
			cloudSsboLayout = services.device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(1);
			cloudSsboPool = services.device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(cloudSsboPool);
			allocInfo.setSetLayouts(cloudSsboLayout);
			cloudSsboSet = services.device.allocateDescriptorSets(allocInfo).front();
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
			if (cloudSsboPool) device.destroyDescriptorPool(cloudSsboPool);
			if (cloudSsboLayout) device.destroyDescriptorSetLayout(cloudSsboLayout);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedVelocity>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					auto tileQueueBuf = registry->GetBuffer<CloudTileQueueSSBO>();
					if (tileQueueBuf) {
						vk::DescriptorBufferInfo b0{tileQueueBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
						std::array<vk::WriteDescriptorSet, 1> writes{};
						writes[0].setDstSet(cloudSsboSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b0);
						m_device.updateDescriptorSets(writes, nullptr);
					}
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				cloudSsboLayout
			};

			CloudRenderPushConstants pushR{
				.depthTextureIdx = ctx.Index<GBufferDepth>(),
				.boundingTextureIdx = ctx.Index<CloudBoundingTexture>(),
				.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>(),
				.weatherTextureIdx = ctx.Index<CloudWeatherTexture>(),
				.volume3DIdx = ctx.Index<Cloud3DVolumeTexture>(),
				.outPackedColorIdx = ctx.Index<CloudPackedColor>(),
				.outPackedDepthIdx = ctx.Index<CloudPackedDepth>(),
				.outPackedVelocityIdx = ctx.Index<CloudPackedVelocity>(),
				.outErrorMapIdx = ctx.Index<CloudErrorMap>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesR{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudRenderPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesR
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 3> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet),
				cloudSsboSet
			};
			if (boundSets[0] && boundSets[1] && boundSets[2]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudRenderPushConstants), &pushR);

			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;
			vkCmd.dispatch(tileCols * tileRows, 1, 1);
		}
	};

	struct CloudTemporalNode : render::NodeRegistrar<CloudTemporalNode> {
		using Resources = graph::Declares<
			graph::Read<CloudPackedColor>,
			graph::Read<CloudPackedDepth>,
			graph::Read<CloudPackedVelocity>,
			graph::Read<CloudBoundingTexture>,
			graph::Create<CloudTemporalColor>,
			graph::Create<CloudTemporalMoments>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;
		vk::DescriptorSetLayout cloudSsboLayout{nullptr};
		vk::DescriptorPool      cloudSsboPool{nullptr};
		vk::DescriptorSet       cloudSsboSet{nullptr};
		vk::Device              m_device{nullptr};

		void Init(const render::NodeServices& services) {
			m_device = services.device;
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_temporal_reprojection.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			std::array<vk::DescriptorSetLayoutBinding, 1> ssboBindings{};
			ssboBindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(ssboBindings);
			cloudSsboLayout = services.device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(1);
			cloudSsboPool = services.device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(cloudSsboPool);
			allocInfo.setSetLayouts(cloudSsboLayout);
			cloudSsboSet = services.device.allocateDescriptorSets(allocInfo).front();
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
			if (cloudSsboPool) device.destroyDescriptorPool(cloudSsboPool);
			if (cloudSsboLayout) device.destroyDescriptorSetLayout(cloudSsboLayout);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTemporalColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTemporalMoments>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					auto tileQueueBuf = registry->GetBuffer<CloudTileQueueSSBO>();
					if (tileQueueBuf) {
						vk::DescriptorBufferInfo b0{tileQueueBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
						std::array<vk::WriteDescriptorSet, 1> writes{};
						writes[0].setDstSet(cloudSsboSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b0);
						m_device.updateDescriptorSets(writes, nullptr);
					}
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				cloudSsboLayout
			};

			CloudTemporalPushConstants pushTemp{
				.packedFrameIdx = ctx.Index<CloudPackedColor>(),
				.packedDepthIdx = ctx.Index<CloudPackedDepth>(),
				.packedVelocityIdx = ctx.Index<CloudPackedVelocity>(),
				.boundingMapIdx = ctx.Index<CloudBoundingTexture>(),
				.outColorIdx = ctx.Index<CloudTemporalColor>(),
				.outDepthIdx = ctx.Index<CloudPackedDepth>(),
				.outMomentsIdx = ctx.Index<CloudTemporalMoments>(),
				.outErrorMapIdx = ctx.Index<CloudErrorMap>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesTemp{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTemporalPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesTemp
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 3> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet),
				cloudSsboSet
			};
			if (boundSets[0] && boundSets[1] && boundSets[2]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTemporalPushConstants), &pushTemp);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
		}
	};

	struct CloudSpatialFilterNode : render::NodeRegistrar<CloudSpatialFilterNode> {
		using Resources = graph::Declares<
			graph::Read<CloudTemporalColor>,
			graph::Read<CloudPackedDepth>,
			graph::Read<CloudTemporalMoments>,
			graph::Read<CloudErrorMap>,
			graph::Create<CloudFilteredColor>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_spatial_filter.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudFilteredColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			CloudSpatialFilterPushConstants pushFilter{
				.cloudColorIdx = ctx.Index<CloudTemporalColor>(),
				.cloudDepthIdx = ctx.Index<CloudPackedDepth>(),
				.cloudMomentsIdx = ctx.Index<CloudTemporalMoments>(),
				.errorMapIdx = ctx.Index<CloudErrorMap>(),
				.outFilteredColorIdx = ctx.Index<CloudFilteredColor>(),
			};
			std::array<vk::PushConstantRange, 1> pushRangesFilter{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudSpatialFilterPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRangesFilter
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudSpatialFilterPushConstants), &pushFilter);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudBakeNode);
	BRASSICA_REGISTER_NODE(CloudBoundingNode);
	BRASSICA_REGISTER_NODE(CloudShadowBakeNode);
	BRASSICA_REGISTER_NODE(CloudTileSchedulerNode);
	BRASSICA_REGISTER_NODE(CloudRenderNode);
	BRASSICA_REGISTER_NODE(CloudTemporalNode);
	BRASSICA_REGISTER_NODE(CloudSpatialFilterNode);

} // namespace brassica
