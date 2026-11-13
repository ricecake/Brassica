#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/DecorAccelerationStructure.hpp"
#include "passes/model/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct DecorTlasInstancePushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t blasAddressBufIdx{0};
		std::uint32_t outInstanceBufIdx{0};
	};

	// Authors DecorTLASInstanceBuffer every frame (model_decor_tlas_instances.comp, far-only MVP
	// -- see that shader's own comment) and rebuilds DecorAccelerationStructure's TLAS from it.
	// Always active, regardless of decor population: a TLAS with every instance dead-masked
	// (mask == 0) is the correct, spec-legal steady state before anything's ever spawned.
	struct DecorTlasNode: render::NodeRegistrar<DecorTlasNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		vk::Device                device{};
		ComputeShader              instanceShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			device = services.device;
			if (!instanceShader.CompileComputeFromFile(services.device, "shaders/model_decor_tlas_instances.comp")) {
				spdlog::critical("DecorTlasNode shader compilation failed.");
				throw std::runtime_error("DecorTlasNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&instanceShader);
			}
		}

		void Destroy(vk::Device d) { instanceShader.Destroy(d); }

		[[nodiscard]] static DecorAccelerationStructure* QueryDecorAS() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<DecorAccelerationStructure>()) {
				return ServiceLocator::Instance().Get<DecorAccelerationStructure>().get();
			}
			return nullptr;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<ModelInstanceBuffer>();
			r.Read<DecorTemplateBlasAddressBuffer>();
			r.Create<DecorTLASInstanceBuffer>();
			r.Modify<DecorTLAS>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* decorAS = QueryDecorAS();
			if (!decorAS) {
				return;
			}

			DecorTlasInstancePushConstants push{};
			push.instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			push.blasAddressBufIdx = ctx.BufferIndex<DecorTemplateBlasAddressBuffer>();
			push.outInstanceBufIdx = ctx.BufferIndex<DecorTLASInstanceBuffer>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, instanceShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((kDecorCapacity + 63u) / 64u, 1, 1);

			vk::MemoryBarrier barrier{
				vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eAccelerationStructureReadKHR
			};
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				vk::DependencyFlags{},
				barrier,
				nullptr,
				nullptr
			);

			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}
			auto instBuf = physReg->GetBuffer<DecorTLASInstanceBuffer>();
			if (!instBuf) {
				return;
			}

			vk::Buffer                  buf = instBuf->GetBuffer();
			vk::BufferDeviceAddressInfo addrInfo{};
			addrInfo.setBuffer(buf);
			vk::DeviceAddress addr = device.getBufferAddress(addrInfo);

			// DecorTLAS's handle itself is registered once, in Engine::Init() right after
			// DecorAccelerationStructure::Init() creates it -- not here. It never changes across
			// frames (same object, rebuilt in place below); re-registering it from inside this
			// node's own Execute() would be too late for this same frame's Setup/Compile anyway
			// (which already ran by the time Execute() does -- PhysicalRegistry.hpp's Provision
			// expects every AS key already in m_accelStructs before that point).
			decorAS->BuildTLAS(vkCmd, addr);
		}
	};

	BRASSICA_REGISTER_NODE(DecorTlasNode);

} // namespace brassica
