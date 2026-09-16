#pragma once

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "ImGuiManager.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "RenderPhases.hpp"
#include "ServiceLocator.hpp"

namespace brassica {

	struct ImGuiNode: render::NodeRegistrar<ImGuiNode> {
		using Resources = graph::Declares<graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = SubPhase::MenuOverlay;

		ImGuiManager* mgr = nullptr;
		vk::Format    swapchainFormat = vk::Format::eUndefined;

		void Init(const render::NodeServices& services) {
			swapchainFormat = services.swapchainFormat;
			if (ServiceLocator::Instance().Has<ImGuiManager>()) {
				mgr = ServiceLocator::Instance().Get<ImGuiManager>().get();
			}
		}

		void Destroy(vk::Device) {}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			if (!mgr && ServiceLocator::Instance().Has<ImGuiManager>()) {
				mgr = ServiceLocator::Instance().Get<ImGuiManager>().get();
			}
			bool          visible = mgr ? mgr->IsVisible() : false;
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics, .isActive = visible};
			if (visible) {
				r.realizations.push_back(
					graph::ResourceRealization{
						.key = graph::IdOf<Swapchain>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
					}
				);
			}
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (!mgr && ServiceLocator::Instance().Has<ImGuiManager>()) {
				mgr = ServiceLocator::Instance().Get<ImGuiManager>().get();
			}
			if (!mgr || !mgr->IsVisible()) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (vkCmd) {
				mgr->Render(vkCmd);
			}
		}
	};

	BRASSICA_REGISTER_NODE(ImGuiNode);

} // namespace brassica
