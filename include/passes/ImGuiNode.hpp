#pragma once

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "ImGuiManager.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "RenderPhases.hpp"
#include "ServiceLocator.hpp"

namespace brassica {

	// Forward-declared only: After<TonemapNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct TonemapNode;

	struct ImGuiNode: render::NodeRegistrar<ImGuiNode> {
		ImGuiManager* mgr = nullptr;
		vk::Format    swapchainFormat = vk::Format::eUndefined;

		void Init(const render::NodeServices& services) {
			swapchainFormat = services.swapchainFormat;
			if (ServiceLocator::Instance().Has<ImGuiManager>()) {
				mgr = ServiceLocator::Instance().Get<ImGuiManager>().get();
			}
		}

		void Destroy(vk::Device) {}

		// Active(visible) only gates whether this node runs -- the Modify<Swapchain> access is
		// still declared unconditionally (the old code pushed no realization at all when
		// !visible, the same unconditional-declaration bug TonemapComputeNode had). No .Desc() --
		// Swapchain is Imported, see TonemapNode.hpp's identical reasoning. S4: no kPhase --
		// After<TonemapNode>() is the real ordering constraint now (both this node and
		// TonemapNode Modify<Swapchain>, so without an explicit relation they'd be an unordered
		// writer pair and a hard ambiguity error).
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			if (!mgr && ServiceLocator::Instance().Has<ImGuiManager>()) {
				mgr = ServiceLocator::Instance().Get<ImGuiManager>().get();
			}
			bool visible = mgr ? mgr->IsVisible() : false;
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Active(visible);
			r.Modify<Swapchain>();
			r.After<TonemapNode>();
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
