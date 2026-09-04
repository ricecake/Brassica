#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "vulkan/vulkan.hpp"
#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/PassResource.hpp"

namespace brassica {

	class Pass {
	public:
		Pass(std::string name, vk::Device device);
		virtual ~Pass();

		Pass(const Pass&) = delete;
		Pass& operator=(const Pass&) = delete;
		Pass(Pass&&) noexcept = default;
		Pass& operator=(Pass&&) noexcept = default;

		[[nodiscard]] const std::string& GetName() const { return name; }
		[[nodiscard]] vk::Device GetDevice() const { return device; }
		[[nodiscard]] vk::Pipeline GetPipeline() const { return pipeline; }
		[[nodiscard]] vk::PipelineLayout GetPipelineLayout() const { return pipelineLayout; }

		virtual void DestroyPipeline();

	protected:
		std::string        name;
		vk::Device         device{nullptr};
		vk::Pipeline       pipeline{nullptr};
		vk::PipelineLayout pipelineLayout{nullptr};
	};

} // namespace brassica
