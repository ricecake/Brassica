#include "passes/GradientPass.hpp"

#include <vector>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"

namespace brassica {

	GradientPass::GradientPass(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher, vk::PipelineCache pCache):
		RenderPass("GradientPass", dev, colorFmt) {
		InitPipeline(dev, colorFmt, watcher, pCache);
	}

	GradientPass::~GradientPass() = default;

	void
	GradientPass::InitPipeline(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher, vk::PipelineCache pCache) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/gradient.vert")) {
			spdlog::error("Failed to compile gradient.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/gradient.frag")) {
			spdlog::error("Failed to compile gradient.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);
		InitRenderPipeline(
			colorFmt,
			vk::Format::eUndefined,
			{},
			{},
			watcher,
			false,
			false,
			vk::CompareOp::eLess,
			vk::CullModeFlagBits::eNone,
			pCache
		);
	}

} // namespace brassica
