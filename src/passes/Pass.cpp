#include "passes/Pass.hpp"

namespace brassica {

	Pass::Pass(std::string passName, vk::Device dev)
		: name(std::move(passName)), device(dev) {}

	Pass::~Pass() {
		// Pipeline destruction should be called explicitly or by derived classes
	}

	void Pass::DestroyPipeline() {
		if (device) {
			if (pipeline) {
				device.destroyPipeline(pipeline);
				pipeline = nullptr;
			}
			if (pipelineLayout) {
				device.destroyPipelineLayout(pipelineLayout);
				pipelineLayout = nullptr;
			}
		}
	}

} // namespace brassica
