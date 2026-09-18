#pragma once

#include <cstdint>

namespace brassica {

	struct CameraData;

	struct FrameDetails {
		float             deltaTime{0.0f};
		double            totalTime{0.0};
		uint32_t          frameIndex{0};
		const CameraData* camera{nullptr};
	};

} // namespace brassica
