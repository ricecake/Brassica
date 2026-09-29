#pragma once

#include <glm/glm.hpp>

namespace brassica {

	struct AudioState {
		glm::vec3 listenerPos{0.0f};
		glm::vec3 listenerFront{0.0f, 0.0f, -1.0f};
		glm::vec3 listenerUp{0.0f, 1.0f, 0.0f};
		float     listenerSpeed{0.0f};
		float     listenerFov{90.0f};
		float     globalPitch{1.0f};
		float     masterVolume{1.0f};
		float     musicVolume{1.0f};
		float     sfxVolume{1.0f};

		// Environmental data for procedural effects
		glm::vec3 windVelocity{0.0f};
		float     windStrength{0.0f};
		float     rainIntensity{0.0f};
		float     grassDensity{0.0f};

		// Camera state for altitude & speed effects
		float     altitude{0.0f}; // camera height/altitude in meters
		float     speed{0.0f};    // camera current speed in m/s
	};

} // namespace brassica
