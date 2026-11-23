#include "passes/model/ManifoldFlowerAsset.hpp"

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace brassica {

	namespace {

		constexpr int kPetalCount = 8;

		// Exact profile math from shaders/foliage.mesh's own emitFlowerItem (headProfile 0-3) --
		// radProfile/dropZ per petal index, the only thing that changes between variants.
		// dropZ is boidish's own "dropY" renamed: in this local frame, growth/orientation is +Z,
		// not +Y, so the droop (Bell profile) happens along local Z instead.
		void PetalOffset(FlowerProfile profile, int i, float& radProfile, float& dropZ) {
			float pAngle = static_cast<float>(i) * (glm::two_pi<float>() / static_cast<float>(kPetalCount));
			radProfile = 1.0f;
			dropZ = 0.0f;
			switch (profile) {
			case FlowerProfile::Diamond:
				radProfile = (i % 2 == 0) ? 1.0f : 0.5f;
				break;
			case FlowerProfile::Bell:
				radProfile = 0.85f;
				dropZ = -0.35f; // scaled by headSize below, same as emitFlowerItem's own dropY/headSize*headSize
				break;
			case FlowerProfile::Star:
				radProfile = (i % 2 == 0) ? 1.1f : 0.35f;
				break;
			case FlowerProfile::Heart:
				radProfile = 0.6f + 0.5f * std::sin(pAngle * 2.0f);
				break;
			}
		}

	} // namespace

	manifold::Manifold BuildFlowerHead(float headSize, FlowerProfile profile) {
		// A thin closed "lens": one apex above and below the petal plane, joined to a ring of
		// kPetalCount petal vertices -- the same silhouette emitFlowerItem's open fan produces,
		// but closed (every ring edge borders exactly one top triangle and one bottom triangle),
		// so it's a valid 2-manifold solid Manifold's CSG pipeline can union.
		const float apexOffset = headSize * 0.15f;

		std::vector<float> vertProperties;
		vertProperties.reserve((2 + kPetalCount) * 3);
		auto pushVert = [&](glm::vec3 p) {
			vertProperties.push_back(p.x);
			vertProperties.push_back(p.y);
			vertProperties.push_back(p.z);
		};

		constexpr uint32_t kCenterTop = 0;
		constexpr uint32_t kCenterBottom = 1;
		constexpr uint32_t kRingBase = 2;

		pushVert(glm::vec3(0.0f, 0.0f, apexOffset));
		pushVert(glm::vec3(0.0f, 0.0f, -apexOffset));
		for (int i = 0; i < kPetalCount; ++i) {
			float pAngle = static_cast<float>(i) * (glm::two_pi<float>() / static_cast<float>(kPetalCount));
			float radProfile;
			float dropZ;
			PetalOffset(profile, i, radProfile, dropZ);
			glm::vec3 ring(std::cos(pAngle) * radProfile * headSize, std::sin(pAngle) * radProfile * headSize, dropZ * headSize);
			pushVert(ring);
		}

		std::vector<uint32_t> triVerts;
		triVerts.reserve(kPetalCount * 2 * 3);
		for (int i = 0; i < kPetalCount; ++i) {
			uint32_t a = kRingBase + static_cast<uint32_t>(i);
			uint32_t b = kRingBase + static_cast<uint32_t>((i + 1) % kPetalCount);
			// Top fan: CCW as seen from +Z (outward/up).
			triVerts.push_back(kCenterTop);
			triVerts.push_back(a);
			triVerts.push_back(b);
			// Bottom fan: reversed winding -- CCW as seen from -Z (outward/down).
			triVerts.push_back(kCenterBottom);
			triVerts.push_back(b);
			triVerts.push_back(a);
		}

		manifold::MeshGL mesh;
		mesh.numProp = 3;
		mesh.vertProperties = std::move(vertProperties);
		mesh.triVerts = std::move(triVerts);

		return manifold::Manifold(mesh);
	}

} // namespace brassica
