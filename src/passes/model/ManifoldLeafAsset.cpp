#include "passes/model/ManifoldLeafAsset.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

namespace brassica {

	namespace {

		// One flat, pointed blade in its own local frame: base at the attachment point (local
		// origin), tip pointing along +Z, width along X, thickness (top/bottom apex offset)
		// along Y -- same closed "two apexes over an outline ring" construction
		// ManifoldFlowerAsset's own BuildFlowerHead uses for its petal lens, just a 4-point
		// elongated diamond outline instead of a symmetric circular one. orientation rotates the
		// whole blade (and its local +Z/+X/+Y axes) before appending, so a cluster can splay
		// several of these outward from a shared origin.
		void AppendLeafBlade(
			float length, float width, glm::quat orientation, std::vector<float>& vertProperties,
			std::vector<std::uint32_t>& triVerts
		) {
			float thickness = std::min(length, width) * 0.2f;
			float midZ = length * 0.4f;

			glm::vec3 localVerts[6] = {
				glm::vec3(0.0f, thickness * 0.5f, length * 0.35f),  // apex top
				glm::vec3(0.0f, -thickness * 0.5f, length * 0.35f), // apex bottom
				glm::vec3(0.0f, 0.0f, 0.0f),                        // base (attachment point)
				glm::vec3(-width * 0.5f, 0.0f, midZ),               // wide left
				glm::vec3(0.0f, 0.0f, length),                      // tip
				glm::vec3(width * 0.5f, 0.0f, midZ),                // wide right
			};

			auto base = static_cast<std::uint32_t>(vertProperties.size() / 3);
			for (glm::vec3 v : localVerts) {
				glm::vec3 world = orientation * v;
				vertProperties.push_back(world.x);
				vertProperties.push_back(world.y);
				vertProperties.push_back(world.z);
			}

			std::uint32_t apexTop = base + 0;
			std::uint32_t apexBottom = base + 1;
			std::uint32_t ring[4] = {base + 2, base + 3, base + 4, base + 5};
			for (int i = 0; i < 4; ++i) {
				std::uint32_t a = ring[i];
				std::uint32_t b = ring[(i + 1) % 4];
				// Top fan / bottom fan with reversed winding, same convention BuildFlowerHead
				// uses for its own petal ring.
				triVerts.push_back(apexTop);
				triVerts.push_back(a);
				triVerts.push_back(b);
				triVerts.push_back(apexBottom);
				triVerts.push_back(b);
				triVerts.push_back(a);
			}
		}

	} // namespace

	manifold::Manifold BuildLeafCluster(float leafSize, int leafCount, std::uint32_t seed, float widthRatio) {
		std::mt19937                          rng(seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		std::vector<float>         vertProperties;
		std::vector<std::uint32_t> triVerts;
		auto                       count = static_cast<std::size_t>(std::max(leafCount, 0));
		vertProperties.reserve(count * 6 * 3);
		triVerts.reserve(count * 8 * 3);

		for (int i = 0; i < leafCount; ++i) {
			// Evenly spaced around the attachment point, each with its own small random wander
			// so the tuft doesn't look mechanically regular -- same spirit as
			// ProceduralPlantGenerator's own per-segment direction jitter.
			float yaw = (glm::two_pi<float>() * static_cast<float>(i)) / static_cast<float>(std::max(leafCount, 1)) +
				(unit(rng) - 0.5f) * 0.6f;
			// Splayed outward/upward from the growth axis (local +Z), not a flat fan in one
			// plane -- a real tuft, not a disc of leaves.
			float pitchDeg = 30.0f + unit(rng) * 30.0f;
			float scale = 0.75f + unit(rng) * 0.5f;

			glm::quat yawRot = glm::angleAxis(yaw, glm::vec3(0.0f, 0.0f, 1.0f));
			glm::quat pitchRot = glm::angleAxis(glm::radians(pitchDeg), glm::vec3(1.0f, 0.0f, 0.0f));
			glm::quat orientation = yawRot * pitchRot;

			AppendLeafBlade(leafSize * scale, leafSize * scale * widthRatio, orientation, vertProperties, triVerts);
		}

		manifold::MeshGL mesh;
		mesh.numProp = 3;
		mesh.vertProperties = std::move(vertProperties);
		mesh.triVerts = std::move(triVerts);
		return manifold::Manifold(mesh);
	}

} // namespace brassica
