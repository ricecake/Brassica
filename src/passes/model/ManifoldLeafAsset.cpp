#include "passes/model/ManifoldLeafAsset.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <spdlog/spdlog.h>

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
			float length, float width, float baseOffset, glm::quat orientation, std::vector<float>& vertProperties,
			std::vector<std::uint32_t>& triVerts
		) {
			float thickness = std::min(length, width) * 0.2f;
			float midZ = baseOffset + length * 0.4f;

			glm::vec3 localVerts[6] = {
				glm::vec3(0.0f, thickness * 0.5f, baseOffset + length * 0.35f),  // apex top
				glm::vec3(0.0f, -thickness * 0.5f, baseOffset + length * 0.35f), // apex bottom
				glm::vec3(0.0f, 0.0f, baseOffset),                               // base (attachment point)
				glm::vec3(-width * 0.5f, 0.0f, midZ),                            // wide left
				glm::vec3(0.0f, 0.0f, baseOffset + length),                      // tip
				glm::vec3(width * 0.5f, 0.0f, midZ),                             // wide right
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

		// A quaternion mapping local +Z onto an arbitrary direction, used to orient a blade
		// radially outward from a ball's surface (BuildLeafBallCluster below) instead of the
		// yaw/pitch-around-the-growth-axis composition BuildLeafCluster's own fan uses.
		glm::quat AlignZTo(glm::vec3 dir) {
			dir = glm::normalize(dir);
			constexpr glm::vec3 z(0.0f, 0.0f, 1.0f);
			float                d = glm::clamp(glm::dot(z, dir), -1.0f, 1.0f);
			if (d > 0.9999f) {
				return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
			}
			if (d < -0.9999f) {
				return glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
			}
			glm::vec3 axis = glm::normalize(glm::cross(z, dir));
			return glm::angleAxis(std::acos(d), axis);
		}

	} // namespace

	manifold::Manifold BuildLeafCluster(
		float leafSize, int leafCount, std::uint32_t seed, float widthRatio, float attachRadius
	) {
		std::mt19937                          rng(seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		// Same margin reasoning as ProceduralPlantOptimizer.cpp's Hub-flare 1.5x fix: each
		// blade's own base starts at local origin, which sits exactly at the attaching tube's
		// end-cap center -- without a real margin past that tube's own radius, BuildSkinnedManifoldAsset's
		// CSG union (BatchBoolean Add, outer-envelope-only) silently clips away whatever part of
		// the blade is still inside the tube's volume, same absorption bug class confirmed for
		// under-sized Hub spheres.
		float baseOffset = attachRadius * 1.5f;

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

			AppendLeafBlade(
				leafSize * scale, leafSize * scale * widthRatio, baseOffset, orientation, vertProperties, triVerts
			);
		}

		manifold::MeshGL mesh;
		mesh.numProp = 3;
		mesh.vertProperties = std::move(vertProperties);
		mesh.triVerts = std::move(triVerts);
		return manifold::Manifold(mesh);
	}

	manifold::Manifold BuildLeafBallCluster(
		float leafSize, int leafCount, std::uint32_t seed, float widthRatio, float attachRadius
	) {
		// 1. Calculate attachment offset to avoid CSG absorption by the branch tip
		float baseOffset = attachRadius * 1.5f;
		glm::vec3 clusterCenter(0.0f, 0.0f, baseOffset + leafSize * 0.5f);

		// 2. Generate a single low-res sphere.
		// Reducing circular segments from the default drastically cuts triangle count.
		manifold::Manifold canopy = manifold::Manifold::Sphere(static_cast<double>(leafSize), 5);

		// 3. Scale into a flattened, wide bean/cloud shape
		canopy = canopy.Scale(manifold::vec3(1.4, 0.6, 1.2));

		// 4. Warp the vertices with a simple pseudo-random noise to make it lumpy
		canopy = canopy.Warp([seed](manifold::vec3& v) {
			// High-frequency spatial sine wave based on position
			float noise = std::sin(v.x * 15.0f + seed) *
						std::cos(v.y * 15.0f + seed) *
						std::sin(v.z * 15.0f + seed);

			// Displace vertices outward/inward by up to 15%
			float displacement = 1.0f + (noise * 0.15f);
			v.x *= displacement;
			v.y *= displacement;
			v.z *= displacement;
		});

		// 5. Move to the branch tip
		return canopy.Translate(manifold::vec3(clusterCenter.x, clusterCenter.y, clusterCenter.z));
	}

	manifold::Manifold BuildGlobalCanopyHull(
		const std::vector<glm::vec3>& terminalPoints, float leafSize, std::uint32_t seed
	) {
		if (terminalPoints.empty()) {
			return manifold::Manifold();
		}

		std::vector<manifold::Manifold> spheres;
		spheres.reserve(terminalPoints.size());

		// Place a low-res sphere at every terminal branch tip
		for (const glm::vec3& pt : terminalPoints) {
			manifold::Manifold sphere = manifold::Manifold::Sphere(static_cast<double>(leafSize), 5);
			spheres.push_back(sphere.Translate(manifold::vec3(pt.x, pt.y, pt.z)));
		}

		// Wrap all spheres in a single convex boundary
		manifold::Manifold canopy = manifold::Manifold::Hull(spheres);

		// Displace the hull vertices slightly to create a pillowy, lumpy volume
		canopy = canopy.Warp([seed](manifold::vec3& v) {
			float noise = std::sin(v.x * 4.0f + seed) *
						std::cos(v.y * 4.0f + seed) *
						std::sin(v.z * 4.0f + seed);
			float displacement = 1.0f + (noise * 0.08f);
			v.x *= displacement;
			v.y *= displacement;
			v.z *= displacement;
		});

		return canopy;
	}

} // namespace brassica
