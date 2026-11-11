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
		std::mt19937                          rng(seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		// Same CSG-absorption margin as BuildLeafCluster's own baseOffset, measured from the
		// attaching tube's tip -- the whole ball cluster sits ahead of that margin, not just the
		// first ball.
		float     baseOffset = attachRadius * 1.5f;
		glm::vec3 clusterCenter(0.0f, 0.0f, baseOffset + leafSize * 0.5f);
		float clusterSpread = leafSize * 0.85f;

		// At least 3 balls so the union reads as a lumpy clump rather than one or two bare
		// spheres -- ballCount otherwise tracks leafCount so a denser preset's tuft also gets a
		// denser ball cluster.
		int ballCount = std::max(3, leafCount);

		std::vector<glm::vec3> ballCenters;
		std::vector<float>     ballRadii;
		ballCenters.reserve(static_cast<std::size_t>(ballCount));
		ballRadii.reserve(static_cast<std::size_t>(ballCount));

		std::vector<manifold::Manifold> parts;
		parts.reserve(static_cast<std::size_t>(ballCount + leafCount));

		for (int i = 0; i < ballCount; ++i) {
			glm::vec3 jitter(unit(rng) * 2.0f - 1.0f, unit(rng) * 2.0f - 1.0f, unit(rng) * 2.0f - 1.0f);
			float     jitterLen = glm::length(jitter);
			if (jitterLen > 1e-4f) {
				jitter = (jitter / jitterLen) * (unit(rng) * clusterSpread);
			} else {
				jitter = glm::vec3(0.0f);
			}
			glm::vec3 center = clusterCenter + jitter;
			float     radius = leafSize * (0.35f + 0.25f * unit(rng));

			ballCenters.push_back(center);
			ballRadii.push_back(radius);

			manifold::Manifold ball = manifold::Manifold::Sphere(static_cast<double>(radius), 8);
			ball = ball.Scale(manifold::vec3(1.0, 1.2, 0.70));
			parts.push_back(ball.Translate(manifold::vec3(center.x, center.y, center.z)));
		}

		// Blades "stuck to the side" of a random ball, radially outward from that ball's own
		// center -- breaks up the ball cluster's otherwise-smooth silhouette, same shape
		// AppendLeafBlade already builds for the plain fan cluster above. Appended into one
		// shared vertex/triangle buffer (same pattern BuildLeafCluster's own per-blade loop
		// already uses for several disjoint, possibly-overlapping blades in a single Manifold),
		// then unioned with the balls below rather than left as a separate untouched Manifold --
		// the CSG union is what actually welds a blade's embedded base into the ball it's
		// attached to instead of leaving a visible seam.
		std::vector<float>         bladeVertProperties;
		std::vector<std::uint32_t> bladeTriVerts;
		for (int i = 0; i < leafCount; ++i) {
			auto      ballIdx = static_cast<std::size_t>(unit(rng) * static_cast<float>(ballCount)) % static_cast<std::size_t>(ballCount);
			glm::vec3 ballCenter = ballCenters[ballIdx];
			float     ballRadius = ballRadii[ballIdx];

			float azimuth = unit(rng) * glm::two_pi<float>();
			// Polar angle from local +Z (the growth axis) -- biased away from straight up/down so
			// blades mostly point outward/sideways rather than stacking along the trunk's own
			// direction.
			float     elevation = glm::radians(20.0f + unit(rng) * 90.0f);
			glm::vec3 dir(
				std::sin(elevation) * std::cos(azimuth), std::sin(elevation) * std::sin(azimuth), std::cos(elevation)
			);

			glm::quat orientation =
				AlignZTo(dir) * glm::angleAxis(unit(rng) * glm::two_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));
			float scale = 0.5f + unit(rng) * 0.4f;
			// Base pulled into the ball (anchors it visually, like a real petiole) while the
			// blade's own length carries its tip well past the ball's surface -- same
			// margin-vs-absorption tradeoff as BuildLeafCluster's baseOffset, just measured from
			// a ball's surface instead of the attach tube's.
			glm::vec3 bladeOrigin = ballCenter + dir * (ballRadius * 0.7f);

			// AppendLeafBlade only takes an orientation, not a translation -- build at the local
			// origin, then shift every vertex it just appended by bladeOrigin.
			std::size_t vertBase = bladeVertProperties.size();
			AppendLeafBlade(
				leafSize * scale, leafSize * scale * widthRatio, 0.0f, orientation, bladeVertProperties, bladeTriVerts
			);
			for (std::size_t v = vertBase; v < bladeVertProperties.size(); v += 3) {
				bladeVertProperties[v + 0] += bladeOrigin.x;
				bladeVertProperties[v + 1] += bladeOrigin.y;
				bladeVertProperties[v + 2] += bladeOrigin.z;
			}
		}

		if (!bladeTriVerts.empty()) {
			manifold::MeshGL bladeMesh;
			bladeMesh.numProp = 3;
			bladeMesh.vertProperties = std::move(bladeVertProperties);
			bladeMesh.triVerts = std::move(bladeTriVerts);
			parts.push_back(manifold::Manifold(bladeMesh));
		}

		manifold::Manifold combined =
			(parts.size() == 1) ? parts[0] : manifold::Manifold::BatchBoolean(parts, manifold::OpType::Add);
		if (combined.Status() != manifold::Manifold::Error::NoError) {
			spdlog::critical("BuildLeafBallCluster: boolean union failed with status {}", static_cast<int>(combined.Status()));
			// A bare sphere instead of an empty manifold -- the outer per-joint union
			// (BuildSkinnedManifoldAsset's own BatchBoolean) tolerates an unexpected shape far
			// better than a joint that silently contributes nothing at all.
			return manifold::Manifold::Sphere(static_cast<double>(leafSize * 0.5), 8);
		}
		return combined;
	}

} // namespace brassica
