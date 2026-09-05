#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	struct AABB {
		glm::vec3 minBound{0.0f};
		glm::vec3 maxBound{0.0f};

		AABB() = default;
		AABB(const glm::vec3& minB, const glm::vec3& maxB) : minBound(minB), maxBound(maxB) {}

		glm::vec3 Center() const { return (minBound + maxBound) * 0.5f; }
		glm::vec3 Extents() const { return (maxBound - minBound) * 0.5f; }

		AABB Transform(const glm::mat4& transform) const;

		// Distance calculations
		float DistanceSqToPoint(const glm::vec3& point) const;
		float DistanceToPoint(const glm::vec3& point) const;

		// Frustum Culling
		bool IntersectsFrustum(const std::array<glm::vec4, 6>& planes) const;

		// LOD calculation based on camera distance and clipmap level bounds
		uint32_t CalculateLOD(const glm::vec3& cameraPos, float baseTexelSize, uint32_t maxLODs) const;

		// Frustum plane extraction from View-Projection matrix (Gribb-Hartmann method)
		static std::array<glm::vec4, 6> ExtractFrustumPlanes(const glm::mat4& viewProj);
	};

} // namespace brassica
