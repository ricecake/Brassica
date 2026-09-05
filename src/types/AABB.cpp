#include "types/AABB.hpp"
#include <algorithm>
#include <cmath>

namespace brassica {

	AABB AABB::Transform(const glm::mat4& transform) const {
		glm::vec3 corners[8] = {
			{minBound.x, minBound.y, minBound.z},
			{maxBound.x, minBound.y, minBound.z},
			{minBound.x, maxBound.y, minBound.z},
			{maxBound.x, maxBound.y, minBound.z},
			{minBound.x, minBound.y, maxBound.z},
			{maxBound.x, minBound.y, maxBound.z},
			{minBound.x, maxBound.y, maxBound.z},
			{maxBound.x, maxBound.y, maxBound.z}
		};

		glm::vec3 newMin(std::numeric_limits<float>::max());
		glm::vec3 newMax(std::numeric_limits<float>::lowest());

		for (const auto& corner : corners) {
			glm::vec4 transformed = transform * glm::vec4(corner, 1.0f);
			glm::vec3 pt = glm::vec3(transformed) / transformed.w;
			newMin = glm::min(newMin, pt);
			newMax = glm::max(newMax, pt);
		}

		return AABB(newMin, newMax);
	}

	float AABB::DistanceSqToPoint(const glm::vec3& point) const {
		glm::vec3 closestPt = glm::clamp(point, minBound, maxBound);
		glm::vec3 diff = point - closestPt;
		return glm::dot(diff, diff);
	}

	float AABB::DistanceToPoint(const glm::vec3& point) const {
		return std::sqrt(DistanceSqToPoint(point));
	}

	bool AABB::IntersectsFrustum(const std::array<glm::vec4, 6>& planes) const {
		for (const auto& plane : planes) {
			// Find positive vertex (p-vertex) relative to plane normal
			glm::vec3 pVertex = minBound;
			if (plane.x >= 0.0f) pVertex.x = maxBound.x;
			if (plane.y >= 0.0f) pVertex.y = maxBound.y;
			if (plane.z >= 0.0f) pVertex.z = maxBound.z;

			// If p-vertex is outside plane (distance < 0), AABB is completely outside frustum
			if (glm::dot(glm::vec3(plane), pVertex) + plane.w < 0.0f) {
				return false;
			}
		}
		return true;
	}

	uint32_t AABB::CalculateLOD(const glm::vec3& cameraPos, float baseTexelSize, uint32_t maxLODs) const {
		if (maxLODs <= 1) return 0;

		float dist = DistanceToPoint(cameraPos);
		// Scale threshold distance with baseTexelSize
		// Distance threshold doubles per LOD level
		float lod0Distance = baseTexelSize * 100.0f;
		if (dist < lod0Distance) {
			return 0;
		}

		float lodLevel = std::log2(dist / lod0Distance);
		uint32_t lodIndex = static_cast<uint32_t>(std::floor(lodLevel)) + 1;
		return std::min(lodIndex, maxLODs - 1);
	}

	std::array<glm::vec4, 6> AABB::ExtractFrustumPlanes(const glm::mat4& viewProj) {
		std::array<glm::vec4, 6> planes;

		// Left plane
		planes[0] = glm::vec4(
			viewProj[0][3] + viewProj[0][0],
			viewProj[1][3] + viewProj[1][0],
			viewProj[2][3] + viewProj[2][0],
			viewProj[3][3] + viewProj[3][0]
		);
		// Right plane
		planes[1] = glm::vec4(
			viewProj[0][3] - viewProj[0][0],
			viewProj[1][3] - viewProj[1][0],
			viewProj[2][3] - viewProj[2][0],
			viewProj[3][3] - viewProj[3][0]
		);
		// Bottom plane
		planes[2] = glm::vec4(
			viewProj[0][3] + viewProj[0][1],
			viewProj[1][3] + viewProj[1][1],
			viewProj[2][3] + viewProj[2][1],
			viewProj[3][3] + viewProj[3][1]
		);
		// Top plane
		planes[3] = glm::vec4(
			viewProj[0][3] - viewProj[0][1],
			viewProj[1][3] - viewProj[1][1],
			viewProj[2][3] - viewProj[2][1],
			viewProj[3][3] - viewProj[3][1]
		);
		// Near plane
		planes[4] = glm::vec4(
			viewProj[0][3] + viewProj[0][2],
			viewProj[1][3] + viewProj[1][2],
			viewProj[2][3] + viewProj[2][2],
			viewProj[3][3] + viewProj[3][2]
		);
		// Far plane
		planes[5] = glm::vec4(
			viewProj[0][3] - viewProj[0][2],
			viewProj[1][3] - viewProj[1][2],
			viewProj[2][3] - viewProj[2][2],
			viewProj[3][3] - viewProj[3][2]
		);

		// Normalize planes
		for (auto& plane : planes) {
			float len = glm::length(glm::vec3(plane));
			if (len > 0.0001f) {
				plane /= len;
			}
		}

		return planes;
	}

} // namespace brassica
