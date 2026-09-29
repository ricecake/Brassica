#ifndef BRASSICA_CULLING_GLSL
#define BRASSICA_CULLING_GLSL

#include "common.glsl"

// Check AABB against frustum planes in camera-relative space (camera at origin).
// margin: half-extent around the camera origin that's always treated as visible, protecting
//         the region directly beneath/around the camera from precision-driven false rejection.
// planeSlack: how far outside a plane (in world units, as a negative distance) an AABB may sit
//             before being rejected -- lets each caller keep its own existing tolerance.
bool isAABBVisible(vec3 relMinB, vec3 relMaxB, mat4 vp, vec3 margin, float planeSlack) {
	if (relMinB.x <= margin.x && relMaxB.x >= -margin.x &&
	    relMinB.z <= margin.z && relMaxB.z >= -margin.z &&
	    relMinB.y <= margin.y && relMaxB.y >= -margin.y) {
		return true;
	}

	vec4 planes[6];
	planes[0] = vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // Left
	planes[1] = vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // Right
	planes[2] = vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // Bottom
	planes[3] = vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // Top
	planes[4] = vec4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]); // Near (Vulkan 0 <= z <= w)
	planes[5] = vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // Far

	for (int i = 0; i < 6; i++) {
		vec3 pVertex = relMinB;
		if (planes[i].x >= 0.0) pVertex.x = relMaxB.x;
		if (planes[i].y >= 0.0) pVertex.y = relMaxB.y;
		if (planes[i].z >= 0.0) pVertex.z = relMaxB.z;

		float len = length(planes[i].xyz);
		vec4 normPlane = (len > 0.0001) ? planes[i] / len : planes[i];

		if (dot(normPlane.xyz, pVertex) + normPlane.w < planeSlack) {
			return false;
		}
	}
	return true;
}

// Exact tangent-line test for the parabolic curvature approximation this engine renders
// (worldY -= d^2/(2*FAKE_PLANET_RADIUS)): a tile is entirely below the geometric horizon as
// seen from a camera at altitude `camAltitudeAboveRef` when the tile's nearest horizontal
// approach `tileDist` exceeds sqrt(2*R*a) + sqrt(2*R*b), where a/b are the camera/tile heights
// above the same reference surface (b clamped >= 0 -- a valley below the reference can't peek
// over a horizon defined by that same reference).
//
// This is the same closed form as the standard spherical horizon-distance formula, not an
// approximation stacked on top of one: with surface S(d) = -d^2/(2R) and the camera at the
// origin, the sightline to a point at (d, b - d^2/(2R)) dips below S iff
// k*t^2 + (b - k - a)*t + a < 0 for some t in (0,1), where k = d^2/(2R). That requires
// (a + k - b)^2 > 4*k*a; substituting A = sqrt(2Ra), B = sqrt(2Rb) reduces it to |d - A| > B,
// i.e. d > A + B.
//
// Only meaningful looking down from above the reference surface -- once the camera itself is at
// or below it (e.g. swimming underwater), there's no curvature bulge blocking line of sight, so
// this never culls in that regime (clamping `a` to >= 0 instead of returning early here would
// collapse the whole horizonSum toward 0, misreading "no curvature effect" as "cull everything
// past `slack`" -- exactly the bug that hid nearby submerged terrain from an underwater camera).
bool isBeyondHorizon(float tileDist, float tileMaxYAboveRef, float camAltitudeAboveRef, float slack) {
	if (camAltitudeAboveRef <= 0.0) {
		return false;
	}
	float b = max(tileMaxYAboveRef, 0.0);
	float horizonSum = sqrt(2.0 * FAKE_PLANET_RADIUS * camAltitudeAboveRef) + sqrt(2.0 * FAKE_PLANET_RADIUS * b);
	return tileDist > horizonSum + slack;
}

#endif // BRASSICA_CULLING_GLSL
