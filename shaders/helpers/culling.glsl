#ifndef BRASSICA_CULLING_GLSL
#define BRASSICA_CULLING_GLSL

#include "bindless.glsl"
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

// HiZ AABB Occlusion Culling for Reverse-Z depth buffers (1.0 = near, 0.0 = far).
// relMinB, relMaxB: Camera-relative AABB bounds (camera at origin).
// viewProjRel: View-projection matrix in camera-relative space.
// hizIndex: Bindless sampled texture index for RG32F HiZ texture (x = minDepth, y = maxDepth).
bool isAABBOccludedByHiZ(vec3 relMinB, vec3 relMaxB, mat4 viewProjRel, uint hizIndex) {
	if (hizIndex == 0u) {
		return false;
	}

	vec3 corners[8];
	corners[0] = vec3(relMinB.x, relMinB.y, relMinB.z);
	corners[1] = vec3(relMaxB.x, relMinB.y, relMinB.z);
	corners[2] = vec3(relMinB.x, relMaxB.y, relMinB.z);
	corners[3] = vec3(relMaxB.x, relMaxB.y, relMinB.z);
	corners[4] = vec3(relMinB.x, relMinB.y, relMaxB.z);
	corners[5] = vec3(relMaxB.x, relMinB.y, relMaxB.z);
	corners[6] = vec3(relMinB.x, relMaxB.y, relMaxB.z);
	corners[7] = vec3(relMaxB.x, relMaxB.y, relMaxB.z);

	vec2 minUV = vec2(1.0);
	vec2 maxUV = vec2(0.0);
	float aabbMaxDepth = 0.0; // In Reverse-Z, largest NDC depth = point CLOSEST to camera

	for (int i = 0; i < 8; ++i) {
		vec4 clipPos = viewProjRel * vec4(corners[i], 1.0);
		if (clipPos.w <= 0.001) {
			return false; // Near plane intersection or behind camera
		}
		vec3 ndc = clipPos.xyz / clipPos.w;
		vec2 uv = ndc.xy * 0.5 + 0.5;

		minUV = min(minUV, uv);
		maxUV = max(maxUV, uv);
		aabbMaxDepth = max(aabbMaxDepth, ndc.z);
	}

	minUV = clamp(minUV, vec2(0.0), vec2(1.0));
	maxUV = clamp(maxUV, vec2(0.0), vec2(1.0));

	if (minUV.x >= maxUV.x || minUV.y >= maxUV.y) {
		return false;
	}

	ivec2 hizTexSize = textureSize(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		0
	);
	if (hizTexSize.x <= 0 || hizTexSize.y <= 0) {
		return false;
	}

	vec2 sizeInPixels = (maxUV - minUV) * vec2(hizTexSize);
	float maxDim = max(sizeInPixels.x, sizeInPixels.y);
	float mipLevel = ceil(log2(max(maxDim, 1.0)));
	int maxMip = textureQueryLevels(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP])
	) - 1;
	mipLevel = clamp(mipLevel, 0.0, float(max(maxMip, 0)));

	vec2 uv00 = minUV;
	vec2 uv10 = vec2(maxUV.x, minUV.y);
	vec2 uv01 = vec2(minUV.x, maxUV.y);
	vec2 uv11 = maxUV;

	vec2 d00 = textureLod(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		uv00,
		mipLevel
	).rg;
	vec2 d10 = textureLod(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		uv10,
		mipLevel
	).rg;
	vec2 d01 = textureLod(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		uv01,
		mipLevel
	).rg;
	vec2 d11 = textureLod(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		uv11,
		mipLevel
	).rg;

	// In Reverse-Z, .r channel is minDepth (furthest scene surface depth in that region).
	float minSceneDepth = min(min(d00.x, d10.x), min(d01.x, d11.x));

	if (minSceneDepth <= 0.0) {
		return false;
	}

	const float depthSlack = 0.0005;
	return (aabbMaxDepth + depthSlack) < minSceneDepth;
}

#endif // BRASSICA_CULLING_GLSL
