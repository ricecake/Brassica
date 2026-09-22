#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform AtmosphereCompositePushConstants {
	vec3  sunDir;
	float maxUnderwaterDistanceKM;
	vec3  sunRadiance;
	float multiScatScale;
	vec3  moonDir;
	uint  stepCount;
	vec3  moonRadiance;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  hdrColorIndex;
	uint  transmittanceIndex;
	uint  multiScatteringIndex;
	uint  skyViewIndex;
}
push;

#include "aerial_perspective.glsl"

// Fogs already-shaded scene geometry with real aerial perspective / underwater extinction --
// distinct from SkyBackgroundNode, which only paints the empty-pixel backdrop. Runs at
// SubPhase::Atmosphere (900), before WaterNode (1200): a submerged camera looking at distant
// terrain needs no water mesh geometry at all, just the mathematical water-level plane (see
// evaluateAtmosphere's two-segment march) -- there's nothing there to occlude with regardless of
// whether water draws before or after this pass.
void main() {
	vec4 albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3 relPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;

	float waterLevelKM = u_waterLevel / 1000.0;
	float camAltKM = uCameraPosition.y / 1000.0;
	bool  hasSurface = albedo.a >= 0.01;

	vec3  rayDir;
	float distanceKM;

	if (hasSurface) {
		float distM = length(relPos);
		rayDir = relPos / max(1e-3, distM);
		distanceKM = distM / 1000.0;
	} else {
		// No opaque geometry. Above water the SkyViewLUT already integrated the whole path via
		// SkyBackgroundNode -- leave that backdrop alone. Below water there's still a real, finite
		// water column between the camera and either the surface or a capped max distance.
		if (camAltKM >= waterLevelKM) {
			discard;
		}

		vec2 clip = inUV * 2.0 - 1.0;
		vec4 viewRay4 = uInvProjMatrix * vec4(clip, 1.0, 1.0);
		vec3 viewDir = viewRay4.xyz / viewRay4.w;
		rayDir = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

		float depthKM = waterLevelKM - camAltKM;
		distanceKM = (rayDir.y > 1e-4) ? min(depthKM / rayDir.y, push.maxUnderwaterDistanceKM)
										: push.maxUnderwaterDistanceKM;
	}

	AtmosphereRay ray;
	ray.originKM = vec3(0.0, kEarthRadius + camAltKM, 0.0);
	ray.direction = rayDir;
	ray.distanceKM = distanceKM;
	ray.sunDir = normalize(push.sunDir);
	ray.sunRadiance = push.sunRadiance;
	ray.moonDir = normalize(push.moonDir);
	ray.moonRadiance = push.moonRadiance;
	ray.multiScatScale = push.multiScatScale;
	ray.transmittanceIndex = push.transmittanceIndex;
	ray.multiScatteringIndex = push.multiScatteringIndex;
	ray.steps = push.stepCount;

	AtmosphereSample atmos = evaluateAtmosphere(ray);

	vec3 sceneRadiance = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;
	vec3 fogged = applyAtmosphere(atmos, sceneRadiance);

	// Converge distant fog toward the real SkyViewLUT radiance for this exact ray direction --
	// the standard aerial-perspective trick, and the fix for this march's color balance being
	// backwards (red fading least, not most) at long range: real multi-scattering makes distant
	// haze converge on the sky's own hue, but this per-pixel march's single-scattering term alone
	// doesn't reproduce that. Weighted by fog opacity (1 - transmittance luminance) so it has no
	// effect on close, unfogged geometry and only takes over once the surface is essentially lost
	// in haze. Only for real surface geometry seen from above water -- SkyViewLUT is computed as
	// if the camera sits at or above the ground (sky_view_lut.comp clamps camAltKM >= 0), so it's
	// not a valid convergence target for an underwater ray's own water-fog color.
	if (hasSurface && camAltKM >= waterLevelKM) {
		float fogOpacity = 1.0 - clamp(dot(atmos.transmittance, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
		vec3  skyColor = sampleSkyView(push.skyViewIndex, rayDir);
		fogged = mix(fogged, skyColor, fogOpacity * u_skyConvergence);
	}

	outColor = vec4(fogged, 1.0);
}
