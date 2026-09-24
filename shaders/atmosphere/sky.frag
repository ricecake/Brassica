#version 460
#include "bindless.glsl"
#include "lighting.glsl"
#include "common.glsl"
#include "helpers/astral.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform SkyPushConstants {
	vec4 sunDirAndAureole;      // xyz = sunDir, w = sunAureole
	vec4 moonDirAndCirrus;     // xyz = moonDir, w = cirrus
	vec4 sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	float worldScale;
	uint  skyViewIndex;
	uint  transmittanceIndex;
	float padding;
} push;

const float solar_flare_speed = 0.25;
const float solar_flare_scale = 0.35;
const float solar_flare_strength = 0.25;
const float cirrusOpacity = 0.01250;

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

vec2 sky_hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

// Distance-to-edge Voronoi for sharp solar flare loop/filament boundaries
float voronoiDistanceToEdgeSky(vec2 x) {
    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 mg, mr;
    float md = 8.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = sky_hash22(n + g);
            o = 0.5 + 0.5 * sin(uTime * 0.3 * solar_flare_speed + 6.2831 * o);
            vec2 r = g + o - f;
            float d = dot(r, r);

            if (d < md) {
                md = d;
                mr = r;
                mg = g;
            }
        }
    }

    md = 8.0;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            vec2 g = mg + vec2(float(i), float(j));
            vec2 o = sky_hash22(n + g);
            o = 0.5 + 0.5 * sin(uTime * 0.3 * solar_flare_speed + 6.2831 * o);
            vec2 r = g + o - f;

            if (dot(mr - r, mr - r) > 0.00001) {
                md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
            }
        }
    }
    return md;
}

void main() {
	vec2 clipCoord = inUV * 2.0 - 1.0;
	vec4 viewRay = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
	vec3 worldRay = normalize((uInvViewMatrix * vec4(viewRay.xy, -1.0, 0.0)).xyz);

	float worldScale = max(0.001, push.worldScale);
	float planetRadius = FAKE_PLANET_RADIUS / 1000.0;
	float r = planetRadius + max(0.0, uCameraPosition.y / (1000.0 * worldScale));

	// Calculate the zenith cosine (mu) of the true curved horizon
	float muHorizon = -sqrt(max(0.0, 1.0 - (planetRadius * planetRadius) / (r * r)));

	// 1. Sky Radiance from SkyViewLUT
	vec3 skyRadiance = sampleSkyView(push.skyViewIndex, worldRay);

	// 2. Sun Disc with Aureole
	vec3  sunDir = normalize(push.sunDirAndAureole.xyz);
	vec3  sunRadiance = push.sunRadianceAndSkyExp.xyz;
	float sunAureoleStrength = push.sunDirAndAureole.w;

	vec3 sunUp = vec3(0, 1, 0);
	vec3 sunRight = normalize(cross(sunUp, sunDir));
	if (length(sunRight) < 0.001)
		sunRight = vec3(1, 0, 0);
	sunUp = cross(sunDir, sunRight);

	float rayLocalX = dot(worldRay, sunRight);
	float rayLocalY = dot(worldRay, sunUp);
	float rayLocalZ = dot(worldRay, sunDir);

	float sunHorizonDist = max(0.0, sunDir.y - muHorizon);
	float atmosphericRefraction = 1.0 + 0.6 * exp(-sunHorizonDist * 10.0);
	float flattenedY = rayLocalY * atmosphericRefraction;

	float distSq = rayLocalX * rayLocalX + flattenedY * flattenedY;
	float distToSun = sqrt(max(0.0, distSq));

	float sunAngularRadius = 0.02; // ~1 degree
	float aureole = exp(-distToSun * 40.0) * sunAureoleStrength * 3.5;

	float sunMask = 1.0 - smoothstep(
		(sunAngularRadius - 0.001) * (sunAngularRadius - 0.001),
		sunAngularRadius * sunAngularRadius,
		distSq
	);
	sunMask += aureole;

	// Add dramatically oversized solar flares/prominences flowing radially outwards
	float solarFlares = 0.0;
	if (true) {
		float theta = atan(rayLocalY, rayLocalX);
		float r_sun = length(vec2(rayLocalX, rayLocalY));

		// Warp polar components with Simplex noise for turbulent plasma motion
		vec3 warpPos = vec3(rayLocalX * 25.0 * solar_flare_scale, rayLocalY * 25.0 * solar_flare_scale, uTime * 0.1 * solar_flare_speed);
		float angleWarp = snoise3d(warpPos) * 0.5;
		float distWarp = snoise3d(warpPos + vec3(19.0, 29.0, 37.0)) * 0.06;

		float warpedTheta = theta + angleWarp;
		float warpedR = r_sun + distWarp;

		// Map to a radial Voronoi cell space, moving outward with time
		vec2 cellCoords = vec2(warpedTheta * 6.5, (warpedR - uTime * 0.06 * solar_flare_speed) * 18.0 * solar_flare_scale);
		float voronoiDist = voronoiDistanceToEdgeSky(cellCoords);

		// Create sharp filaments and thick prominence loops
		float filament = 1.0 - smoothstep(0.0, 0.09, voronoiDist);
		float loopArc = smoothstep(0.04, 0.45, voronoiDist);
		float prominence = max(filament * 0.95, loopArc * 0.2);

		// Radial decay starting from the sun surface (sunAngularRadius)
		// Oversized flares: extend decay range
		float flareDecay = exp(-max(0.0, r_sun - sunAngularRadius) * (14.0 / solar_flare_scale));

		solarFlares = prominence * flareDecay * solar_flare_strength * 2.0;
	}
	sunMask += solarFlares;

	sunMask = (sunMask * (1.0 + sunMask * 0.05)) / (1.0 + sunMask * 0.06);
	sunMask *= step(0.0, rayLocalZ);

	float sunFade = smoothstep(muHorizon - 0.015, muHorizon + 0.005, sunDir.y);
	vec3 sunTransmittance = max(getTransmittance(r, sunDir.y), vec3(0.001));
	vec3 sunDisc = sunRadiance * sunMask * sunTransmittance * sunFade;

	// 3. Stars and Nebula
	float skyBrightness = max(max(skyRadiance.r, skyRadiance.g), skyRadiance.b);
	float starVisibility = 1.0 - smoothstep(0.05, 0.5, skyBrightness);
	vec3 spaceBackground = vec3(0.0);

	if (starVisibility > 0.0) {
		vec3 skyTransmittance = getTransmittance(r, worldRay.y);
		if (any(greaterThan(skyTransmittance, vec3(0.0)))) {
			vec3 stars = computeStars(worldRay, uTime);
			vec3 nebula = computeNebula(worldRay, uTime);
			spaceBackground = (stars + nebula) * skyTransmittance * starVisibility;
		}
	}

	// 4. Moon Disc
	vec3  moonDir = normalize(push.moonDirAndCirrus.xyz);
	vec3  moonRadiance = vec3(0.12, 0.14, 0.18) * push.sunRadianceAndSkyExp.w;
	float moonAngularRadius = 0.018;

	vec3 moonUp = vec3(0, 1, 0);
	vec3 moonRight = normalize(cross(moonUp, moonDir));
	if (length(moonRight) < 0.001)
		moonRight = vec3(1, 0, 0);
	moonUp = cross(moonDir, moonRight);

	float moonLocalX = dot(worldRay, moonRight);
	float moonLocalY = dot(worldRay, moonUp);
	float moonLocalZ = dot(worldRay, moonDir);

	float moonHorizonDist = max(0.0, moonDir.y - muHorizon);
	float moonFlattenedY = moonLocalY * (1.0 + 0.6 * exp(-moonHorizonDist * 10.0));
	float moonDistSq = moonLocalX * moonLocalX + moonFlattenedY * moonFlattenedY;

	float moonMask = smoothstep(
		moonAngularRadius * moonAngularRadius,
		(moonAngularRadius - 0.001) * (moonAngularRadius - 0.001),
		moonDistSq
	);
	moonMask *= step(0.99, moonLocalZ);

	float r2 = dot(
		vec2(moonLocalX, moonFlattenedY) / moonAngularRadius,
		vec2(moonLocalX, moonFlattenedY) / moonAngularRadius
	);
	float z = sqrt(max(0.0, 1.0 - r2));
	vec3  moonNormal = normalize(moonLocalX * moonRight + moonFlattenedY * moonUp + z * moonDir);
	float moonIllum = max(0.0, dot(moonNormal, sunDir)) * 5.8;
	float phasedMask = moonMask * mix(0.02, 1.0, moonIllum);

	float moonFade = smoothstep(muHorizon - 0.015, muHorizon + 0.005, moonDir.y);
	vec3 moonTransmittance = max(getTransmittance(r, moonDir.y), vec3(0.001));
	vec3 moonDisc = moonRadiance * phasedMask * moonTransmittance * moonFade;

	// 5. Cirrus Cloud Layer
	vec3 cirrusColor = vec3(0.0);
	if (true) {
		float cirrusAlt = 10.0; // 10 km altitude
		float cloudRadius = planetRadius + cirrusAlt;

		float b = 2.0 * r * worldRay.y;
		float c = (r * r) - (cloudRadius * cloudRadius);
		float det = (b * b) - (4.0 * c);

		if (det > 0.0) {
			float sqrtDet = sqrt(det);
			float t1 = (-b - sqrtDet) * 0.5;
			float t2 = (-b + sqrtDet) * 0.5;

			float t_cirrus = (t1 > 0.0) ? t1 : t2;

			if (t_cirrus > 0.0) {
				vec3 p_cirrus = uCameraPosition.xyz + worldRay * (t_cirrus * 1000.0 * worldScale);

				vec3 advect = vec3(1.0, 0.0, 1.0) * uTime * 0.5;
				vec2 uv_cirrus = (p_cirrus.xz + advect.xz) * (0.00005 / worldScale);

				float n = (fbm_astral(vec3(uv_cirrus * 2.0, uTime * 0.01)) + 1.0) * 0.5;
				float n2 = (fbm_astral(vec3(uv_cirrus * 5.0, uTime * 0.02 + 10.0)) + 1.0) * 0.5;
				float noise = smoothstep(0.3, 0.8, n * n2);

				vec3 T_cirrus = max(getTransmittance(planetRadius + cirrusAlt, sunDir.y), vec3(0.001));
				float cirrusPhase = mix(0.2, 1.0, pow(max(0.0, dot(worldRay, sunDir)), 3.0));

				vec3 cirrusLighting = (T_cirrus * sunRadiance * cirrusPhase * 5.0) + (skyRadiance * 0.5);
				cirrusColor = cirrusLighting * noise * cirrusOpacity * 15.0;

				float opticalDepthFade = exp(-t_cirrus * 0.0025);
				cirrusColor *= opticalDepthFade;
			}
		}
	}

	vec3 finalColor = skyRadiance + sunDisc + moonDisc + cirrusColor + spaceBackground;

	outColor = vec4(finalColor, 1.0);
}
