#ifndef HELPERS_AMBIENT_GLSL
#define HELPERS_AMBIENT_GLSL

// Temporary stand-in for a real spherical-harmonics ambient probe grid. LightingUBO already
// reserves uSHCoeffs[81] for exactly that (shaders/lighting.glsl), but nothing bakes it yet --
// these three functions approximate the same ground/sky/horizon split a probe grid would
// give, directly from the sun direction, so callers can be written against the same shape
// (three directional colors, attenuated by how much cloud is above vs below the sample) and
// swapped onto real probe data later without restructuring the calling code.

vec3 getOutdoorAmbient(vec3 normal, vec3 sunDir) {
    // 1. Blend factors based on sun elevation (Y-up)
    float sunElev = sunDir.y;

    // Smoothstep ranges: night < -0.2 < twilight < 0.2 < day
    float dayMix      = smoothstep(-0.05, 0.20, sunElev);
    float nightMix    = 1.0 - smoothstep(-0.20, 0.00, sunElev);
    float twilightMix = 1.0 - max(dayMix, nightMix);

    // 2. Define sky states in Lux (approximate clear-sky physical values)
    // Day: Blue zenith, brighter hazy horizon (~15-20% of 100k direct sun)
    vec3 zenithDay  = vec3(0.15, 0.35, 0.70) * 15000.0;
    vec3 horizonDay = vec3(0.50, 0.60, 0.70) * 20000.0;

    // Twilight: Deep blue/purple zenith, vibrant orange/red horizon
    vec3 zenithTwilight  = vec3(0.02, 0.03, 0.10) * 500.0;
    vec3 horizonTwilight = vec3(0.80, 0.30, 0.10) * 1500.0;

    // Night: Starlight zenith, slight green-cyan airglow at the horizon
    vec3 zenithNight  = vec3(0.001, 0.002, 0.005) * 0.01;
    vec3 horizonNight = vec3(0.002, 0.005, 0.003) * 0.02;

    // 3. Directional sunset/sunrise glow
    // Keeps the horizon glowing where the sun just set, preventing uniform blackness
    float sunDot = max(0.0, dot(normal, sunDir));
    float horizonMask = 1.0 - abs(normal.y); // Restricts glow to horizontal normals
    float twilightGlowWeight = pow(sunDot, 4.0) * horizonMask * smoothstep(-0.2, 0.1, sunElev) * (1.0 - dayMix);
    horizonTwilight += vec3(1.0, 0.4, 0.1) * twilightGlowWeight * 4000.0;

    // 4. Interpolate current zenith and horizon states
    vec3 currentZenith  = zenithDay * dayMix + zenithTwilight * twilightMix + zenithNight * nightMix;
    vec3 currentHorizon = horizonDay * dayMix + horizonTwilight * twilightMix + horizonNight * nightMix;

    // 5. Calculate Ground Bounce (Average Albedo * Sky Irradiance)
    vec3 groundAlbedo = vec3(0.12, 0.15, 0.10); // Typical dirt/vegetation
    vec3 groundBounce = (currentZenith * 0.3 + currentHorizon * 0.7) * groundAlbedo;

    // 6. Hemispheric mapping based on surface normal
    // Using a split interpolation so the horizon color sits exactly at normal.y == 0
    vec3 skyHemisphere = mix(currentHorizon, currentZenith, max(normal.y, 0.0));
    vec3 lowerHemisphere = mix(groundBounce, currentHorizon, max(normal.y + 1.0, 0.0));

    // Step blends smoothly because both hemispheres equal currentHorizon at normal.y == 0
    return 0.01*mix(lowerHemisphere, skyHemisphere, step(0.0, normal.y));
}

vec3 getApproximateSkyAmbient(vec3 sunDir, vec3 sunRadiance) {
	return getOutdoorAmbient(vec3(0,1,0), sunDir);
	vec3  zenithColor = vec3(0.35, 0.55, 0.9);
	// Sky brightens with sun elevation (more scattered light overall) but never goes fully
	// dark at night -- a dim floor keeps night clouds from reading as pure black silhouettes.
	float skyBrightness = mix(0.08, 1.0, smoothstep(-0.1, 0.4, sunDir.y));
	return zenithColor * skyBrightness * max(sunRadiance.r, 0.05);
}

vec3 getApproximateGroundAmbient(vec3 sunDir, vec3 sunRadiance) {
	return getOutdoorAmbient(vec3(0,-1,0), sunDir);
	vec3  groundColor = vec3(0.25, 0.95, 0.15);
	float groundBrightness = mix(0.05, 0.6, clamp(sunRadiance.r / 1.0, 0.0, 10.0));
	return groundColor * groundBrightness;
}

// Horizon color shifts warm (toward the sun's own color) when the sun is near the horizon --
// real Rayleigh scattering reddens light travelling a long path through the atmosphere -- and
// settles toward a neutral sky/ground blend when the sun is high overhead or well below.
vec3 getApproximateHorizonAmbient(vec3 sunDir, vec3 normal, vec3 sunRadiance) {
    return getOutdoorAmbient(normal, sunDir);
}

vec3 getApproximateHorizonAmbient(vec3 sunDir, vec3 sunRadiance, vec3 skyAmbient, vec3 groundAmbient) {
	vec3  warmHorizon = vec3(1.0, 0.55, 0.3) * max(sunRadiance.r, 0.05);
	float horizonCloseness = 1.0 - smoothstep(0.0, 0.35, abs(sunDir.y));
	vec3  neutralHorizon = mix(groundAmbient, skyAmbient, 0.6);
	return mix(neutralHorizon, warmHorizon, horizonCloseness);
}

#endif // HELPERS_AMBIENT_GLSL
