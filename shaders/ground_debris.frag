#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "terrain.glsl"
#include "helpers/wind.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in float inHeightFactor;

layout(push_constant) uniform GroundDebrisPushConstants {
	uvec4 gridParams;        // x = tilesPerSide, y = totalTiles, z = reserved, w = textureDim
	vec4  cameraPos;         // camera world position xyz, w = camera altitude above ground
	uint  clipmapIndex;
	uint  minMaxIndex;
	float maxGroundDistance; // max camera altitude above ground to render
	float tileSize;          // world size per tile
	float heightOffset;      // height offset above terrain
	float time;
	uint  biomeIndex;
	uint  weatherBiomeIndex;
} push;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

// Helper to evaluate a seashell / bivalve shell near water shorelines
float evaluateShellShape(vec2 uv, vec2 cellID, out float shellRidge, out float shellTint) {
	vec2 h1 = hash(cellID * 29.41 + vec2(7.3, 1.8));
	vec2 h2 = hash(cellID * 61.17 + vec2(3.5, 8.2));
	vec2 h3 = hash(cellID * 83.29 + vec2(4.1, 9.6));

	shellTint = h1.y;

	// Center jitter in cell
	vec2 center = (h1 - 0.5) * 0.7;
	vec2 diff = uv - center;

	// Orientation
	float angle = h2.x * 6.2831853;
	float s = sin(angle);
	float c = cos(angle);
	mat2  rot = mat2(c, -s, s, c);
	vec2  localP = rot * diff;

	// Fan-shaped seashell: wider top, tapering hinge base
	float r = length(localP);
	float theta = atan(localP.y, localP.x);
	float fan = mix(0.12, 0.24, h2.y) * (0.8 + 0.3 * cos(theta));

	// Concentric and radial growth ridges
	shellRidge = sin(r * 45.0 + h3.x * 6.28) * 0.5 + 0.5;
	float radialRidges = cos(theta * 7.0) * 0.5 + 0.5;
	shellRidge = mix(shellRidge, radialRidges, 0.4);

	return r / max(0.01, fan);
}

// Helper to evaluate small pebbles / stones in sandy / rocky terrain
float evaluatePebbleShape(vec2 uv, vec2 cellID, out float stoneRoundness) {
	vec2 h1 = hash(cellID * 37.19 + vec2(4.8, 6.2));
	vec2 h2 = hash(cellID * 71.53 + vec2(2.1, 5.7));

	vec2 center = (h1 - 0.5) * 0.75;
	vec2 diff = uv - center;

	float angle = h2.x * 6.2831853;
	float s = sin(angle);
	float c = cos(angle);
	mat2  rot = mat2(c, -s, s, c);
	vec2  localP = rot * diff;

	// Ellipsoidal pebble with irregular facets
	float aspect = mix(0.65, 0.95, h2.y);
	vec2 scaled = localP / vec2(1.0, aspect);
	float d = length(scaled);

	float radius = mix(0.08, 0.20, h1.y);
	stoneRoundness = h2.y;
	return d / max(0.01, radius);
}

// Helper to evaluate a stick / twig segment in a 2D tile cell with chaotic spatial jitter and orientation
float evaluateStickSegment(vec2 uv, vec2 cellID, out float stickWidth) {
	vec2  h1 = hash(cellID * 17.13 + vec2(1.2, 5.7));
	vec2  h2 = hash(cellID * 31.81 + vec2(8.4, 2.1));
	vec2  h3 = hash(cellID * 53.47 + vec2(4.9, 9.3));

	// Chaotic orientation angle in [0, 2PI] with local rotational jitter
	float angle = (h1.x * 6.2831853) + (h3.x - 0.5) * 1.5;
	float len = mix(0.25, 0.65, h1.y);
	vec2  dir = vec2(cos(angle), sin(angle));

	// Chaotic center translation offset across the cell [-0.65, 0.65] to thoroughly break grid alignment
	vec2  center = (h2 - 0.5) * 0.75;
	vec2  p0 = center - dir * (len * 0.5);
	vec2  p1 = center + dir * (len * 0.5);

	// Add slight curvature / crook to the stick segment for realism
	vec2  mid = center + vec2(-dir.y, dir.x) * ((h3.y - 0.5) * 0.08);
	stickWidth = mix(0.015, 0.038, h2.y);

	float d0 = sdLine(uv, p0, mid);
	float d1 = sdLine(uv, mid, p1);
	return min(d0, d1);
}

// Helper to evaluate fallen leaves with chaotic orientation, curling, and position jitter
float evaluateLeafShape(vec2 uv, vec2 cellID, out float leafHue) {
	vec2  h1 = hash(cellID * 43.17 + vec2(9.1, 3.4));
	vec2  h2 = hash(cellID * 57.29 + vec2(4.7, 7.8));
	vec2  h3 = hash(cellID * 79.63 + vec2(2.5, 6.1));

	leafHue = fract(h1.y + h3.x * 0.3);

	// Substantial chaotic positional jitter across cell
	vec2 center = (h1 - 0.5) * 0.8;
	vec2 diff = uv - center;

	// Fully chaotic rotation angle in [0, 2PI]
	float angle = (h2.x * 6.2831853) + (h3.y * 3.14159);
	float s = sin(angle);
	float c = cos(angle);
	mat2  rot = mat2(c, -s, s, c);
	vec2  localP = rot * diff;

	// Slight leaf curvature / bending
	localP.y += (localP.x * localP.x) * (h3.x - 0.5) * 1.8;

	// Varied leaf aspect ratio and length
	float leafSize = mix(0.09, 0.22, h2.y);
	float leafAspect = mix(0.35, 0.60, h3.y);
	vec2  scaled = localP / vec2(leafSize, leafSize * leafAspect);
	float dist = length(scaled);
	return dist;
}

void main() {
	// Global camera proximity fade
	float distToCam = length(inWorldPos.xz - uCameraPosition.xz);
	float distanceFade = smoothstep(45.0, 15.0, distToCam) * inHeightFactor;
	if (distanceFade <= 0.001) {
		discard;
	}

	vec2 p = inWorldPos.xz;

	// Add continuous organic domain warping to break any lingering grid alignment
	vec2 warp = vec2(
		dot_noise_fbm(vec3(p * 0.4, 0.5), 2.0, 0.0),
		dot_noise_fbm(vec3(p * 0.4 + vec2(5.2, 1.3), 1.7), 2.0, 2.0)
	) * 0.35;
	vec2 warpedP = p + warp;

	// Multi-frequency macro distribution: debris clusters around natural patches
	float clusterNoise = dot_noise_fbm(vec3(warpedP * 0.12, 0.0), 2.0, 0.0) * 0.6 + dot_noise_fbm(vec3(warpedP * 0.35, 1.5), 2.0, 1.0) * 0.4;
	float clusterMask = smoothstep(0.0, 0.38, clusterNoise);
	if (clusterMask <= 0.05) {
		discard;
	}

	// 1. Sticks / Twigs evaluation with 2-level neighborhood & chaotic jitter
	float stickScale = 3.2; // ~0.3m cell size
	vec2  stickGrid = warpedP * stickScale;
	vec2  stickCell = floor(stickGrid);
	vec2  stickFrac = fract(stickGrid) - 0.5;

	float minStickDist = 100.0;
	float activeStickWidth = 0.02;

	// Check neighboring cells for overlapping sticks
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			vec2 neighbor = vec2(float(dx), float(dy));
			vec2 cID = stickCell + neighbor;
			float w;
			float d = evaluateStickSegment(stickFrac - neighbor, cID, w);
			if (d < minStickDist) {
				minStickDist = d;
				activeStickWidth = w;
			}
		}
	}

	bool isStick = minStickDist < activeStickWidth;
	float stickEdge = smoothstep(activeStickWidth, activeStickWidth * 0.7, minStickDist);

	// 2. Fallen Leaves evaluation with chaotic orientations & offsets
	float leafScale = 4.8; // ~0.2m cell size
	vec2  leafGrid = (warpedP + vec2(17.3, 31.9)) * leafScale;
	vec2  leafCell = floor(leafGrid);
	vec2  leafFrac = fract(leafGrid) - 0.5;

	float minLeafDist = 100.0;
	float chosenHue = 0.0;

	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			vec2 neighbor = vec2(float(dx), float(dy));
			vec2 cID = leafCell + neighbor;
			float hue;
			float d = evaluateLeafShape(leafFrac - neighbor, cID, hue);
			if (d < minLeafDist) {
				minLeafDist = d;
				chosenHue = hue;
			}
		}
	}

	bool isLeaf = minLeafDist < 1.0;
	float leafEdge = smoothstep(1.0, 0.78, minLeafDist);

	// 3. Small Brush / Fine needle fibers with chaotic jitter & varied orientation
	float brushScale = 11.0;
	vec2  brushGrid = (warpedP + vec2(43.7, 8.2)) * brushScale;
	vec2  brushCell = floor(brushGrid);
	vec2  brushFrac = fract(brushGrid) - 0.5;
	vec2  hBrush = hash(brushCell * 73.11 + vec2(3.3, 1.9));
	vec2  brushCenter = (hBrush - 0.5) * 0.7;
	float brushDist = length(brushFrac - brushCenter);
	bool  isBrush = (hBrush.x > 0.62) && (brushDist < mix(0.12, 0.32, hBrush.y));

	// 4. Sample biome data to determine environment conditions (water proximity, sand/rock substrate)
	float waterLevel = 0.0;
	float substrateSand = 0.0;
	float moisture = 0.5;
	if (push.biomeIndex > 0u && push.biomeIndex != 0xFFFFFFFFu) {
		vec4 biomeData = sampleTerrainBiome(push.biomeIndex, p, 0u, push.gridParams.w);
		waterLevel = biomeData.g;
		substrateSand = biomeData.a; // substrate map (sand/gravel vs soil)
	}
	if (push.weatherBiomeIndex > 0u && push.weatherBiomeIndex != 0xFFFFFFFFu) {
		vec4 weatherData = sampleTerrainWeatherBiome(push.weatherBiomeIndex, inWorldPos);
		moisture = weatherData.b;
	}

	// Height above water level: shoreline proximity
	float heightAboveWater = inWorldPos.y - waterLevel;
	bool nearWater = (heightAboveWater >= -0.2 && heightAboveWater <= 4.0);
	bool isSandy = (substrateSand > 0.45) || (nearWater && heightAboveWater <= 2.5);

	// 5. Seashells evaluation near water shorelines
	float shellScale = 6.0;
	vec2  shellGrid = (warpedP + vec2(64.1, 19.4)) * shellScale;
	vec2  shellCell = floor(shellGrid);
	vec2  shellFrac = fract(shellGrid) - 0.5;
	float minShellDist = 100.0;
	float chosenRidge = 0.0;
	float chosenTint = 0.0;

	if (nearWater) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				vec2 neighbor = vec2(float(dx), float(dy));
				vec2 cID = shellCell + neighbor;
				float ridge, tint;
				float d = evaluateShellShape(shellFrac - neighbor, cID, ridge, tint);
				if (d < minShellDist) {
					minShellDist = d;
					chosenRidge = ridge;
					chosenTint = tint;
				}
			}
		}
	}
	bool isShell = nearWater && (minShellDist < 1.0);
	float shellEdge = smoothstep(1.0, 0.82, minShellDist);

	// 6. Small Pebbles / Rocks in sandy / beach / gravel areas
	float pebbleScale = 5.2;
	vec2  pebbleGrid = (warpedP + vec2(88.3, 47.9)) * pebbleScale;
	vec2  pebbleCell = floor(pebbleGrid);
	vec2  pebbleFrac = fract(pebbleGrid) - 0.5;
	float minPebbleDist = 100.0;
	float stoneRoundness = 0.5;

	if (isSandy) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				vec2 neighbor = vec2(float(dx), float(dy));
				vec2 cID = pebbleCell + neighbor;
				float roundness;
				float d = evaluatePebbleShape(pebbleFrac - neighbor, cID, roundness);
				if (d < minPebbleDist) {
					minPebbleDist = d;
					stoneRoundness = roundness;
				}
			}
		}
	}
	bool isPebble = isSandy && (minPebbleDist < 1.0);
	float pebbleEdge = smoothstep(1.0, 0.75, minPebbleDist);

	// In sandy / water areas, prioritize shells and pebbles; reduce dense forest leaves
	if (nearWater || isSandy) {
		isLeaf = isLeaf && (!isShell && !isPebble);
		isStick = isStick && (!isShell && !isPebble);
	}

	// If neither stick, leaf, brush, shell, nor pebble, this spot is transparent ground
	if (!isStick && !isLeaf && !isBrush && !isShell && !isPebble) {
		discard;
	}

	// Calculate base color and material properties
	vec3  albedo = vec3(0.0);
	float roughness = 0.8;
	float metallic = 0.0;
	vec3  perturbedNormal = inNormal;

	if (isShell) {
		// Seashell coloration: pearlescent ivory, peach, soft rose-pink, cream with fine growth ridges
		vec3 shellWhite = vec3(0.92, 0.88, 0.82);
		vec3 shellPeach = vec3(0.95, 0.81, 0.71);
		vec3 shellRose  = vec3(0.88, 0.75, 0.74);
		vec3 shellColor = mix(shellWhite, mix(shellPeach, shellRose, chosenTint), 0.5);
		shellColor = mix(shellColor * 0.82, shellColor * 1.12, chosenRidge);
		albedo = shellColor * shellEdge;
		roughness = 0.35; // smooth seashell finish
		metallic = 0.05;
	} else if (isPebble) {
		// Small pebble / rock coloration: granite, quartzite, slate, ochre sandstone
		vec3 stoneDark  = vec3(0.32, 0.30, 0.28);
		vec3 stoneLight = vec3(0.55, 0.52, 0.48);
		vec3 sandStone  = vec3(0.62, 0.53, 0.38);
		vec3 stoneBase = mix(stoneDark, stoneLight, stoneRoundness);
		if (substrateSand > 0.6) {
			stoneBase = mix(stoneBase, sandStone, 0.6);
		}
		// Fine mineral grain
		float mineralGrain = fract(sin(dot(p, vec2(93.1, 417.3))) * 43758.5453);
		albedo = mix(stoneBase * 0.9, stoneBase * 1.1, mineralGrain) * pebbleEdge;
		roughness = 0.82;
	} else if (isStick) {
		// Wood / Bark tones: rich brown with subtle bark striations
		vec3 barkDark = vec3(0.18, 0.11, 0.06);
		vec3 barkLight = vec3(0.32, 0.22, 0.13);
		float grain = fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
		albedo = mix(barkDark, barkLight, grain * 0.5 + 0.5) * stickEdge;
		roughness = 0.88;
	} else if (isLeaf) {
		// Leaf coloration: palette ranging from autumn ochre/russet to dry olive green
		vec3 leafBrown = vec3(0.36, 0.22, 0.08);
		vec3 leafOchre = vec3(0.48, 0.33, 0.12);
		vec3 leafOlive = vec3(0.25, 0.32, 0.14);
		vec3 leafDry = vec3(0.42, 0.28, 0.16);

		if (chosenHue < 0.3) {
			albedo = mix(leafBrown, leafOchre, chosenHue / 0.3);
		} else if (chosenHue < 0.7) {
			albedo = mix(leafOchre, leafOlive, (chosenHue - 0.3) / 0.4);
		} else {
			albedo = mix(leafOlive, leafDry, (chosenHue - 0.7) / 0.3);
		}
		albedo *= leafEdge;
		roughness = 0.65;
	} else if (isBrush) {
		// Small brush tufts / lichen
		vec3 brushGreen = vec3(0.20, 0.30, 0.12);
		vec3 brushDry = vec3(0.32, 0.27, 0.15);
		albedo = mix(brushGreen, brushDry, hBrush.y);
		roughness = 0.75;
	}

	// Bayer dither for smooth boundary transparency and distance fading
	float dither = bayerDither4x4(gl_FragCoord.xy);
	if (dither >= distanceFade * clusterMask) {
		discard;
	}

	vec3 relPos = inWorldPos - uCameraPosition.xyz;

	// G-Buffer Outputs
	outPosition = vec4(relPos, 0.0); // .a = 0.0 (non-emissive)
	outNormal = vec4(normalize(perturbedNormal), roughness);
	outAlbedo = vec4(albedo, 1.0);
	outMaterial = vec4(metallic, roughness, 0.0, 1.0);
}
