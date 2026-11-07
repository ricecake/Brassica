#version 460
#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in float inHeightFactor;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

// Helper to evaluate a stick / twig segment in a 2D tile cell
float evaluateStickSegment(vec2 uv, vec2 cellID, out float stickWidth) {
	vec2  h1 = hash(cellID * 17.13 + vec2(1.2, 5.7));
	vec2  h2 = hash(cellID * 31.81 + vec2(8.4, 2.1));

	// Random stick endpoints in cell [-0.4, 0.4]
	float angle = h1.x * 6.2831853;
	float len = mix(0.18, 0.42, h1.y);
	vec2  dir = vec2(cos(angle), sin(angle));
	vec2  center = (h2 - 0.5) * 0.4;
	vec2  p0 = center - dir * (len * 0.5);
	vec2  p1 = center + dir * (len * 0.5);

	stickWidth = mix(0.015, 0.035, h2.y);
	float d = sdLine(uv, p0, p1);
	return d;
}

// Helper to evaluate fallen leaves (elliptical patches with rotation)
float evaluateLeafShape(vec2 uv, vec2 cellID, out float leafHue) {
	vec2  h1 = hash(cellID * 43.17 + vec2(9.1, 3.4));
	vec2  h2 = hash(cellID * 57.29 + vec2(4.7, 7.8));

	leafHue = h1.y;
	vec2 center = (h1 - 0.5) * 0.45;
	vec2 diff = uv - center;

	float angle = h2.x * 6.2831853;
	float s = sin(angle);
	float c = cos(angle);
	mat2  rot = mat2(c, -s, s, c);
	vec2  localP = rot * diff;

	// Leaf aspect ratio (elongated oval)
	float leafSize = mix(0.08, 0.16, h2.y);
	vec2  scaled = localP / vec2(leafSize, leafSize * 0.45);
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

	// Multi-frequency macro distribution: debris clusters around trees/brush patches
	// using FBM noise to ensure patchy, non-uniform coverage
	vec2 p = inWorldPos.xz;
	float clusterNoise = dot_noise_fbm(vec3(p * 0.12, 0.0), 2.0, 0.0) * 0.6 + dot_noise_fbm(vec3(p * 0.35, 1.5), 2.0, 1.0) * 0.4;
	// Threshold to create clear patches where debris is absent (transparency / emptiness)
	float clusterMask = smoothstep(0.0, 0.4, clusterNoise);
	if (clusterMask <= 0.05) {
		discard;
	}

	// 1. Sticks / Twigs evaluation across grid cells
	float stickScale = 3.5; // ~0.3m cell size
	vec2  stickGrid = p * stickScale;
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

	// 2. Fallen Leaves evaluation
	float leafScale = 5.0; // ~0.2m cell size
	vec2  leafGrid = p * leafScale;
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
	float leafEdge = smoothstep(1.0, 0.8, minLeafDist);

	// 3. Small Brush / Fine pine needle fibers
	float brushScale = 12.0;
	vec2  brushGrid = p * brushScale;
	vec2  brushCell = floor(brushGrid);
	vec2  brushFrac = fract(brushGrid) - 0.5;
	vec2  hBrush = hash(brushCell * 73.11 + vec2(3.3, 1.9));
	float brushDist = length(brushFrac);
	bool  isBrush = (hBrush.x > 0.65) && (brushDist < mix(0.15, 0.35, hBrush.y));

	// If neither stick, leaf, nor brush, this spot is transparent ground
	if (!isStick && !isLeaf && !isBrush) {
		discard;
	}

	// Calculate base color and material properties
	vec3  albedo = vec3(0.0);
	float roughness = 0.8;
	float metallic = 0.0;
	vec3  perturbedNormal = inNormal;

	if (isStick) {
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
