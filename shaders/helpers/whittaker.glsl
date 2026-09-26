#ifndef WHITTAKER_GLSL
#define WHITTAKER_GLSL

// Whittaker Biome Classification Mapping
// Maps temperature [0, 1] (cold to hot) and moisture [0, 1] (dry to wet) to Whittaker biome types and shading styles.

struct WhittakerBiome {
	vec3  color;
	float roughness;
	float biomeIndex; // 0..9 index
	float snowCover;
};

WhittakerBiome evaluateWhittakerBiome(float temp, float moisture, float severity, float precipType) {
	WhittakerBiome b;
	b.snowCover = clamp((0.35 - temp) * 3.0 + (precipType > 0.5 ? 0.3 : 0.0), 0.0, 1.0);

	// Temperature zones:
	// T < 0.2: Polar / Alpine
	// T < 0.4: Subarctic / Boreal
	// T < 0.7: Temperate
	// T >= 0.7: Tropical / Subtropical

	if (temp < 0.2) {
		if (moisture < 0.3) {
			// Tundra
			b.color = mix(vec3(0.55, 0.53, 0.45), vec3(0.48, 0.50, 0.40), moisture * 3.0);
			b.roughness = 0.85;
			b.biomeIndex = 1.0;
		} else {
			// Ice / Glacier / Snow
			b.color = vec3(0.92, 0.95, 0.98);
			b.roughness = 0.3;
			b.biomeIndex = 0.0;
		}
	} else if (temp < 0.4) {
		if (moisture < 0.3) {
			// Cold Desert / Taiga Scrub
			b.color = vec3(0.50, 0.48, 0.38);
			b.roughness = 0.80;
			b.biomeIndex = 3.0;
		} else if (moisture < 0.6) {
			// Taiga / Boreal Forest
			b.color = vec3(0.12, 0.28, 0.16);
			b.roughness = 0.75;
			b.biomeIndex = 2.0;
		} else {
			// Moist Taiga / Bogs
			b.color = vec3(0.08, 0.24, 0.18);
			b.roughness = 0.70;
			b.biomeIndex = 2.0;
		}
	} else if (temp < 0.7) {
		if (moisture < 0.25) {
			// Temperate Grassland / Steppe
			b.color = vec3(0.62, 0.58, 0.35);
			b.roughness = 0.80;
			b.biomeIndex = 3.0;
		} else if (moisture < 0.5) {
			// Temperate Woodland / Shrubland
			b.color = vec3(0.35, 0.45, 0.22);
			b.roughness = 0.75;
			b.biomeIndex = 4.0;
		} else if (moisture < 0.75) {
			// Temperate Deciduous Forest
			b.color = vec3(0.18, 0.42, 0.15);
			b.roughness = 0.70;
			b.biomeIndex = 5.0;
		} else {
			// Temperate Rainforest
			b.color = vec3(0.06, 0.35, 0.12);
			b.roughness = 0.65;
			b.biomeIndex = 6.0;
		}
	} else {
		if (moisture < 0.2) {
			// Subtropical Desert
			b.color = vec3(0.82, 0.68, 0.45);
			b.roughness = 0.90;
			b.biomeIndex = 7.0;
		} else if (moisture < 0.55) {
			// Tropical Seasonal Forest / Savanna
			b.color = vec3(0.48, 0.52, 0.20);
			b.roughness = 0.80;
			b.biomeIndex = 8.0;
		} else {
			// Tropical Rainforest
			b.color = vec3(0.04, 0.38, 0.08);
			b.roughness = 0.60;
			b.biomeIndex = 9.0;
		}
	}

	// Apply thunderstorm severity weathering (darker lushness or storm darkening)
	b.color = mix(b.color, b.color * 0.75 + vec3(0.02, 0.05, 0.08), severity * 0.4);

	// Blend snow cover for cold temperatures / high altitude precipitation
	if (b.snowCover > 0.01) {
		b.color = mix(b.color, vec3(0.92, 0.95, 0.98), b.snowCover);
		b.roughness = mix(b.roughness, 0.35, b.snowCover);
	}

	return b;
}

#endif // WHITTAKER_GLSL
