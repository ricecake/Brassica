#ifndef WHITTAKER_GLSL
#define WHITTAKER_GLSL

// Whittaker Biome Classification Mapping
// Maps temperature [0, 1] (cold to hot) and moisture [0, 1] (dry to wet) to Whittaker biome types and shading styles.

struct WhittakerBiome {
	vec3  color;
	float roughness;
	float biomeIndex; // 0..9 index
	float snowCover;
	vec3  palA;
	vec3  palB;
	vec3  palC;
	vec3  palD;
};

// Cosine color palette evaluation: color(t) = a + b * cos(2 * PI * (c * t + d))
vec3 evalCosinePalette(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
	return clamp(a + b * cos(6.28318530718 * (c * t + d)), 0.0, 1.0);
}

// Returns cosine palette parameters (a, b, c, d) for a Whittaker biome index (0..9)
// Parameterized such that a + b corresponds to the top color and a - b to the bottom color matching FoliageManager
void getWhittakerBiomePalette(uint biomeIdx, out vec3 a, out vec3 b, out vec3 c, out vec3 d) {
	uint idx = min(biomeIdx, 9u);
	if (idx == 0u) { // Ice / Glacier / Snow (top: 0.80, 0.85, 0.80; bottom: 0.55, 0.60, 0.55)
		a = vec3(0.675, 0.725, 0.675);
		b = vec3(0.125, 0.125, 0.125);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.50, 0.48, 0.44);
	} else if (idx == 1u) { // Tundra (top: 0.45, 0.48, 0.30; bottom: 0.20, 0.22, 0.12)
		a = vec3(0.325, 0.350, 0.210);
		b = vec3(0.125, 0.130, 0.090);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.48, 0.54, 0.46);
	} else if (idx == 2u) { // Taiga / Boreal Forest (top: 0.18, 0.40, 0.18; bottom: 0.07, 0.20, 0.08)
		a = vec3(0.125, 0.300, 0.130);
		b = vec3(0.055, 0.100, 0.050);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.52, 0.55, 0.48);
	} else if (idx == 3u) { // Cold Desert / Temperate Grassland (top: 0.62, 0.58, 0.32; bottom: 0.30, 0.26, 0.12)
		a = vec3(0.460, 0.420, 0.220);
		b = vec3(0.160, 0.160, 0.100);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.48, 0.52, 0.44);
	} else if (idx == 4u) { // Temperate Woodland / Shrubland (top: 0.32, 0.52, 0.20; bottom: 0.12, 0.28, 0.08)
		a = vec3(0.220, 0.400, 0.140);
		b = vec3(0.100, 0.120, 0.060);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.50, 0.55, 0.45);
	} else if (idx == 5u) { // Temperate Deciduous Forest (top: 0.22, 0.55, 0.18; bottom: 0.08, 0.30, 0.06)
		a = vec3(0.150, 0.425, 0.120);
		b = vec3(0.070, 0.125, 0.060);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.52, 0.56, 0.46);
	} else if (idx == 6u) { // Temperate Rainforest (top: 0.10, 0.48, 0.14; bottom: 0.04, 0.26, 0.08)
		a = vec3(0.070, 0.370, 0.110);
		b = vec3(0.030, 0.110, 0.030);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.53, 0.55, 0.47);
	} else if (idx == 7u) { // Subtropical Desert (top: 0.72, 0.60, 0.35; bottom: 0.40, 0.32, 0.16)
		a = vec3(0.560, 0.460, 0.255);
		b = vec3(0.160, 0.140, 0.095);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.46, 0.50, 0.42);
	} else if (idx == 8u) { // Savanna / Tropical Seasonal Forest (top: 0.58, 0.56, 0.22; bottom: 0.28, 0.28, 0.10)
		a = vec3(0.430, 0.420, 0.160);
		b = vec3(0.150, 0.140, 0.060);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.48, 0.53, 0.45);
	} else { // 9u: Tropical Rainforest (top: 0.08, 0.46, 0.12; bottom: 0.03, 0.24, 0.06)
		a = vec3(0.055, 0.350, 0.090);
		b = vec3(0.025, 0.110, 0.030);
		c = vec3(0.50, 0.50, 0.50);
		d = vec3(0.52, 0.56, 0.46);
	}
}

// Backwards-compatible overload returning top and bottom colors matching FoliageManager
void getWhittakerBiomePalette(uint biomeIdx, out vec3 colorTop, out vec3 colorBottom) {
	vec3 a, b, c, d;
	getWhittakerBiomePalette(biomeIdx, a, b, c, d);
	colorTop = a + b;
	colorBottom = a - b;
}

// Unified tree species selection across canopy proxies and mesh instances
// 0: Pine / Evergreen, 1: Deciduous, 2: Palm / Tropical Canopy, 3: Cactus / Arid Scrub
uint getWhittakerTreeType(uint biomeIdx, float temp, float moisture) {
	if (biomeIdx == 7u || moisture < 0.25) {
		return 3u; // Cactus / Arid Scrub
	} else if (biomeIdx == 8u || biomeIdx == 9u || (temp > 0.70 && moisture > 0.60)) {
		return 2u; // Palm / Tropical Canopy
	} else if (biomeIdx == 0u || biomeIdx == 1u || biomeIdx == 2u || temp < 0.38) {
		return 0u; // Pine / Evergreen
	} else {
		return 1u; // Deciduous
	}
}

// Checks whether (temp, moisture) remains within the hysteresis-expanded boundary of a biome index
bool inExpandedBiomeDomain(uint prevIdx, float temp, float moisture, float margin) {
	if (prevIdx == 0u) { // Ice / Glacier
		return (temp < 0.2 + margin) && (moisture >= 0.3 - margin);
	} else if (prevIdx == 1u) { // Tundra
		return (temp < 0.2 + margin) && (moisture < 0.3 + margin);
	} else if (prevIdx == 2u) { // Taiga / Boreal Forest
		return (temp >= 0.2 - margin) && (temp < 0.4 + margin) && (moisture >= 0.3 - margin);
	} else if (prevIdx == 3u) { // Cold Desert / Temperate Grassland
		bool condA = (temp >= 0.2 - margin) && (temp < 0.4 + margin) && (moisture < 0.3 + margin);
		bool condB = (temp >= 0.4 - margin) && (temp < 0.7 + margin) && (moisture < 0.25 + margin);
		return condA || condB;
	} else if (prevIdx == 4u) { // Temperate Woodland
		return (temp >= 0.4 - margin) && (temp < 0.7 + margin) && (moisture >= 0.25 - margin) && (moisture < 0.5 + margin);
	} else if (prevIdx == 5u) { // Temperate Deciduous Forest
		return (temp >= 0.4 - margin) && (temp < 0.7 + margin) && (moisture >= 0.5 - margin) && (moisture < 0.75 + margin);
	} else if (prevIdx == 6u) { // Temperate Rainforest
		return (temp >= 0.4 - margin) && (temp < 0.7 + margin) && (moisture >= 0.75 - margin);
	} else if (prevIdx == 7u) { // Subtropical Desert
		return (temp >= 0.7 - margin) && (moisture < 0.2 + margin);
	} else if (prevIdx == 8u) { // Savanna / Tropical Seasonal
		return (temp >= 0.7 - margin) && (moisture >= 0.2 - margin) && (moisture < 0.55 + margin);
	} else if (prevIdx == 9u) { // Tropical Rainforest
		return (temp >= 0.7 - margin) && (moisture >= 0.55 - margin);
	}
	return false;
}

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
			b.roughness = 0.85;
			b.biomeIndex = 1.0;
		} else {
			// Ice / Glacier / Snow
			b.roughness = 0.3;
			b.biomeIndex = 0.0;
		}
	} else if (temp < 0.4) {
		if (moisture < 0.3) {
			// Cold Desert / Taiga Scrub
			b.roughness = 0.80;
			b.biomeIndex = 3.0;
		} else if (moisture < 0.6) {
			// Taiga / Boreal Forest
			b.roughness = 0.75;
			b.biomeIndex = 2.0;
		} else {
			// Moist Taiga / Bogs
			b.roughness = 0.70;
			b.biomeIndex = 2.0;
		}
	} else if (temp < 0.7) {
		if (moisture < 0.25) {
			// Temperate Grassland / Steppe
			b.roughness = 0.80;
			b.biomeIndex = 3.0;
		} else if (moisture < 0.5) {
			// Temperate Woodland / Shrubland
			b.roughness = 0.75;
			b.biomeIndex = 4.0;
		} else if (moisture < 0.75) {
			// Temperate Deciduous Forest
			b.roughness = 0.70;
			b.biomeIndex = 5.0;
		} else {
			// Temperate Rainforest
			b.roughness = 0.65;
			b.biomeIndex = 6.0;
		}
	} else {
		if (moisture < 0.2) {
			// Subtropical Desert
			b.roughness = 0.90;
			b.biomeIndex = 7.0;
		} else if (moisture < 0.55) {
			// Tropical Seasonal Forest / Savanna
			b.roughness = 0.80;
			b.biomeIndex = 8.0;
		} else {
			// Tropical Rainforest
			b.roughness = 0.60;
			b.biomeIndex = 9.0;
		}
	}

	getWhittakerBiomePalette(uint(clamp(b.biomeIndex, 0.0, 9.0)), b.palA, b.palB, b.palC, b.palD);
	b.color = evalCosinePalette(b.palA, b.palB, b.palC, b.palD, 0.65);

	// Apply thunderstorm severity weathering (darker lushness or storm darkening)
	b.color = mix(b.color, b.color * 0.75 + vec3(0.02, 0.05, 0.08), severity * 0.4);

	// Blend snow cover for cold temperatures / high altitude precipitation
	if (b.snowCover > 0.01) {
		b.color = mix(b.color, vec3(0.92, 0.95, 0.98), b.snowCover);
		b.roughness = mix(b.roughness, 0.35, b.snowCover);
	}

	return b;
}

WhittakerBiome evaluateWhittakerBiomeWithHysteresis(
	float temp, float moisture, float severity, float precipType, float prevBiomeIndex
) {
	WhittakerBiome b;
	b.snowCover = clamp((0.35 - temp) * 3.0 + (precipType > 0.5 ? 0.3 : 0.0), 0.0, 1.0);

	const float HYSTERESIS_MARGIN = 0.035;
	if (prevBiomeIndex >= 0.0 && prevBiomeIndex <= 9.0) {
		uint prevIdx = uint(round(prevBiomeIndex));
		if (inExpandedBiomeDomain(prevIdx, temp, moisture, HYSTERESIS_MARGIN)) {
			b.biomeIndex = float(prevIdx);
			if (prevIdx == 0u) b.roughness = 0.3;
			else if (prevIdx == 1u) b.roughness = 0.85;
			else if (prevIdx == 2u) b.roughness = (moisture < 0.6) ? 0.75 : 0.70;
			else if (prevIdx == 3u) b.roughness = 0.80;
			else if (prevIdx == 4u) b.roughness = 0.75;
			else if (prevIdx == 5u) b.roughness = 0.70;
			else if (prevIdx == 6u) b.roughness = 0.65;
			else if (prevIdx == 7u) b.roughness = 0.90;
			else if (prevIdx == 8u) b.roughness = 0.80;
			else b.roughness = 0.60;

			getWhittakerBiomePalette(prevIdx, b.palA, b.palB, b.palC, b.palD);
			b.color = evalCosinePalette(b.palA, b.palB, b.palC, b.palD, 0.65);
			b.color = mix(b.color, b.color * 0.75 + vec3(0.02, 0.05, 0.08), severity * 0.4);
			if (b.snowCover > 0.01) {
				b.color = mix(b.color, vec3(0.92, 0.95, 0.98), b.snowCover);
				b.roughness = mix(b.roughness, 0.35, b.snowCover);
			}
			return b;
		}
	}

	return evaluateWhittakerBiome(temp, moisture, severity, precipType);
}

#endif // WHITTAKER_GLSL
