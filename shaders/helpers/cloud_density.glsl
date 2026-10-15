#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

#include "lygia/generative/psrdnoise.glsl"

float EvaluateAnalyticalCloudDensity(vec3 worldPos, float minFeatureSize, float phase) {
    float density = 0.0;
    float frequency = 1.0 / 32000.0; // 32km base shape
    float amplitude = 1.0;
    float maxAmplitude = 0.0;

    for (int i = 0; i < 3; i++) {
        float wavelength = 1.0 / frequency;

        // Nyquist clamp prevents sub-pixel/sub-voxel aliasing
        float fade = clamp((wavelength - minFeatureSize) / minFeatureSize, 0.0, 1.0);
        if (fade <= 0.0) break;

        float n = psrdnoise(worldPos * frequency, vec3(0.0), phase);

        density += n * amplitude * fade;
        maxAmplitude += amplitude * fade;

        frequency *= 1.87;
        amplitude *= 0.43;
    }

    if (maxAmplitude > 0.0) {
        density = max(0.0, density / maxAmplitude);
    }

    // Height shaping (assuming baseAlt = 5000.0, vSpan = 15000.0)
    float voxelNormY = (worldPos.y - 5000.0) / 15000.0;
    density *= smoothstep(0.1, 0.15, voxelNormY) * (1.0 - smoothstep(0.85, 1.0, voxelNormY));
    density *= smoothstep(0.15, 0.85, density);

    return density;
}
#endif