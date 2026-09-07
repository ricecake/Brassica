#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

const float PI = 3.14159265359;
const float kEarthRadius = 6360.0; // km

layout(push_constant) uniform AtmospherePushConstants {
	vec3  rayleighScatteringBase;
	float rayleighScaleHeight;
	vec3  ozoneAbsorptionBase;
	float mieScaleHeight;
	vec3  hazeColor;
	float mieScatteringBase;
	float mieExtinctionBase;
	float rayleighScale;
	float mieScale;
	float mieAnisotropy;
	float atmosphereHeight;
	float hazeDensity;
	float hazeHeight;
} u_atmosphere;

#define kRayleighScattering u_atmosphere.rayleighScatteringBase
#define kRayleighScaleHeight u_atmosphere.rayleighScaleHeight
#define kOzoneAbsorption u_atmosphere.ozoneAbsorptionBase
#define kMieScaleHeight u_atmosphere.mieScaleHeight
#define hazeColor u_atmosphere.hazeColor
#define kMieScattering u_atmosphere.mieScatteringBase
#define kMieExtinction u_atmosphere.mieExtinctionBase
#define u_rayleighScale u_atmosphere.rayleighScale
#define u_mieScale u_atmosphere.mieScale
#define u_mieAnisotropy u_atmosphere.mieAnisotropy
#define kAtmosphereHeight u_atmosphere.atmosphereHeight
#define kTopRadius (kEarthRadius + kAtmosphereHeight)

bool intersectSphere(vec3 ro, vec3 rd, float radius, out float t0, out float t1) {
	float b = dot(ro, rd);
	float rLen = length(ro);
	float c = (rLen - radius) * (rLen + radius);
	float det = b * b - c;
	if (det < 0.0)
		return false;
	det = sqrt(det);
	t0 = -b - det;
	t1 = -b + det;
	return true;
}

float getRayleighDensity(float h) {
	return exp(-max(0.0, h) / kRayleighScaleHeight);
}

float getMieDensity(float h) {
	return exp(-max(0.0, h) / kMieScaleHeight);
}

float getOzoneDensity(float h) {
	return max(0.0, 1.0 - abs(max(0.0, h) - 25.0) / 15.0);
}

struct Sampling {
	vec3 rayleigh;
	vec3 mie;
	vec3 extinction;
};

Sampling getAtmosphereProperties(float h) {
	float rd = getRayleighDensity(h);
	float md = getMieDensity(h);
	float od = getOzoneDensity(h);

	Sampling s;
	s.rayleigh = kRayleighScattering * rd * u_rayleighScale;
	s.mie = hazeColor * (kMieScattering * md * u_mieScale);
	s.extinction = s.rayleigh + hazeColor * (kMieExtinction * md * u_mieScale) + kOzoneAbsorption * od;
	return s;
}

float rayleighPhase(float cosTheta) {
	return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float miePhase(float cosTheta) {
	float g = u_mieAnisotropy;
	float g2 = g * g;
	return (1.0 - g2) / (4.0 * PI * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

vec2 transmittanceToUV(float r, float mu) {
	float x_mu = mu * 0.5 + 0.5;
	float x_r = (r - kEarthRadius) / kAtmosphereHeight;
	return vec2(x_mu, x_r);
}

void UVToTransmittance(vec2 uv, out float r, out float mu) {
	mu = uv.x * 2.0 - 1.0;
	r = kEarthRadius + uv.y * kAtmosphereHeight;
}

#endif
