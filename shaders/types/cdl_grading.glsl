#ifndef TYPES_CDL_GRADING_GLSL
#define TYPES_CDL_GRADING_GLSL

// One depth-scoped ASC CDL grading layer for the scene (isSky == 0 in tonemap.frag). isMain
// entries always apply at full weight; every other entry blends in based on how close its
// targetDepth is to the pixel's linearized depth, falling off over falloffWidth at falloffRate.
struct CdlEntry {
	vec4  cdlSlope;
	vec4  cdlOffset;
	vec4  cdlPower;
	float cdlSaturation;
	float targetDepth;
	float falloffWidth;
	float falloffRate;
	int   priority;
	int   enabled;
	int   isMain;
	float padding;
};

layout(std430, set = 0, binding = 6) buffer CdlGradingLayers {
	CdlEntry cdlEntries[];
};

#endif // TYPES_CDL_GRADING_GLSL
