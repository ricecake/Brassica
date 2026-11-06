#ifndef BRASSICA_FOLIAGE_EMIT_GLSL
#define BRASSICA_FOLIAGE_EMIT_GLSL

// Tree/bush/cactus procedural generation, shared between foliage.mesh (the live, per-frame path --
// still used for grass/fern/flower/rock/seaweed, and for now still called for trees/bushes too
// until the archetype-palette render path replaces it) and foliage_bake.mesh (bakes a small fixed
// set of these into a reusable buffer instead of regenerating per instance per frame). Moved out
// of foliage.mesh unchanged -- same math, same vertex/primitive counts -- so there's exactly one
// copy instead of two drifting apart.
//
// Every function here writes through gl_MeshVerticesEXT/outWorldPos/outNormal/outUV/outColor/
// outMaterialType/outFade and reads params.lengthMultiplier/params.widthMultiplier directly, so
// whichever file includes this must declare those (as a real push-constant block for the live
// path, or as a small neutral-valued stand-in for the bake path -- see foliage_bake.mesh) and
// those output arrays *before* this include.

void emitSphereBlob(
	vec3 sphereCenter,
	vec3 radiusVec,
	vec4 colorBottom,
	vec4 colorTop,
	uint materialType,
	float fade,
	vec3 overallCenter,
	uint sphereSeed,
	inout uint vCount,
	inout uint pCount
) {
	uint vBase = vCount;
	uint pBase = pCount;

	const float kIcoA = 0.5257311121191336;
	const float kIcoB = 0.8506508083520399;

	const vec3 icoUnitDirs[12] = vec3[12](
		vec3(-kIcoA,  kIcoB, 0.0),
		vec3( kIcoA,  kIcoB, 0.0),
		vec3(-kIcoA, -kIcoB, 0.0),
		vec3( kIcoA, -kIcoB, 0.0),
		vec3(0.0, -kIcoA,  kIcoB),
		vec3(0.0,  kIcoA,  kIcoB),
		vec3(0.0, -kIcoA, -kIcoB),
		vec3(0.0,  kIcoA, -kIcoB),
		vec3( kIcoB, 0.0, -kIcoA),
		vec3( kIcoB, 0.0,  kIcoA),
		vec3(-kIcoB, 0.0, -kIcoA),
		vec3(-kIcoB, 0.0,  kIcoA)
	);

	const uvec3 icoTris[20] = uvec3[20](
		uvec3(0u, 11u, 5u),  uvec3(0u, 5u, 1u),   uvec3(0u, 1u, 7u),   uvec3(0u, 7u, 10u),  uvec3(0u, 10u, 11u),
		uvec3(1u, 5u, 9u),   uvec3(5u, 11u, 4u),  uvec3(11u, 10u, 2u), uvec3(10u, 7u, 6u),  uvec3(7u, 1u, 8u),
		uvec3(3u, 9u, 4u),   uvec3(3u, 4u, 2u),   uvec3(3u, 2u, 6u),   uvec3(3u, 6u, 8u),   uvec3(3u, 8u, 9u),
		uvec3(4u, 9u, 5u),   uvec3(2u, 4u, 11u),  uvec3(6u, 2u, 10u),  uvec3(8u, 6u, 7u),   uvec3(9u, 8u, 1u)
	);

	// Rotation per sphere blob using sphereSeed for natural organic variation
	float rotAngle = (pcg_hash(sphereSeed) * 0.8 - 0.4);
	float sRot = sin(rotAngle);
	float cRot = cos(rotAngle);

	for (uint i = 0u; i < 12u; ++i) {
		vec3 uDir = icoUnitDirs[i];
		vec3 rDir = vec3(uDir.x * cRot - uDir.z * sRot, uDir.y, uDir.x * sRot + uDir.z * cRot);

		vec3 vPos = sphereCenter + rDir * radiusVec;

		// Surface normal: blend local sphere radial normal with global cluster normal for smooth 3D shading
		vec3 sphereNorm = normalize(rDir);
		vec3 globalNorm = normalize(vPos - overallCenter + vec3(0.0, 0.2, 0.0));
		vec3 finalNorm = normalize(mix(sphereNorm, globalNorm, 0.35));

		// Vertical color blend based on normalized height
		float t = clamp(rDir.y * 0.5 + 0.5, 0.0, 1.0);
		vec4 vertColor = mix(colorBottom, colorTop, t);

		vec3 viewP = mat3(uViewMatrix) * vPos;
		gl_MeshVerticesEXT[vBase + i].gl_Position = uProjMatrix * vec4(viewP, 1.0);
		outWorldPos[vBase + i] = vPos + uCameraPosition.xyz;
		outNormal[vBase + i] = finalNorm;
		outUV[vBase + i] = vec2(0.5) + rDir.xz * 0.5;
		outColor[vBase + i] = vertColor;
		outMaterialType[vBase + i] = materialType;
		outFade[vBase + i] = fade;
	}

	// 20 triangular faces forming full 3D closed icosahedron sphere volume
	for (uint f = 0u; f < 20u; ++f) {
		gl_PrimitiveTriangleIndicesEXT[pBase + f] = uvec3(
			vBase + icoTris[f].x,
			vBase + icoTris[f].y,
			vBase + icoTris[f].z
		);
	}

	vCount += 12u;
	pCount += 20u;
}

void emitBushItem(
	vec3 basePos, float angle, float windStrength, uint itemSeed, vec3 terrainNormal, inout uint vCount,
	inout uint pCount
) {
	float bushHeight = (1.2 + pcg_hash(itemSeed + 2u) * 1.0) * params.lengthMultiplier;
	float bushWidth = (1.1 + pcg_hash(itemSeed + 3u) * 0.8) * params.widthMultiplier;

	vec4 bushBottomColor = vec4(0.06, 0.24, 0.05, 1.0);
	vec4 bushTopColor = vec4(0.18 + pcg_hash(itemSeed + 5u) * 0.08, 0.58, 0.12, 1.0);

	vec3 bushCenter = basePos + vec3(windStrength * 0.3, bushHeight * 0.5, windStrength * 0.3);
	float halfW = bushWidth * 0.5;

	// Tetrahedron of 4 overlapping icosahedron spheres forming a 3D volumetric bush
	// Apex (top) sphere
	float rApexH = bushHeight * 0.38 * (0.85 + pcg_hash(itemSeed + 10u) * 0.3);
	float rApexW = halfW * 0.70 * (0.85 + pcg_hash(itemSeed + 11u) * 0.3);
	vec3 apexCenter = bushCenter + vec3(0.0, bushHeight * 0.20, 0.0);
	emitSphereBlob(apexCenter, vec3(rApexW, rApexH, rApexW), bushBottomColor, bushTopColor, 5u, 1.0, bushCenter, itemSeed + 100u, vCount, pCount);

	// 3 base spheres forming the tetrahedral base triangle (120 degrees apart)
	float baseSpread = halfW * 0.55;
	for (uint i = 0u; i < 3u; ++i) {
		float bAngle = angle + float(i) * (TAU / 3.0) + (pcg_hash(itemSeed + 20u + i) * 0.3 - 0.15);
		vec3  bOffset = vec3(cos(bAngle) * baseSpread, -bushHeight * 0.12, sin(bAngle) * baseSpread);

		float rH = bushHeight * 0.32 * (0.85 + pcg_hash(itemSeed + 30u + i) * 0.3);
		float rW = halfW * 0.62 * (0.85 + pcg_hash(itemSeed + 40u + i) * 0.3);

		emitSphereBlob(bushCenter + bOffset, vec3(rW, rH, rW), bushBottomColor, bushTopColor, 5u, 1.0, bushCenter, itemSeed + 200u + i * 10u, vCount, pCount);
	}
}

void emitPalmTreeItem(
	vec3 basePos, float heightScale, float widthScale, float angle, vec3 sway, uint itemSeed, vec3 terrainNormal, float fade, inout uint vCount, inout uint pCount
) {
	float treeHeight = (6.0 + pcg_hash(itemSeed + 2u) * 2.5) * params.lengthMultiplier * fade;
	float trunkRadius = (0.22 + pcg_hash(itemSeed + 3u) * 0.08) * params.widthMultiplier * fade;

	vec4 trunkColor = vec4(0.42, 0.35, 0.25, 1.0);
	vec4 frondColor = vec4(0.18, 0.65, 0.20, 1.0);

	uint vBase = vCount;

	vec3 curveAxis = vec3(cos(angle), 0.0, sin(angle));
	vec3 sideAxis = vec3(-curveAxis.z, 0.0, curveAxis.x);

	for (uint s = 0u; s < 5u; ++s) {
		float t = float(s) * 0.25;
		float bend = t * t * 0.8;
		vec3 center = basePos + vec3(0.0, treeHeight * t, 0.0) + curveAxis * bend + sway * (t * t);
		float rad = trunkRadius * (1.0 - t * 0.35);

		vec3 pL = center - sideAxis * rad;
		vec3 pR = center + sideAxis * rad;

		uint li = vBase + s * 2u;
		vec3 viewL = mat3(uViewMatrix) * pL;
		gl_MeshVerticesEXT[li].gl_Position = uProjMatrix * vec4(viewL, 1.0);
		outWorldPos[li] = pL + uCameraPosition.xyz;
		outNormal[li] = normalize(pL - center + vec3(0.0, 0.1, 0.0));
		outUV[li] = vec2(0.0, t);
		outColor[li] = trunkColor;
		outMaterialType[li] = 6u;
		outFade[li] = 1.0;

		uint ri = li + 1u;
		vec3 viewR = mat3(uViewMatrix) * pR;
		gl_MeshVerticesEXT[ri].gl_Position = uProjMatrix * vec4(viewR, 1.0);
		outWorldPos[ri] = pR + uCameraPosition.xyz;
		outNormal[ri] = normalize(pR - center + vec3(0.0, 0.1, 0.0));
		outUV[ri] = vec2(1.0, t);
		outColor[ri] = trunkColor;
		outMaterialType[ri] = 6u;
		outFade[ri] = 1.0;
	}

	uint pBase = pCount;
	for (uint s = 0u; s < 4u; ++s) {
		uint i0 = vBase + s * 2u;
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u] = uvec3(i0, i0 + 1u, i0 + 2u);
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u + 1u] = uvec3(i0 + 1u, i0 + 3u, i0 + 2u);
	}

	vCount += 10u;
	pCount += 8u;

	vec3 tipCenter = basePos + vec3(0.0, treeHeight, 0.0) + curveAxis * 0.8 + sway;
	float frondLen = (2.2 + pcg_hash(itemSeed + 5u) * 1.0) * params.widthMultiplier * fade;

	uint fBase = vCount;
	uint fPBase = pCount;

	for (uint f = 0u; f < 5u; ++f) {
		float fAngle = angle + float(f) * (TAU / 5.0) + pcg_hash(itemSeed + 10u + f) * 0.2;
		vec3 fDir = vec3(cos(fAngle), 0.0, sin(fAngle));
		vec3 fRight = vec3(-fDir.z, 0.0, fDir.x);

		uint fVertBase = fBase + f * 6u;
		for (uint s = 0u; s < 3u; ++s) {
			float ft = float(s) * 0.5;
			float dropY = -ft * ft * 0.9 * frondLen;
			vec3 fCenter = tipCenter + fDir * (frondLen * ft) + vec3(0.0, dropY, 0.0);
			float fWidth = (0.4 + pcg_hash(itemSeed + 20u + f) * 0.2) * (1.0 - ft * 0.5);

			vec3 pL = fCenter - fRight * (fWidth * 0.5);
			vec3 pR = fCenter + fRight * (fWidth * 0.5);

			uint li = fVertBase + s * 2u;
			vec3 viewL = mat3(uViewMatrix) * pL;
			gl_MeshVerticesEXT[li].gl_Position = uProjMatrix * vec4(viewL, 1.0);
			outWorldPos[li] = pL + uCameraPosition.xyz;
			outNormal[li] = vec3(0.0, 1.0, 0.0);
			outUV[li] = vec2(0.0, ft);
			outColor[li] = frondColor;
			outMaterialType[li] = 7u;
			outFade[li] = 1.0;

			uint ri = li + 1u;
			vec3 viewR = mat3(uViewMatrix) * pR;
			gl_MeshVerticesEXT[ri].gl_Position = uProjMatrix * vec4(viewR, 1.0);
			outWorldPos[ri] = pR + uCameraPosition.xyz;
			outNormal[ri] = vec3(0.0, 1.0, 0.0);
			outUV[ri] = vec2(1.0, ft);
			outColor[ri] = frondColor;
			outMaterialType[ri] = 7u;
			outFade[ri] = 1.0;
		}

		uint fp = fPBase + f * 4u;
		for (uint s = 0u; s < 2u; ++s) {
			uint i0 = fVertBase + s * 2u;
			gl_PrimitiveTriangleIndicesEXT[fp + s * 2u] = uvec3(i0, i0 + 1u, i0 + 2u);
			gl_PrimitiveTriangleIndicesEXT[fp + s * 2u + 1u] = uvec3(i0 + 1u, i0 + 3u, i0 + 2u);
		}
	}

	vCount += 30u;
	pCount += 20u;
}

void emitPineTreeItem(
	vec3 basePos, float heightScale, float widthScale, float angle, vec3 sway, uint itemSeed, vec3 terrainNormal, float fade, inout uint vCount, inout uint pCount
) {
	float treeHeight = (5.5 + pcg_hash(itemSeed + 2u) * 3.0) * params.lengthMultiplier * fade;
	float trunkRadius = (0.20 + pcg_hash(itemSeed + 3u) * 0.08) * params.widthMultiplier * fade;

	vec4 barkColor = vec4(0.28, 0.20, 0.14, 1.0);
	vec4 needleColor = vec4(0.08, 0.32, 0.12, 1.0);

	uint vBase = vCount;

	vec3 sideAxis = vec3(cos(angle), 0.0, sin(angle));
	for (uint s = 0u; s < 4u; ++s) {
		float t = float(s) * 0.25;
		vec3 center = basePos + vec3(0.0, treeHeight * t * 0.7, 0.0) + sway * (t * t * 0.3);
		float rad = trunkRadius * (1.0 - t * 0.4);

		vec3 pL = center - sideAxis * rad;
		vec3 pR = center + sideAxis * rad;

		uint li = vBase + s * 2u;
		vec3 viewL = mat3(uViewMatrix) * pL;
		gl_MeshVerticesEXT[li].gl_Position = uProjMatrix * vec4(viewL, 1.0);
		outWorldPos[li] = pL + uCameraPosition.xyz;
		outNormal[li] = normalize(pL - center + vec3(0.0, 0.1, 0.0));
		outUV[li] = vec2(0.0, t);
		outColor[li] = barkColor;
		outMaterialType[li] = 6u;
		outFade[li] = 1.0;

		uint ri = li + 1u;
		vec3 viewR = mat3(uViewMatrix) * pR;
		gl_MeshVerticesEXT[ri].gl_Position = uProjMatrix * vec4(viewR, 1.0);
		outWorldPos[ri] = pR + uCameraPosition.xyz;
		outNormal[ri] = normalize(pR - center + vec3(0.0, 0.1, 0.0));
		outUV[ri] = vec2(1.0, t);
		outColor[ri] = barkColor;
		outMaterialType[ri] = 6u;
		outFade[ri] = 1.0;
	}

	uint pBase = pCount;
	for (uint s = 0u; s < 3u; ++s) {
		uint i0 = vBase + s * 2u;
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u] = uvec3(i0, i0 + 1u, i0 + 2u);
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u + 1u] = uvec3(i0 + 1u, i0 + 3u, i0 + 2u);
	}

	vCount += 8u;
	pCount += 6u;

	uint tBase = vCount;
	uint tPBase = pCount;

	for (uint tier = 0u; tier < 3u; ++tier) {
		float tFrac = 0.25 + float(tier) * 0.25;
		float nextFrac = tFrac + 0.35;

		float baseRad = (1.8 - float(tier) * 0.45) * params.widthMultiplier * fade;
		vec3 baseCenter = basePos + vec3(0.0, treeHeight * tFrac, 0.0) + sway * (tFrac * tFrac);
		vec3 peakCenter = basePos + vec3(0.0, treeHeight * nextFrac, 0.0) + sway * (nextFrac * nextFrac);

		uint tierVBase = tBase + tier * 6u;
		vec3 viewPeak = mat3(uViewMatrix) * peakCenter;
		gl_MeshVerticesEXT[tierVBase].gl_Position = uProjMatrix * vec4(viewPeak, 1.0);
		outWorldPos[tierVBase] = peakCenter + uCameraPosition.xyz;
		outNormal[tierVBase] = vec3(0.0, 1.0, 0.0);
		outUV[tierVBase] = vec2(0.5, 1.0);
		outColor[tierVBase] = needleColor * 1.1;
		outMaterialType[tierVBase] = 7u;
		outFade[tierVBase] = 1.0;

		for (uint i = 0u; i < 5u; ++i) {
			float pAngle = float(i) * (TAU / 5.0) + angle;
			vec3 bOffset = vec3(cos(pAngle), 0.0, sin(pAngle)) * baseRad;
			vec3 bPos = baseCenter + bOffset;

			uint bi = tierVBase + 1u + i;
			vec3 viewB = mat3(uViewMatrix) * bPos;
			gl_MeshVerticesEXT[bi].gl_Position = uProjMatrix * vec4(viewB, 1.0);
			outWorldPos[bi] = bPos + uCameraPosition.xyz;
			outNormal[bi] = normalize(vec3(cos(pAngle), 0.3, sin(pAngle)));
			outUV[bi] = vec2(float(i) / 5.0, 0.0);
			outColor[bi] = needleColor;
			outMaterialType[bi] = 7u;
			outFade[bi] = 1.0;
		}

		uint tp = tPBase + tier * 5u;
		for (uint i = 0u; i < 5u; ++i) {
			uint nextI = (i + 1u) % 5u;
			gl_PrimitiveTriangleIndicesEXT[tp + i] = uvec3(tierVBase, tierVBase + 1u + i, tierVBase + 1u + nextI);
		}
	}

	vCount += 18u;
	pCount += 15u;
}

void emitDeciduousTreeItem(
	vec3 basePos, float heightScale, float widthScale, float angle, vec3 sway, uint itemSeed, vec3 terrainNormal, float fade, inout uint vCount, inout uint pCount
) {
	float treeHeight = (5.0 + pcg_hash(itemSeed + 2u) * 2.5) * params.lengthMultiplier * fade;
	float crownWidth = (3.2 + pcg_hash(itemSeed + 3u) * 1.2) * params.widthMultiplier * fade;

	vec4 barkColor = vec4(0.36, 0.28, 0.18, 1.0);
	vec4 leafBottomColor = vec4(0.12, 0.42, 0.08, 1.0);
	vec4 leafTopColor = vec4(0.24, 0.62, 0.16, 1.0);

	uint vBase = vCount;
	float trunkRadius = crownWidth * 0.12;

	// 3D Triangular Cylinder Trunk (4 height levels = 12 vertices, 3 segments * 6 = 18 triangles)
	for (uint s = 0u; s < 4u; ++s) {
		float t = float(s) * 0.25;
		vec3 center = basePos + vec3(0.0, treeHeight * t * 0.55, 0.0) + sway * (t * t * 0.3);
		float rad = trunkRadius * (1.2 - t * 0.4);

		for (uint i = 0u; i < 3u; ++i) {
			float ringAngle = angle + float(i) * (TAU / 3.0);
			vec3 ringDir = vec3(cos(ringAngle), 0.0, sin(ringAngle));
			vec3 pPos = center + ringDir * rad;

			uint vi = vBase + s * 3u + i;
			vec3 viewP = mat3(uViewMatrix) * pPos;
			gl_MeshVerticesEXT[vi].gl_Position = uProjMatrix * vec4(viewP, 1.0);
			outWorldPos[vi] = pPos + uCameraPosition.xyz;
			outNormal[vi] = normalize(ringDir + vec3(0.0, 0.1, 0.0));
			outUV[vi] = vec2(float(i) / 3.0, t);
			outColor[vi] = barkColor;
			outMaterialType[vi] = 6u;
			outFade[vi] = 1.0;
		}
	}

	uint pBase = pCount;
	for (uint s = 0u; s < 3u; ++s) {
		uint layer0 = vBase + s * 3u;
		uint layer1 = vBase + (s + 1u) * 3u;

		for (uint i = 0u; i < 3u; ++i) {
			uint iNext = (i + 1u) % 3u;

			uint i0 = layer0 + i;
			uint i1 = layer0 + iNext;
			uint i2 = layer1 + i;
			uint i3 = layer1 + iNext;

			gl_PrimitiveTriangleIndicesEXT[pBase + s * 6u + i * 2u] = uvec3(i0, i1, i2);
			gl_PrimitiveTriangleIndicesEXT[pBase + s * 6u + i * 2u + 1u] = uvec3(i1, i3, i2);
		}
	}

	vCount += 12u;
	pCount += 18u;

	// Canopy: Tetrahedral cluster of 4 overlapping 3D icosahedron spheres
	vec3 crownCenter = basePos + vec3(0.0, treeHeight * 0.70, 0.0) + sway * 0.5;
	float halfW = crownWidth * 0.5;
	float crownH = treeHeight * 0.50;

	// Apex (top) sphere
	float rApexH = crownH * 0.42 * (0.85 + pcg_hash(itemSeed + 10u) * 0.3);
	float rApexW = halfW * 0.65 * (0.85 + pcg_hash(itemSeed + 11u) * 0.3);
	vec3 apexCenter = crownCenter + vec3(0.0, crownH * 0.30, 0.0);
	emitSphereBlob(apexCenter, vec3(rApexW, rApexH, rApexW), leafBottomColor, leafTopColor, 7u, 1.0, crownCenter, itemSeed + 100u, vCount, pCount);

	// 3 base spheres forming the tetrahedral base triangle (120 degrees apart)
	float baseSpread = halfW * 0.52;
	for (uint i = 0u; i < 3u; ++i) {
		float bAngle = angle + float(i) * (TAU / 3.0) + (pcg_hash(itemSeed + 20u + i) * 0.3 - 0.15);
		vec3  bOffset = vec3(cos(bAngle) * baseSpread, -crownH * 0.15, sin(bAngle) * baseSpread);

		float rH = crownH * 0.38 * (0.85 + pcg_hash(itemSeed + 30u + i) * 0.3);
		float rW = halfW * 0.60 * (0.85 + pcg_hash(itemSeed + 40u + i) * 0.3);

		emitSphereBlob(crownCenter + bOffset, vec3(rW, rH, rW), leafBottomColor, leafTopColor, 7u, 1.0, crownCenter, itemSeed + 200u + i * 10u, vCount, pCount);
	}
}

void emitCactusItem(
	vec3 basePos, float heightScale, float widthScale, float angle, uint itemSeed, vec3 terrainNormal, float fade, inout uint vCount, inout uint pCount
) {
	float cactusHeight = (2.5 + pcg_hash(itemSeed + 2u) * 1.5) * params.lengthMultiplier * fade;
	float columnRadius = (0.20 + pcg_hash(itemSeed + 3u) * 0.08) * params.widthMultiplier * fade;

	vec4 cactusColor = vec4(0.22, 0.48, 0.26, 1.0);

	uint vBase = vCount;

	vec3 sideAxis = vec3(cos(angle), 0.0, sin(angle));

	for (uint s = 0u; s < 4u; ++s) {
		float t = float(s) * 0.33;
		vec3 center = basePos + vec3(0.0, cactusHeight * t, 0.0);
		float rad = columnRadius * (s == 3u ? 0.7 : 1.0);

		vec3 pL = center - sideAxis * rad;
		vec3 pR = center + sideAxis * rad;

		uint li = vBase + s * 2u;
		vec3 viewL = mat3(uViewMatrix) * pL;
		gl_MeshVerticesEXT[li].gl_Position = uProjMatrix * vec4(viewL, 1.0);
		outWorldPos[li] = pL + uCameraPosition.xyz;
		outNormal[li] = normalize(pL - center + vec3(0.0, 0.1, 0.0));
		outUV[li] = vec2(0.0, t);
		outColor[li] = cactusColor;
		outMaterialType[li] = 8u;
		outFade[li] = 1.0;

		uint ri = li + 1u;
		vec3 viewR = mat3(uViewMatrix) * pR;
		gl_MeshVerticesEXT[ri].gl_Position = uProjMatrix * vec4(viewR, 1.0);
		outWorldPos[ri] = pR + uCameraPosition.xyz;
		outNormal[ri] = normalize(pR - center + vec3(0.0, 0.1, 0.0));
		outUV[ri] = vec2(1.0, t);
		outColor[ri] = cactusColor;
		outMaterialType[ri] = 8u;
		outFade[ri] = 1.0;
	}

	uint pBase = pCount;
	for (uint s = 0u; s < 3u; ++s) {
		uint i0 = vBase + s * 2u;
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u] = uvec3(i0, i0 + 1u, i0 + 2u);
		gl_PrimitiveTriangleIndicesEXT[pBase + s * 2u + 1u] = uvec3(i0 + 1u, i0 + 3u, i0 + 2u);
	}

	vCount += 8u;
	pCount += 6u;

	uint aBase = vCount;
	uint aPBase = pCount;

	for (uint arm = 0u; arm < 2u; ++arm) {
		float armDir = (arm == 0u) ? 1.0 : -1.0;
		vec3 armOut = sideAxis * armDir;

		float armBaseY = cactusHeight * (0.35 + float(arm) * 0.15);
		float armLen = cactusHeight * 0.30;
		float armRad = columnRadius * 0.75;

		vec3 p0 = basePos + vec3(0.0, armBaseY, 0.0);
		vec3 p1 = p0 + armOut * (armLen * 0.8);
		vec3 p2 = p1 + vec3(0.0, armLen * 0.7, 0.0);

		vec3 armPts[3] = vec3[3](p0, p1, p2);

		uint armVBase = aBase + arm * 6u;
		for (uint s = 0u; s < 3u; ++s) {
			vec3 center = armPts[s];
			vec3 pL = center - vec3(0.0, armRad, 0.0);
			vec3 pR = center + vec3(0.0, armRad, 0.0);

			uint li = armVBase + s * 2u;
			vec3 viewL = mat3(uViewMatrix) * pL;
			gl_MeshVerticesEXT[li].gl_Position = uProjMatrix * vec4(viewL, 1.0);
			outWorldPos[li] = pL + uCameraPosition.xyz;
			outNormal[li] = armOut;
			outUV[li] = vec2(0.0, float(s) * 0.5);
			outColor[li] = cactusColor;
			outMaterialType[li] = 8u;
			outFade[li] = 1.0;

			uint ri = armVBase + s * 2u + 1u;
			vec3 viewR = mat3(uViewMatrix) * pR;
			gl_MeshVerticesEXT[ri].gl_Position = uProjMatrix * vec4(viewR, 1.0);
			outWorldPos[ri] = pR + uCameraPosition.xyz;
			outNormal[ri] = armOut;
			outUV[ri] = vec2(1.0, float(s) * 0.5);
			outColor[ri] = cactusColor;
			outMaterialType[ri] = 8u;
			outFade[ri] = 1.0;
		}

		uint ap = aPBase + arm * 4u;
		for (uint s = 0u; s < 2u; ++s) {
			uint i0 = armVBase + s * 2u;
			gl_PrimitiveTriangleIndicesEXT[ap + s * 2u] = uvec3(i0, i0 + 1u, i0 + 2u);
			gl_PrimitiveTriangleIndicesEXT[ap + s * 2u + 1u] = uvec3(i0 + 1u, i0 + 3u, i0 + 2u);
		}
	}

	vCount += 12u;
	pCount += 8u;
}

#endif // BRASSICA_FOLIAGE_EMIT_GLSL
