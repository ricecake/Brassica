#ifndef HELPERS_CLOUD_SHADOW_MAPPING_GLSL
#define HELPERS_CLOUD_SHADOW_MAPPING_GLSL

// Fixed 1024x1024 texel grid, matching CloudShadowMap's resource declaration.
#define CLOUD_SHADOW_MAP_DIM 1024.0

// Projects a world point down to where its sun ray (travelling in -sunDir) crosses the flat
// y=0 plane. This is the "address" a point maps to in the shadow map: unlike projecting to the
// real terrain height (the old approach), it needs no terrain sample, and -- critically -- the
// writer and reader both compute it identically, with no way for them to disagree about which
// texel a point belongs to. The old terrain-height lookup was also only valid within ~245m of
// the camera (hardcoded clipmap level 0 against a 320km shadow-map extent), so almost every
// lookup was silently aliasing to an unrelated nearby height that shifted as the camera moved --
// a second, independent source of the shadow-map ghosting, on top of the unsnapped window center
// below.
//
// max(0.0, p.y): points at or below the plane (underwater, low-lying terrain) use their own XZ
// directly instead of the full ray correction -- the old terrain-relative projection had the
// same clamp (max(0.0, p.y - groundY)) for the same reason. Without it, the horizontal offset is
// p.y/raySunY, which at grazing sun angles (raySunY floors at 0.05) is ~20x p.y: a point just
// 16m below the plane would shift over a full 312m texel, and small depth changes would swing
// the lookup across unrelated, wrapped-around texels -- read as cloud-shaped noise with no
// actual cloud behind it (jagged shadow edges, and spurious transmittance variation feeding
// false god rays in the froxel pass). Underwater light doesn't travel in a straight line along
// the in-air sunDir anyway (refraction bends it toward vertical, "Snell's window"), so there's
// no physically "more correct" unclamped answer to trade away stability for below the plane --
// using the straight-down column is both the stable choice and the more defensible one.
vec2 cloudShadowLabel(vec3 p, vec3 sunDir, float raySunY) {
	return p.xz - (max(0.0, p.y) / raySunY) * sunDir.xz;
}

// True toroidal (wrap-around) addressing, the same trick terrain.glsl's sampleToroidalUV uses
// for its clipmap: a given world label always maps to the same texel, forever, regardless of
// camera position. The camera-following behavior comes entirely from which texels the writer
// bothers refreshing each frame (cloudShadowWriteLabel below), never from this addressing
// formula -- so a non-refreshed texel's stored data goes stale in time, but is never
// reinterpreted as belonging to a different world location just because the camera moved, which
// is what caused the ghosting: the old center-relative UV mapping recomputed fresh from the raw
// camera position every frame, so texels that weren't rewritten that frame (3/4 of them, under
// the 2x2 stagger below) got reinterpreted under a shifted mapping on every single frame, not
// just the one in four they were actually refreshed.
vec2 cloudShadowUV(vec2 label, float shadowMapExtent) {
	float texelSize = shadowMapExtent / CLOUD_SHADOW_MAP_DIM;
	vec2  gridPos = label / texelSize;
	return mod(gridPos, CLOUD_SHADOW_MAP_DIM) / CLOUD_SHADOW_MAP_DIM;
}

// Inverse of cloudShadowUV for the writer: given a texel index and a camera-derived window
// center, returns the world-space label that texel should be written for. centerXZ needs no
// snapping -- mod() finds the correct nearby representative for any real-valued center, so the
// window follows the camera continuously with no discrete re-centering pop.
vec2 cloudShadowWriteLabel(ivec2 vox, vec2 centerXZ, float shadowMapExtent) {
	float texelSize = shadowMapExtent / CLOUD_SHADOW_MAP_DIM;
	vec2  gridCenter = centerXZ / texelSize;
	vec2  windowStart = gridCenter - CLOUD_SHADOW_MAP_DIM * 0.5;
	vec2  target = vec2(vox) + 0.5;
	vec2  cellForVox = windowStart + mod(target - windowStart, CLOUD_SHADOW_MAP_DIM);
	return cellForVox * texelSize;
}

#endif // HELPERS_CLOUD_SHADOW_MAPPING_GLSL
