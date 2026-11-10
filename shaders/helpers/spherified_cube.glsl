#ifndef BRASSICA_SPHERIFIED_CUBE_GLSL
#define BRASSICA_SPHERIFIED_CUBE_GLSL

#include "helpers/constants.glsl"

// Spherified cube face projection
// 6 Cube Faces:
// Face 0: +X, Face 1: -X
// Face 2: +Y, Face 3: -Y
// Face 4: +Z, Face 5: -Z

vec3 cubeFaceToCubePoint(uint face, vec2 uv) {
    vec3 c;
    if (face == 0u) {        // +X
        c = vec3(1.0, uv.y, uv.x);
    } else if (face == 1u) { // -X
        c = vec3(-1.0, uv.y, -uv.x);
    } else if (face == 2u) { // +Y
        c = vec3(uv.x, 1.0, uv.y);
    } else if (face == 3u) { // -Y
        c = vec3(uv.x, -1.0, -uv.y);
    } else if (face == 4u) { // +Z
        c = vec3(uv.x, uv.y, 1.0);
    } else {                 // -Z
        c = vec3(-uv.x, uv.y, -1.0);
    }
    return c;
}

vec3 spherifyCubePoint(vec3 c) {
    float x2 = c.x * c.x;
    float y2 = c.y * c.y;
    float z2 = c.z * c.z;

    float sx = c.x * sqrt(max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + (y2 * z2) / 3.0));
    float sy = c.y * sqrt(max(0.0, 1.0 - z2 * 0.5 - x2 * 0.5 + (z2 * x2) / 3.0));
    float sz = c.z * sqrt(max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + (x2 * y2) / 3.0));

    return normalize(vec3(sx, sy, sz));
}

vec3 cubeFaceToSphereNormal(uint face, vec2 uv) {
    vec3 c = cubeFaceToCubePoint(face, uv);
    return spherifyCubePoint(c);
}

void sphereNormalToCubeFace(vec3 dir, out uint face, out vec2 uv) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x >= 0.0) {
            face = 0u; // +X
            uv = vec2(dir.z / dir.x, dir.y / dir.x);
        } else {
            face = 1u; // -X
            uv = vec2(-dir.z / -dir.x, dir.y / -dir.x);
        }
    } else if (a.y >= a.x && a.y >= a.z) {
        if (dir.y >= 0.0) {
            face = 2u; // +Y
            uv = vec2(dir.x / dir.y, dir.z / dir.y);
        } else {
            face = 3u; // -Y
            uv = vec2(dir.x / -dir.y, -dir.z / -dir.y);
        }
    } else {
        if (dir.z >= 0.0) {
            face = 4u; // +Z
            uv = vec2(dir.x / dir.z, dir.y / dir.z);
        } else {
            face = 5u; // -Z
            uv = vec2(-dir.x / -dir.z, dir.y / -dir.z);
        }
    }
    uv = clamp(uv, vec2(-1.0), vec2(1.0));
}

vec3 worldXZToSphereDir(vec2 worldXZ) {
    float theta = worldXZ.x / FAKE_PLANET_RADIUS;
    float phi   = -worldXZ.y / FAKE_PLANET_RADIUS;
    float sin_t = sin(theta); float cos_t = cos(theta);
    float sin_p = sin(phi);   float cos_p = cos(phi);
    return vec3(cos_p * sin_t, sin_p, cos_p * cos_t);
}

vec3 spherifiedChunkVertexPos(uint face, vec2 uv, float height) {
    vec3 n = cubeFaceToSphereNormal(face, uv);
    return n * (FAKE_PLANET_RADIUS + height);
}

// Convert 3D world position to latitude and longitude
void spherePositionToLatLong(vec3 sphereDir, out float latRad, out float lonRad) {
    latRad = asin(clamp(sphereDir.y, -1.0, 1.0));
    lonRad = atan(sphereDir.x, sphereDir.z);
}

// Topocentric tangent frame at a point on the sphere
void getSphereTopocentricFrame(vec3 sphereNormal, out vec3 outUp, out vec3 outEast, out vec3 outNorth) {
    outUp = normalize(sphereNormal);
    outEast = (length(outUp.xz) > 1e-5) ? normalize(vec3(outUp.z, 0.0, -outUp.x)) : vec3(1.0, 0.0, 0.0);
    outNorth = cross(outUp, outEast);
}

void transformToCameraTopocentric(
    vec3 world3D,
    vec3 normal3D,
    vec3 camPos3D,
    vec3 camUp,
    vec3 camEast,
    vec3 camNorth,
    out vec3 relPos,
    out vec3 relNormal
) {
    vec3 deltaP = world3D - camPos3D;
    relPos = vec3(dot(deltaP, camEast), dot(deltaP, camUp), -dot(deltaP, camNorth));
    relNormal = vec3(dot(normal3D, camEast), dot(normal3D, camUp), -dot(normal3D, camNorth));
}

#endif // BRASSICA_SPHERIFIED_CUBE_GLSL
