#ifndef BRASSICA_SPHERIFIED_CUBE_GLSL
#define BRASSICA_SPHERIFIED_CUBE_GLSL

#include "helpers/constants.glsl"

vec3 projectToSpherifiedCube(vec3 dir, out uint outFace) {
    vec3 absDir = abs(dir);
    vec3 cubePoint;
    uint face = 0u;
    if (absDir.z >= absDir.x && absDir.z >= absDir.y) {
        if (dir.z >= 0.0) {
            face = 0u; // +Z
            cubePoint = vec3(dir.x / dir.z, dir.y / dir.z, 1.0);
        } else {
            face = 1u; // -Z
            cubePoint = vec3(-dir.x / -dir.z, dir.y / -dir.z, -1.0);
        }
    } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        if (dir.y >= 0.0) {
            face = 2u; // +Y
            cubePoint = vec3(dir.x / dir.y, 1.0, dir.z / dir.y);
        } else {
            face = 3u; // -Y
            cubePoint = vec3(dir.x / -dir.y, -1.0, -dir.z / -dir.y);
        }
    } else {
        if (dir.x >= 0.0) {
            face = 4u; // +X
            cubePoint = vec3(1.0, dir.y / dir.x, dir.z / dir.x);
        } else {
            face = 5u; // -X
            cubePoint = vec3(-1.0, dir.y / -dir.x, -dir.z / -dir.x);
        }
    }
    outFace = face;
    return normalize(cubePoint);
}

vec3 projectToSpherifiedCube(vec3 dir) {
    uint face;
    return projectToSpherifiedCube(dir, face);
}

vec3 worldXZToSphereDir(vec2 worldXZ) {
    float theta = worldXZ.x / FAKE_PLANET_RADIUS;
    float phi   = -worldXZ.y / FAKE_PLANET_RADIUS;
    float sin_t = sin(theta); float cos_t = cos(theta);
    float sin_p = sin(phi);   float cos_p = cos(phi);
    return vec3(cos_p * sin_t, sin_p, cos_p * cos_t);
}

vec3 spherifiedChunkPos(vec2 worldXZ, float height) {
    vec3 dir = worldXZToSphereDir(worldXZ);
    uint face;
    vec3 norm = projectToSpherifiedCube(dir, face);
    return norm * (FAKE_PLANET_RADIUS + height);
}

void cameraTopocentricBasis(vec3 camPos, out vec3 camPos3D, out vec3 camUp, out vec3 camEast, out vec3 camNorth) {
    vec3 dir = worldXZToSphereDir(camPos.xz);
    uint face;
    camUp = projectToSpherifiedCube(dir, face);
    camPos3D = camUp * (FAKE_PLANET_RADIUS + camPos.y);
    camEast = (length(camUp.xz) > 1e-5) ? normalize(vec3(camUp.z, 0.0, -camUp.x)) : vec3(1.0, 0.0, 0.0);
    camNorth = cross(camUp, camEast);
}

void transformToCameraTopocentric(vec3 world3D, vec3 normal3D, vec3 camPos3D, vec3 camUp, vec3 camEast, vec3 camNorth, out vec3 relPos, out vec3 relNormal) {
    vec3 deltaP = world3D - camPos3D;
    relPos = vec3(dot(deltaP, camEast), dot(deltaP, camUp), -dot(deltaP, camNorth));
    relNormal = vec3(dot(normal3D, camEast), dot(normal3D, camUp), -dot(normal3D, camNorth));
}

#endif // BRASSICA_SPHERIFIED_CUBE_GLSL
