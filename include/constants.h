#pragma once

#include <cstdint>
#include <utility>

#include <glm/glm.hpp>

namespace brassica {
	namespace constants {
		namespace Engine {
			constexpr std::uint32_t FrameOverlap = 2;
			constexpr float         FakePlanetRadius = 600000.0f; // 600km (1/10th scale planet)
			constexpr float         FakePlanetDiameter = FakePlanetRadius * 2.0f;
		} // namespace Engine

		namespace General {
			namespace Math {
				constexpr float Pi = 3.14159265358979323846f;
				constexpr float TwoPi = Pi * 2.0f;
				constexpr float HalfPi = Pi * 0.5f;
				constexpr float DegToRad = Pi / 180.0f;
				constexpr float RadToDeg = 180.0f / Pi;
			} // namespace Math

			namespace Colors {
				constexpr float DefaultAmbientR = 90.0f / 255.0f;
				constexpr float DefaultAmbientG = 81.0f / 255.0f;
				constexpr float DefaultAmbientB = 62.0f / 255.0f;

				consteval glm::vec3 DefaultAmbient() {
					return glm::vec3(DefaultAmbientR, DefaultAmbientG, DefaultAmbientB);
				}
			} // namespace Colors
		} // namespace General

		namespace Project {
			namespace Window {
				constexpr int DefaultWidth = 1280;
				constexpr int DefaultHeight = 720;
				constexpr float DefaultAspectRatio = static_cast<float>(DefaultWidth) / static_cast<float>(DefaultHeight);

				consteval int GetDefaultWidth() { return DefaultWidth; }
				consteval int GetDefaultHeight() { return DefaultHeight; }
			} // namespace Window

			namespace Camera {
				constexpr float DefaultFOV = 45.0f;
				constexpr float DefaultNearPlane = 0.1f;
				constexpr float MinHeight = 0.1f;
				constexpr float MinSpeed = 0.5f;
				constexpr float DefaultSpeed = 15.0f;
				constexpr float SpeedStep = 2.5f;
				constexpr float RollSpeed = 45.0f;

				constexpr float FirstPersonEyeHeight = 4.8f;
				constexpr float FirstPersonCrouchHeight = 1.5f;
				constexpr float FirstPersonSprintMultiplier = 2.0f;
				constexpr float FirstPersonJumpForce = 12.5f;
				constexpr float FirstPersonGravity = 18.0f;
				constexpr float FirstPersonGroundSmoothing = 5.0f;

				// Derived First-Person Values
				constexpr float FirstPersonSprintSpeed = DefaultSpeed * FirstPersonSprintMultiplier;

				// Path following & chase camera constants
				constexpr float DefaultPathSpeed = 20.0f;
				constexpr float PathBankFactor = 1.8f;
				constexpr float PathBankSpeed = 3.5f;
				constexpr float ChaseTrailBehind = 15.0f;
				constexpr float ChaseElevation = 5.0f;
				constexpr float ChaseLookAhead = 10.0f;
				constexpr float ChaseResponsiveness = 1.5f;
				constexpr float PathFollowSmoothing = 5.0f;
			} // namespace Camera
		} // namespace Project

		namespace Library {
			namespace Input {
				constexpr int MaxKeys = 1024;
				constexpr int MaxMouseButtons = 8;
			} // namespace Input

			namespace ShaderWatcher {
				constexpr int DefaultPollIntervalMs = 250;
			} // namespace ShaderWatcher
		} // namespace Library

		namespace Class {
			namespace Lighting {
				constexpr std::uint32_t MaxLights = 1024;
				constexpr std::uint32_t TotalClusters = 16 * 9 * 24 + 1; // 3457 clusters

				// Sun Defaults
				constexpr float DefaultSunAzimuth = 0.0f;
				constexpr float DefaultSunElevation = 45.0f;
				constexpr float DefaultSunIntensity = 1.0f;
				constexpr float DefaultSunColorR = 2.5f;
				constexpr float DefaultSunColorG = 2.3f;
				constexpr float DefaultSunColorB = 2.0f;

				// Moon Defaults
				constexpr float DefaultMoonAzimuth = 180.0f;
				constexpr float DefaultMoonElevation = -45.0f;
				constexpr float DefaultMoonIntensity = 1.0f;
				constexpr float DefaultMoonColorR = 0.95f;
				constexpr float DefaultMoonColorG = 0.93f;
				constexpr float DefaultMoonColorB = 0.88f;

				// Day/Night Cycle Defaults
				constexpr float DefaultCycleTime = 8.0f;
				constexpr float DefaultCycleSpeed = 0.0125f;
				constexpr float DefaultMoonOffset = 6.0f;
				constexpr float DefaultMoonAzimuthBase = 70.0f;
				constexpr float DefaultLunarAlbedo = 0.08f;
				constexpr float DefaultLunarMonth = 2.0f;

				// Ambient & Exposures
				constexpr float DefaultAmbientR = 0.15f;
				constexpr float DefaultAmbientG = 0.15f;
				constexpr float DefaultAmbientB = 0.2f;
				constexpr float DefaultSkyExposure = 1.0f;
				constexpr float DefaultStarExposure = 1.0f;
				constexpr float DefaultTerrainExposure = 1.0f;

				// Lightning Defaults
				constexpr float DefaultLightningMaxLifetime = 0.3f;
				constexpr float DefaultLightningBranchProbability = 0.15f;
				constexpr float DefaultLightningThickness = 1.5f;
				constexpr float DefaultLightningColorR = 0.8f;
				constexpr float DefaultLightningColorG = 0.9f;
				constexpr float DefaultLightningColorB = 1.0f;

				consteval glm::vec3 DefaultSunColor() {
					return glm::vec3(DefaultSunColorR, DefaultSunColorG, DefaultSunColorB);
				}

				consteval glm::vec3 DefaultMoonColor() {
					return glm::vec3(DefaultMoonColorR, DefaultMoonColorG, DefaultMoonColorB);
				}

				consteval glm::vec3 DefaultAmbientLight() {
					return glm::vec3(DefaultAmbientR, DefaultAmbientG, DefaultAmbientB);
				}

				consteval glm::vec3 DefaultLightningColor() {
					return glm::vec3(DefaultLightningColorR, DefaultLightningColorG, DefaultLightningColorB);
				}
			} // namespace Lighting

			namespace Shadows {
				constexpr int   MaxLights = 10;
				constexpr int   MaxCascades = 4;
				constexpr int   MaxShadowMaps = 16;
				constexpr int   MapSize = 2048;
				constexpr float DefaultSceneRadius = 500.0f;
				constexpr float DefaultFOV = 45.0f;

				// Cascade split distances (logarithmic distribution)
				constexpr float CascadeSplit0 = 20.0f;
				constexpr float CascadeSplit1 = 50.0f;
				constexpr float CascadeSplit2 = 150.0f;
				constexpr float CascadeSplit3 = 700.0f;

				// Grid snapping sizes per cascade
				constexpr float GridSnapCascade0 = 0.25f;
				constexpr float GridSnapCascade1 = 1.0f;
				constexpr float GridSnapCascade2 = 4.0f;
				constexpr float GridSnapCascade3 = 8.0f;
			} // namespace Shadows

			namespace Terrain {
				constexpr int   BaseMeshletDimension = 11;
				constexpr int   VerticesPerMeshlet = BaseMeshletDimension * BaseMeshletDimension; // 121
				constexpr int   SkirtVertices = (BaseMeshletDimension - 1) * 4;                    // 40
				constexpr int   TotalMeshletVertices = VerticesPerMeshlet + SkirtVertices;          // 161
				constexpr int   DefaultMaxLODs = 23;
				constexpr int   DefaultWindowLODs = 12;
				constexpr int   MapDim = 1088;                                                     // 1024 + 64 grid cell padding
				constexpr float BaseTexelSize = 0.5f;
				constexpr int   MeshletsPerRow = 16;                                               // 16x16 grid of meshlets per LOD
				constexpr int   MeshletGridSizePerLOD = 4;                                          // 4x4
				constexpr int   MeshletCountPerLOD = MeshletsPerRow * MeshletsPerRow;               // 256
				constexpr int   TotalMeshlets = DefaultWindowLODs * MeshletCountPerLOD;            // 3072 (active window)
				constexpr float PathCorridorWidth = 0.15f;
			} // namespace Terrain

			namespace AsyncTerrain {
				constexpr std::size_t MaxUploadSlots = 32;
			} // namespace AsyncTerrain

			namespace ClusteredLighting {
				constexpr std::uint32_t GridX = 16;
				constexpr std::uint32_t GridY = 9;
				constexpr std::uint32_t GridZ = 24;
				constexpr std::uint32_t TotalClusters = GridX * GridY * GridZ; // 3456
				constexpr std::uint32_t MaxLightsPerCluster = 100;
			} // namespace ClusteredLighting

			namespace ShadingRate {
				constexpr std::uint32_t TileWidth = 16;
				constexpr std::uint32_t TileHeight = 16;
				constexpr std::uint32_t TileArea = TileWidth * TileHeight; // 256
			} // namespace ShadingRate

			namespace Particles {
				constexpr int   MaxParticles = 128000;
				constexpr int   AmbientParticleScale = 8192;
				constexpr int   MaxEmitters = 100;
				constexpr int   ComputeGroupSize = 256;
				constexpr int   ParticleGridSize = 131072;
				constexpr float ParticleGridCellSize = 2.0f;
				constexpr float DefaultAmbientDensity = 1.0f;
			} // namespace Particles

			namespace Explosions {
				constexpr int   MaxFragments = 50000;
				constexpr int   ComputeGroupSize = 64;
				constexpr float DefaultVelocity = 10.0f;
				constexpr float DefaultRandomVelocity = 5.0f;
			} // namespace Explosions

			namespace Shockwaves {
				constexpr int   MaxShockwaves = 16;
				constexpr float DefaultIntensity = 0.5f;
				constexpr float DefaultRingWidth = 3.0f;
				constexpr float DefaultDuration = 1.2f;

				consteval glm::vec3 DefaultColor() {
					return glm::vec3(1.0f, 0.6f, 0.2f);
				}
			} // namespace Shockwaves

			namespace SdfVolumes {
				constexpr int   MaxSources = 128;
				constexpr float DefaultRadius = 5.0f;
				constexpr float DefaultSmoothness = 2.0f;
			} // namespace SdfVolumes

			namespace Trails {
				constexpr int   DefaultMaxLength = 250;
				constexpr int   DefaultTrailLength = 10;
				constexpr int   Segments = 8;
				constexpr int   CurveSegments = 4;
				constexpr float BaseThickness = 0.06f;
				constexpr float DefaultRoughness = 0.3f;
				constexpr float DefaultMetallic = 0.0f;
				constexpr int   FloatsPerVertex = 9;
				constexpr int   InitialVertexCapacity = 500000;
				constexpr float GrowthFactor = 1.5f;
			} // namespace Trails

			namespace Rendering {
				constexpr int BlurPasses = 4;
			} // namespace Rendering

			namespace Akira {
				constexpr float DefaultGrowthDuration = 0.5f;
				constexpr float DefaultFadeDuration = 3.0f;
				constexpr float DefaultRadius = 20.0f;
			} // namespace Akira

			namespace Checkpoint {
				constexpr float DefaultRadius = 10.0f;
				constexpr float DefaultHaloWidth = 2.0f;
				constexpr float DefaultAuraWidth = 5.0f;
				constexpr float DefaultLifespan = 60.0f;

				namespace Colors {
					constexpr float GoldR = 1.0f;
					constexpr float GoldG = 0.84f;
					constexpr float GoldB = 0.0f;

					constexpr float SilverR = 0.75f;
					constexpr float SilverG = 0.75f;
					constexpr float SilverB = 0.75f;

					constexpr float BlackR = 0.01f;
					constexpr float BlackG = 0.01f;
					constexpr float BlackB = 0.01f;

					constexpr float BlueR = 0.0f;
					constexpr float BlueG = 0.5f;
					constexpr float BlueB = 1.0f;

					constexpr float NeonGreenR = 0.2f;
					constexpr float NeonGreenG = 1.0f;
					constexpr float NeonGreenB = 0.2f;

					consteval glm::vec3 Gold() {
						return glm::vec3(GoldR, GoldG, GoldB);
					}

					consteval glm::vec3 Silver() {
						return glm::vec3(SilverR, SilverG, SilverB);
					}

					consteval glm::vec3 Black() {
						return glm::vec3(BlackR, BlackG, BlackB);
					}

					consteval glm::vec3 Blue() {
						return glm::vec3(BlueR, BlueG, BlueB);
					}

					consteval glm::vec3 NeonGreen() {
						return glm::vec3(NeonGreenR, NeonGreenG, NeonGreenB);
					}
				} // namespace Colors
			} // namespace Checkpoint
		} // namespace Class
	} // namespace constants

	// Backward compatibility alias for Brassica::Constants
	namespace Constants = constants;
} // namespace brassica

namespace Brassica {
	namespace Constants = brassica::constants;
} // namespace Brassica
