#pragma once

#include <utility>

#include <glm/glm.hpp>

namespace Brassica {
	namespace Constants {
		namespace General {
			namespace Math {
				constexpr float Pi = 3.14159265358979323846f;
			} // namespace Math

			namespace Colors {
				// Default ambient light color: glm::vec3(90.0f/255.0f, 81.0f/255.0f, 62.0f/255.0f)
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
				consteval int DefaultWidth() {
					return 1280;
				}

				consteval int DefaultHeight() {
					return 720;
				}
			} // namespace Window

			namespace Camera {
				constexpr float DefaultFOV() {
					return 45.0f;
				}

				constexpr float DefaultNearPlane() {
					return 0.1f;
				}

				constexpr float DefaultFarPlane();

				constexpr float MinHeight() {
					return 0.1f;
				}

				constexpr float MinSpeed() {
					return 0.5f;
				}

				constexpr float DefaultSpeed() {
					return 15.0f;
				}

				constexpr float FirstPersonEyeHeight() {
					return 4.8f;
				}

				constexpr float FirstPersonCrouchHeight() {
					return 1.5f;
				}

				constexpr float FirstPersonSprintMultiplier() {
					return 2.0f;
				}

				constexpr float FirstPersonJumpForce() {
					return 12.5f;
				}

				constexpr float FirstPersonGravity() {
					return 18.0f;
				}

				constexpr float FirstPersonGroundSmoothing() {
					return 5.0f;
				}

				constexpr float SpeedStep() {
					return 2.5f;
				}

				constexpr float RollSpeed() {
					return 45.0f;
				}

				// Path following
				constexpr float DefaultPathSpeed() {
					return 20.0f;
				}

				constexpr float PathBankFactor() {
					return 1.8f;
				}

				constexpr float PathBankSpeed() {
					return 3.5f;
				}

				constexpr float ChaseTrailBehind() {
					return 15.0f;
				}

				constexpr float ChaseElevation() {
					return 5.0f;
				}

				constexpr float ChaseLookAhead() {
					return 10.0f;
				}

				constexpr float ChaseResponsiveness() {
					return 1.5f;
				}

				constexpr float PathFollowSmoothing() {
					return 5.0f;
				}
			} // namespace Camera
		} // namespace Project

		namespace Library {
			namespace Input {
				constexpr int MaxKeys() {
					return 1024;
				}

				constexpr int MaxMouseButtons() {
					return 8;
				}
			} // namespace Input
		} // namespace Library

		namespace Class {
			namespace Terrain {
			} // namespace Terrain

			namespace Shadows {
				consteval int MaxLights() {
					return 10;
				}

				consteval int MaxCascades() {
					return 4;
				}

				consteval int MaxShadowMaps() {
					return 16;
				}

				consteval int MapSize() {
					return 2048;
				}

				consteval float DefaultSceneRadius() {
					return 500.0f;
				}

				consteval float DefaultFOV() {
					return 45.0f;
				}

				// Cascade split distances (logarithmic distribution)
				// Near splits are tighter for crisp close shadows
				// Far cascade acts as catchall for distant terrain
				consteval float CascadeSplit0() {
					return 20.0f;
				}

				consteval float CascadeSplit1() {
					return 50.0f;
				}

				consteval float CascadeSplit2() {
					return 150.0f;
				}

				consteval float CascadeSplit3() {
					return 700.0f;
				}

				// Grid snapping sizes per cascade (finer for near, coarser for far)
				consteval float GridSnapCascade0() {
					return 0.25f;
				}

				consteval float GridSnapCascade1() {
					return 1.0f;
				}

				consteval float GridSnapCascade2() {
					return 4.0f;
				}

				consteval float GridSnapCascade3() {
					return 8.0f;
				}
			} // namespace Shadows

			namespace Terrain {
				consteval float PathCorridorWidth() {
					return 0.15f;
				}
			} // namespace Terrain

			namespace Particles {
				consteval int MaxParticles() {
					return 128000;
				}

				consteval int AmbientParticleScale() {
					return 8192;
				}

				consteval int MaxEmitters() {
					return 100;
				}

				consteval int ComputeGroupSize() {
					return 256;
				}

				consteval int ParticleGridSize() {
					return 131072;
				}

				consteval float ParticleGridCellSize() {
					return 2.0f;
				}

				consteval float DefaultAmbientDensity() {
					return 1.0f;
				}
			} // namespace Particles

			namespace Explosions {
				consteval int MaxFragments() {
					return 50000;
				}

				consteval int ComputeGroupSize() {
					return 64;
				}

				consteval float DefaultVelocity() {
					return 10.0f;
				}

				consteval float DefaultRandomVelocity() {
					return 5.0f;
				}
			} // namespace Explosions

			namespace Shockwaves {
				consteval int MaxShockwaves() {
					return 16;
				}

				consteval float DefaultIntensity() {
					return 0.5f;
				}

				consteval float DefaultRingWidth() {
					return 3.0f;
				}

				consteval float DefaultDuration() {
					return 1.2f;
				} // Based on CreateExplosion logic

				consteval glm::vec3 DefaultColor() {
					return glm::vec3(1.0f, 0.6f, 0.2f);
				}
			} // namespace Shockwaves

			namespace SdfVolumes {
				consteval int MaxSources() {
					return 128;
				}

				consteval float DefaultRadius() {
					return 5.0f;
				}

				consteval float DefaultSmoothness() {
					return 2.0f;
				}
			} // namespace SdfVolumes

			namespace Trails {
				consteval int DefaultMaxLength() {
					return 250;
				}

				consteval int DefaultTrailLength() {
					return 10;
				}

				consteval int Segments() {
					return 8;
				}

				consteval int CurveSegments() {
					return 4;
				}

				consteval float BaseThickness() {
					return 0.06f;
				}

				consteval float DefaultRoughness() {
					return 0.3f;
				}

				consteval float DefaultMetallic() {
					return 0.0f;
				}

				consteval int FloatsPerVertex() {
					return 9;
				}

				consteval int InitialVertexCapacity() {
					return 500000;
				}

				consteval float GrowthFactor() {
					return 1.5f;
				}
			} // namespace Trails

			namespace Rendering {
				consteval int BlurPasses() {
					return 4;
				}
			} // namespace Rendering

			namespace Akira {
				consteval float DefaultGrowthDuration() {
					return 0.5f;
				}

				consteval float DefaultFadeDuration() {
					return 3.0f;
				}

				consteval float DefaultRadius() {
					return 20.0f;
				}
			} // namespace Akira

			namespace Checkpoint {
				consteval float DefaultRadius() {
					return 10.0f;
				}

				consteval float DefaultHaloWidth() {
					return 2.0f;
				}

				consteval float DefaultAuraWidth() {
					return 5.0f;
				}

				consteval float DefaultLifespan() {
					return 60.0f;
				}

				namespace Colors {
					consteval float GoldR() {
						return 1.0f;
					}

					consteval float GoldG() {
						return 0.84f;
					}

					consteval float GoldB() {
						return 0.0f;
					}

					consteval float SilverR() {
						return 0.75f;
					}

					consteval float SilverG() {
						return 0.75f;
					}

					consteval float SilverB() {
						return 0.75f;
					}

					consteval float BlackR() {
						return 0.01f;
					}

					consteval float BlackG() {
						return 0.01f;
					}

					consteval float BlackB() {
						return 0.01f;
					}

					consteval float BlueR() {
						return 0.0f;
					}

					consteval float BlueG() {
						return 0.5f;
					}

					consteval float BlueB() {
						return 1.0f;
					}

					consteval float NeonGreenR() {
						return 0.2f;
					}

					consteval float NeonGreenG() {
						return 1.0f;
					}

					consteval float NeonGreenB() {
						return 0.2f;
					}

					consteval glm::vec3 Gold() {
						return glm::vec3(GoldR(), GoldG(), GoldB());
					}

					consteval glm::vec3 Silver() {
						return glm::vec3(SilverR(), SilverG(), SilverB());
					}

					consteval glm::vec3 Black() {
						return glm::vec3(BlackR(), BlackG(), BlackB());
					}

					consteval glm::vec3 Blue() {
						return glm::vec3(BlueR(), BlueG(), BlueB());
					}

					consteval glm::vec3 NeonGreen() {
						return glm::vec3(NeonGreenR(), NeonGreenG(), NeonGreenB());
					}
				} // namespace Colors
			} // namespace Checkpoint
		} // namespace Class
	} // namespace Constants
} // namespace Brassica
