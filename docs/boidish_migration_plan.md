# Boidish to Brassica Feature Inventory & Migration Plan

## Executive Summary & Migration Principles

`Boidish` is an OpenGL 4.3-based 3D simulation and rendering framework featuring procedural terrain, atmospheric scattering, compute particle systems, and screen-space post-processing.
`Brassica` is a modern C++23 Vulkan 1.4 engine utilizing FrameGraph resource management, task/mesh shaders (`VK_EXT_mesh_shader`), bindless descriptor indexing, VMA allocation, EnTT ECS, and hardware ray tracing/queries.

### Key Adaptation Principles
1. **No 1:1 Code Porting**: Adapt all legacy OpenGL pipelines to Vulkan 1.4 idiomatic patterns (FrameGraph passes, timeline semaphores, VMA buffers, descriptor sets).
2. **Mesh & Task Shader Modernization**: Re-implement legacy vertex/geometry shader pipelines (specifically **Particles** and **Grass/Foliage**) using Vulkan **Task & Mesh Shaders** (`VK_EXT_mesh_shader`). This eliminates CPU geometry generation, vertex buffer overhead, and complex indirect draw fixup dispatches.
3. **Decoupled Architecture via FrameGraph**: All migrated passes publish and consume resources through `FrameGraphBlackboard` and `PassResource` types (`FrameGraphTexture2D`, `FrameGraphTexture3D`, `FrameGraphSSBO`, `FrameGraphUBO`).
4. **ECS Integration**: Entity behaviors (boids, procedural walking creatures, emitters) map cleanly to `entt::registry` components.

---

## 1. Complete Feature Inventory of Boidish

### Core Priority Systems
| System | Boidish Source Files | Boidish Shader Files | Migration Target in Brassica |
| :--- | :--- | :--- | :--- |
| **Volumetric Lighting** | `VolumetricLightingEffect.cpp` | `volumetric_injection.comp`, `volumetric_integration.comp`, `volumetric_temporal_accumulation_2d.comp`, `volumetric_composite.frag` | `VolumetricLightingPass` (3D Froxel Grid $160\times 90\times 64$) |
| **Cloud System** | `PostProcessingManager.cpp` | `cloud_3d_volume_bake.comp`, `cloud_weather_bake.comp`, `cloud_shadow_bake.comp`, `cloud_spatial_filter.comp`, `cloud_temporal_reprojection.comp`, `helpers/clouds.glsl` | `CloudBakePass`, `CloudRenderPass`, `CloudShadowPass`, `CloudTemporalFilterPass` |
| **Atmosphere System** | `atmosphere_manager.cpp`, `AtmosphereEffect.cpp` | `transmittance_lut.comp`, `multiscattering_lut.comp`, `sky_view_lut.comp`, `aerial_perspective_lut.comp`, `sky_to_sh.comp`, `sky.vert`, `sky.frag`, `atmosphere_composite.frag` | Extend `AtmosphereLUTPass`, `AtmosphereSkyPass` (Task/Mesh Shader sky quad), Aerial Perspective in `DeferredPass` |
| **Particle System** | `fire_effect_manager.cpp`, `fire_effect.cpp` | `fire_behavior.comp`, `fire_lifecycle.comp`, `particle_alloc.comp`, `particle_grid_build.comp`, `particle_command_fixup.comp`, `fire.vert`, `fire.frag` | `ParticleSimulationPass` + Task & Mesh Shaders (`particle.task`, `particle.mesh`, `particle.frag`) |
| **Grass & Foliage** | `grass_manager.cpp`, `decor_manager.cpp` | `grass_placement.comp`, `grass_pre_pass.comp`, `grass_command_fixup.comp`, `grass.vert`, `grass.frag`, `decor_placement.comp`, `decor_cull.comp`, `fern.vert`, `fern.frag` | `GrassPlacementPass` + Task & Mesh Shaders (`grass.task`, `grass.mesh`, `grass.frag`, `foliage.task`, `foliage.mesh`, `foliage.frag`) |
| **Clustered Lighting** | `light_manager.cpp` | `cluster_light_assignment.comp`, `clustered_lighting.glsl` | `ClusteredLightingPass` ($16\times 9\times 32$ cluster grid) + `DeferredPass` integration |
| **GI / GTAO / SSS** | `UnifiedScreenSpaceEffect.cpp` | `unified_screen_space.comp`, `unified_screen_space_composite.frag` | `UnifiedScreenSpacePass` (Compute SSGI/GTAO/SSS) + `TemporalAccumulationPass` |
| **Bloom & Tonemapping** | `BloomEffect.cpp` | `bloom_downsample.comp`, `bloom_upsample.frag`, `bloom_composite.frag`, `ltm_fuse.comp`, `tonemapping.glsl`, `postprocess.frag` | `HiZPyramidPass`, `BloomPass`, `HistogramAutoExposurePass`, `TonemapAndGradingPass` |

### Auxiliary Systems
| System | Boidish Source Files | Boidish Shader Files | Brassica Target Strategy |
| :--- | :--- | :--- | :--- |
| **Terrain Runtime Deformation** | `terrain_deformation_manager.cpp` | `terrain_bake.comp`, `terrain_height_mip.comp`, `terrain_probes.comp` | `TerrainDeformPass` operating on `TerrainClipmap` |
| **Dynamic CSM Shadows** | `shadow_manager.cpp`, `shadow_render_pass.cpp` | `shadow_depth.vert`, `shadow_depth.frag` | `ShadowMapPass` for dynamic glTF mesh instances |
| **Hi-Z Occlusion Culling** | `hiz_manager.cpp` | `terrain_hiz_generate.comp`, `occlusion_cull.comp` | `HiZPyramidPass` (Compute Pass building depth pyramid) |
| **Trails & Visual Effects** | `trail_render_manager.cpp`, `mesh_explosion_manager.cpp`, `shockwave_effect.cpp` | `trail_tess.comp`, `trail.vert`, `trail.frag`, `mesh_explosion.comp`, `shockwave.frag`, `akira.frag` | Task/Mesh Shader `TrailPass`, `MeshExplosionPass`, `ShockwavePass` |
| **Neural / MLP Networks** | `mlp_network.cpp` | `mlp_eval.comp`, `mlp_train.comp`, `mlp_nca.comp`, `mlp_particles.comp` | `MLPComputePass` (In-shader neural cellular automata/particles) |
| **Entities & Simulation** | `bvh_spatial_structure.cpp`, `procedural_walking_creature.cpp` | N/A (CPU / compute) | `entt::registry` systems (`BoidsSystem`, `IKCreatureSystem`) |
| **3D Positional Audio** | `audio_manager.cpp`, `sound.cpp` | N/A (OpenAL) | `AudioSystem` (OpenAL-Soft backend integrated with EnTT) |

---

## 2. Detailed Architecture & Adaptation Plans for Core Systems

---

### Core System 1: Volumetric Lighting System
- **Legacy Approach**: Boidish uses 4 cascaded 3D froxel grids ($160\times 90\times 64$ RGBA16F). Compute shaders inject lighting, raymarch along depth, and blend into a 2D composite texture.
- **Brassica Adaptation**:
  - **Class**: `VolumetricLightingPass` (`include/passes/VolumetricLightingPass.hpp`, `src/passes/VolumetricLightingPass.cpp`).
  - **FrameGraph Resources**:
    - `FrameGraphTexture3D`: `VolumetricInjectionGrid` ($160\times 90\times 64$, `eR16G16B16A16Sfloat`).
    - `FrameGraphTexture3D`: `VolumetricIntegratedGrid` ($160\times 90\times 64$, `eR16G16B16A16Sfloat`).
    - `FrameGraphTexture2D`: `VolumetricHistory2D` (Viewport size, `eR16G16B16A16Sfloat`).
  - **Shaders**:
    - `shaders/volumetric/injection.comp`: Samples Sun directional light, hardware ray query shadows/CSM, sky transmittance LUT, and point lights.
    - `shaders/volumetric/integration.comp`: March front-to-back along z-slices to calculate accumulated scattering and extinction.
    - `shaders/volumetric/composite.frag`: Screen-space volumetric fog integration during deferred composition.

---

### Core System 2: Volumetric Cloud System
- **Legacy Approach**: Raymarches a cloud slab using pre-baked 3D Worley noise ($128^3$) and 2D weather maps ($1024^2$), rendering at half-resolution with temporal reprojection.
- **Brassica Adaptation**:
  - **Classes**:
    - `CloudBakePass` (`include/passes/CloudBakePass.hpp`): Generates 3D Worley noise and weather maps on initialization or parameter changes.
    - `CloudRenderPass` (`include/passes/CloudRenderPass.hpp`): Executes half-resolution raymarching.
    - `CloudTemporalFilterPass` (`include/passes/CloudTemporalFilterPass.hpp`): Performs motion-vector reprojection and depth-aware bilateral upsampling.
  - **FrameGraph Resources**:
    - `FrameGraphTexture3D`: `CloudNoise3D` ($128\times 128\times 128$, `eR8G8B8A8Unorm`).
    - `FrameGraphTexture2D`: `CloudWeatherMap` ($1024\times 1024$, `eR8G8B8A8Unorm`).
    - `FrameGraphTexture2D`: `LowResCloudColor` (Half resolution, `eR16G16B16A16Sfloat`).
    - `FrameGraphTexture2D`: `CloudShadowMap` ($2048\times 2048$, `eR16Unorm`).
  - **Shaders**: `shaders/clouds/bake_noise.comp`, `shaders/clouds/render.comp`, `shaders/clouds/shadow.comp`, `shaders/clouds/temporal_upsample.comp`.

---

### Core System 3: Atmosphere & Sky System
- **Legacy Approach**: Computes 2D Transmittance ($256\times 64$), Multi-Scattering ($32\times 32$), Sky-View ($192\times 108$), and 3D Aerial Perspective ($32\times 32\times 32$) LUTs.
- **Brassica Adaptation**:
  - **Current State**: Brassica already has `AtmosphereLUTPass` precomputing Transmittance and Multi-Scattering LUTs.
  - **Extension**:
    - Expand `AtmosphereLUTPass` (`include/passes/AtmosphereLUTPass.hpp`) to generate `SkyViewLUT` ($192\times 108$) and `AerialPerspectiveLUT` ($32\times 32\times 32$).
    - Implement `AtmosphereSkyPass` (`include/passes/AtmosphereSkyPass.hpp`) using Vulkan Task/Mesh shaders (`shaders/atmosphere/sky.task`, `shaders/atmosphere/sky.mesh`, `shaders/atmosphere/sky.frag`) to render the background atmosphere and celestial bodies.
    - Integrate `AerialPerspectiveLUT` sampling into `DeferredPass` (`shaders/deferred.frag`) for distant terrain and model fogging.

---

### Core System 4: Particle System (Mesh Shader Modernization)
- **Legacy Approach**: Compute shaders simulate particles and build indirect draw commands (`DrawArraysIndirect`). A vertex shader expands billboards.
- **Brassica Adaptation (Task & Mesh Shader Pipeline)**:
  - **Classes**:
    - `ParticleSimulationPass` (`include/passes/ParticleSimulationPass.hpp`): Compute pass updating particle position, curl noise turbulence, velocity, life, and spatial sorting.
    - `ParticleRenderPass` (`include/passes/ParticleRenderPass.hpp`): Task and Mesh shader pass rendering particle billboards directly on GPU.
  - **FrameGraph Resources**:
    - `FrameGraphSSBO`: `ParticleStateSSBO` (Array of `Particle` structs containing `pos`, `vel`, `color`, `life`, `size`).
  - **Shaders**:
    - `shaders/particles/simulate.comp`: Physics, curl noise, life decay, dead particle list recycle.
    - `shaders/particles/particle.task`: Inspects particle blocks (64 particles/workgroup), culls dead/frustum-culled particles using `subgroupBallot`, and emits mesh shader tasks via `EmitMeshTasksEXT`.
    - `shaders/particles/particle.mesh`: Generates camera-facing billboard quad vertices and index primitives directly in meshlet output memory.
    - `shaders/particles/particle.frag`: Soft depth-clipping against G-Buffer depth, PBR emissive blending.

---

### Core System 5: Grass & Foliage System (Mesh Shader Modernization)
- **Legacy Approach**: Compute placement -> Compute pre-pass/fixup -> Vertex shader instance expansion with MDI (`DrawArraysIndirect`).
- **Brassica Adaptation (Task & Mesh Shader Pipeline)**:
  - **Classes**:
    - `GrassPlacementPass` (`include/passes/GrassPlacementPass.hpp`): Evaluates `TerrainClipmap` heights/normals and FastNoise2 biomes to build active patch grids.
    - `GrassRenderPass` (`include/passes/GrassRenderPass.hpp`): Task & Mesh shader pipeline rendering grass blades and foliage models.
  - **Shaders**:
    - `shaders/grass/placement.comp`: Generates instance density maps across terrain chunks.
    - `shaders/grass/grass.task` / `foliage.task`: Evaluates terrain patch distance, frustum visibility, and CDLOD morphing parameters. Emits active mesh tasks.
    - `shaders/grass/grass.mesh`: Procedurally generates grass blade meshes (3 to 7 segments based on distance LOD) with wind sway, width/height jitter, and curvature on the GPU.
    - `shaders/grass/foliage.mesh`: Emits instanced foliage meshlets for shrubs and trees.
    - `shaders/grass/grass.frag`: Dual-sided subsurface scattering (SSS), PBR terrain color blending, and wind highlights.

---

### Core System 6: Clustered Lighting System
- **Legacy Approach**: $16\times 9\times 24$ frustum grid built on CPU/GPU, assign point/spot lights per cluster via compute shader, evaluate in deferred fragment shader.
- **Brassica Adaptation**:
  - **Class**: `ClusteredLightingPass` (`include/passes/ClusteredLightingPass.hpp`).
  - **FrameGraph Resources**:
    - `FrameGraphSSBO`: `ClusterAABBsSSBO` ($16\times 9\times 32$ cluster bounding boxes).
    - `FrameGraphSSBO`: `ClusterLightGridSSBO` (Light counts and offsets per cluster).
    - `FrameGraphSSBO`: `ClusterLightIndexSSBO` (Flat array of active light indices).
  - **Shaders**:
    - `shaders/lighting/build_clusters.comp`: Builds cluster frustum AABBs (recomputed on FOV change).
    - `shaders/lighting/cluster_cull.comp`: Intersects active EnTT point/spot light components with cluster AABBs.
  - **Deferred Pass Integration**: Update `shaders/deferred.frag` to compute $3D$ cluster index $(x, y, z)$ from fragment screen coordinate and linear depth, looping through cluster light indices for fast multi-light shading.

---

### Core System 7: Post-Processing Pipeline (GI / GTAO / SSS & Bloom / Tonemapping)
- **Legacy Approach**: `UnifiedScreenSpaceEffect` combined SSGI, GTAO, and SSS in one compute shader. `BloomEffect` handled downsampling, upsampling, LTM exposure fusion, and tonemapping.
- **Brassica Adaptation**:
  - **Classes & Passes**:
    - `HiZPyramidPass` (`include/passes/HiZPyramidPass.hpp`): Generates mipmapped Hi-Z depth texture from G-Buffer depth (`shaders/postprocessing/hiz_generate.comp`).
    - `UnifiedScreenSpacePass` (`include/passes/UnifiedScreenSpacePass.hpp`): Raymarches Hi-Z depth and G-Buffer for SSGI, GTAO, and Screen-Space Subsurface Scattering / Contact Shadows (`shaders/postprocessing/unified_screen_space.comp`).
    - `TemporalAccumulationPass` (`include/passes/TemporalAccumulationPass.hpp`): Reprojects SSGI/GTAO/SSS using G-Buffer velocity vectors with exponential moving average (EMA) and variance clamping (`shaders/postprocessing/temporal_accumulation.comp`).
    - `BloomPass` (`include/passes/BloomPass.hpp`): Multi-pass compute downsampling (Karis threshold filtering), upsampling (tent filter), and LTM exposure fusion (`shaders/postprocessing/bloom_downsample.comp`, `shaders/postprocessing/bloom_upsample.comp`, `shaders/postprocessing/ltm_fuse.comp`).
    - `HistogramAutoExposurePass` (`include/passes/HistogramAutoExposurePass.hpp`): Generates 256-bin log-luminance histogram and calculates adapted exposure (`shaders/postprocessing/histogram.comp`, `shaders/postprocessing/exposure_adapt.comp`).
    - `TonemapAndGradingPass` (`include/passes/TonemapAndGradingPass.hpp`): Evaluates ASC CDL, white balance, ACES/Uchimura/AgX tonemapping, film grain, and outputs final LDR color (`shaders/postprocessing/tonemap_grading.frag`).

---

## 3. Phased Implementation Roadmap

```
Phase 1: Sky, Atmosphere & Volumetric Lighting
  ├─ 1.1 Extend AtmosphereLUTPass (Sky-View & Aerial Perspective LUTs)
  ├─ 1.2 Implement AtmosphereSkyPass (Task/Mesh Shader Sky Quad)
  └─ 1.3 Implement VolumetricLightingPass (3D Froxel Grid)

Phase 2: Volumetric Cloud System
  ├─ 2.1 Implement CloudBakePass (3D Worley Noise & Weather Map)
  ├─ 2.2 Implement CloudRenderPass (Raymarched Cloud Slab)
  └─ 2.3 Implement CloudShadowPass & CloudTemporalFilterPass

Phase 3: Mesh Shader Particle System
  ├─ 3.1 Implement ParticleSimulationPass (Compute Physics & Curl Noise)
  └─ 3.2 Implement ParticleRenderPass (particle.task, particle.mesh, particle.frag)

Phase 4: Mesh Shader Grass & Foliage (Decor) System
  ├─ 4.1 Implement GrassPlacementPass (Clipmap & FastNoise2 Biome Placement)
  └─ 4.2 Implement GrassRenderPass (grass.task, grass.mesh, grass.frag)

Phase 5: Clustered Lighting System
  ├─ 5.1 Implement ClusteredLightingPass (Grid AABBs & Light Culling)
  └─ 5.2 Integrate Clustered Light Lookups in DeferredPass (shaders/deferred.frag)

Phase 6: Post-Processing Chain
  ├─ 6.1 Implement HiZPyramidPass
  ├─ 6.2 Implement UnifiedScreenSpacePass (SSGI, GTAO, SSS)
  ├─ 6.3 Implement TemporalAccumulationPass
  ├─ 6.4 Implement BloomPass & HistogramAutoExposurePass
  └─ 6.5 Implement TonemapAndGradingPass

Phase 7: Secondary Systems & ECS
  ├─ 7.1 Implement TrailPass & MeshExplosionPass (Task/Mesh Shaders)
  ├─ 7.2 Implement TerrainDeformPass (Runtime Clipmap Craters/Deformation)
  ├─ 7.3 Implement EnTT BoidsSystem & OpenAL AudioSystem
  └─ 7.4 Implement ImGui Profiler & Environment Debug Widgets
```

---

## 4. Summary Table of Files to Create / Modify

| File Path | Purpose / Action |
| :--- | :--- |
| `include/passes/VolumetricLightingPass.hpp` | Header for 3D Froxel Volumetric Lighting Pass |
| `src/passes/VolumetricLightingPass.cpp` | Implementation for 3D Froxel Volumetric Lighting Pass |
| `shaders/volumetric/injection.comp` | Compute shader injecting light into 3D froxel grid |
| `shaders/volumetric/integration.comp` | Compute shader integrated scattering/extinction |
| `shaders/volumetric/composite.frag` | Fragment shader blending fog onto G-Buffer |
| `include/passes/CloudBakePass.hpp` | Header for 3D Worley noise & weather map baking |
| `src/passes/CloudBakePass.cpp` | Implementation for Cloud noise baking |
| `include/passes/CloudRenderPass.hpp` | Header for Raymarched Volumetric Cloud Pass |
| `src/passes/CloudRenderPass.cpp` | Implementation for Raymarched Volumetric Cloud Pass |
| `shaders/clouds/render.comp` | Compute shader raymarching cloud volume |
| `shaders/clouds/temporal_upsample.comp` | Compute shader for motion-reprojected cloud upsampling |
| `include/passes/AtmosphereLUTPass.hpp` | Update to compute Sky-View and Aerial Perspective LUTs |
| `src/passes/AtmosphereLUTPass.cpp` | Update implementation for extra LUTs |
| `include/passes/AtmosphereSkyPass.hpp` | Header for Task/Mesh shader sky dome pass |
| `src/passes/AtmosphereSkyPass.cpp` | Implementation for Task/Mesh shader sky dome pass |
| `shaders/atmosphere/sky.task` | Task shader for sky quad |
| `shaders/atmosphere/sky.mesh` | Mesh shader for sky quad |
| `shaders/atmosphere/sky.frag` | Fragment shader for physical sky & sun/moon |
| `include/passes/ParticleSimulationPass.hpp` | Header for compute particle simulation pass |
| `src/passes/ParticleSimulationPass.cpp` | Implementation for compute particle simulation |
| `include/passes/ParticleRenderPass.hpp` | Header for Task/Mesh shader particle render pass |
| `src/passes/ParticleRenderPass.cpp` | Implementation for Task/Mesh shader particle render pass |
| `shaders/particles/simulate.comp` | Compute shader for particle physics & curl noise |
| `shaders/particles/particle.task` | Task shader for particle block culling |
| `shaders/particles/particle.mesh` | Mesh shader generating camera-facing quad meshlets |
| `shaders/particles/particle.frag` | Fragment shader for particle shading & soft depth |
| `include/passes/GrassPlacementPass.hpp` | Header for grass patch placement pass |
| `src/passes/GrassPlacementPass.cpp` | Implementation for grass placement pass |
| `include/passes/GrassRenderPass.hpp` | Header for Task/Mesh shader grass render pass |
| `src/passes/GrassRenderPass.cpp` | Implementation for Task/Mesh shader grass render pass |
| `shaders/grass/placement.comp` | Compute shader placing grass seeds on clipmap terrain |
| `shaders/grass/grass.task` | Task shader evaluating terrain patch LOD and frustum |
| `shaders/grass/grass.mesh` | Mesh shader generating procedural grass blade meshlets |
| `shaders/grass/grass.frag` | Fragment shader for dual-sided SSS grass PBR shading |
| `include/passes/ClusteredLightingPass.hpp` | Header for Clustered Lighting Pass |
| `src/passes/ClusteredLightingPass.cpp` | Implementation for Clustered Light Assignment |
| `shaders/lighting/cluster_cull.comp` | Compute shader culling point/spot lights per 3D frustum cluster |
| `shaders/deferred.frag` | Update to incorporate clustered lighting lookups |
| `include/passes/HiZPyramidPass.hpp` | Header for Hi-Z Depth Pyramid compute pass |
| `src/passes/HiZPyramidPass.cpp` | Implementation for Hi-Z depth pyramid generation |
| `shaders/postprocessing/hiz_generate.comp` | Compute shader for downsampling depth mips |
| `include/passes/UnifiedScreenSpacePass.hpp` | Header for SSGI / GTAO / SSS compute pass |
| `src/passes/UnifiedScreenSpacePass.cpp` | Implementation for SSGI / GTAO / SSS compute pass |
| `shaders/postprocessing/unified_screen_space.comp` | Compute shader raymarching Hi-Z depth for SSGI/GTAO/SSS |
| `include/passes/TemporalAccumulationPass.hpp` | Header for motion-vector temporal reprojection pass |
| `src/passes/TemporalAccumulationPass.cpp` | Implementation for temporal reprojection |
| `include/passes/BloomPass.hpp` | Header for Compute Bloom downsample/upsample pass |
| `src/passes/BloomPass.cpp` | Implementation for Compute Bloom pass |
| `shaders/postprocessing/bloom_downsample.comp` | Compute shader downsampling with Karis filter |
| `shaders/postprocessing/bloom_upsample.comp` | Compute shader upsampling with tent filter |
| `include/passes/HistogramAutoExposurePass.hpp` | Header for Auto-Exposure luminance histogram pass |
| `src/passes/HistogramAutoExposurePass.cpp` | Implementation for Auto-Exposure pass |
| `include/passes/TonemapAndGradingPass.hpp` | Header for Tonemapping & Color Grading pass |
| `src/passes/TonemapAndGradingPass.cpp` | Implementation for Tonemapping & Color Grading pass |
| `shaders/postprocessing/tonemap_grading.frag` | Fragment shader for ASC CDL, White Balance, ACES/Uchimura |
