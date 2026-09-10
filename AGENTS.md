# AGENTS.md

## FrameGraph System Architecture (`brassica::graph`)

The engine uses a strongly-typed, declarative FrameGraph system located in `include/graph/`.

### Core Concepts

1. **Resource Keys (`include/passes/ResourceKeys.hpp`)**:
   - Resource keys are empty struct tags satisfying `brassica::graph::ResourceKey`.
   - Key identity is purely at the type level (`IdOf<T>()`).
   - Active keys in production passes:
     - `Swapchain`: Render/present target
     - `GradientBackground`: Pre-pass environment/gradient background
     - `GBufferPosition`: G-Buffer world positions (`eR16G16B16A16Sfloat`)
     - `GBufferNormal`: G-Buffer normals (`eR16G16B16A16Sfloat`)
     - `GBufferAlbedo`: G-Buffer albedo and valid terrain sentinel alpha (`eR8G8B8A8Unorm`)
     - `GBufferDepth`: Depth buffer
     - `TerrainTLAS`: Ray query acceleration structure
     - `TransmittanceLUT`, `MultiScatteringLUT`: Precomputed atmosphere LUTs

2. **Declarative Node Contract**:
   - Each graph node declares its resource accesses using operations:
     - `graph::Create<K>`: Creates/overwrites resource $K$.
     - `graph::Read<K>`: Reads/samples resource $K$.
     - `graph::Modify<K>`: Reads and modifies resource $K$ in-place.
     - `graph::Transform<From, To>`: Consumes $From$ and produces $To$.
   - Lowered via `graph::Declares<Ops...>`.

3. **Hybrid Deferred-Forward Pipeline Flow**:
   - **Stage 0**:
     - `GradientNode`: `Create<GradientBackground>`
     - `TerrainNode`: `Create<GBufferPosition>`, `Create<GBufferNormal>`, `Create<GBufferAlbedo>`, `Create<GBufferDepth>`, `Create<TerrainTLAS>`
   - **Stage 1 (Deferred Shading)**:
     - `DeferredNode`: `Read<GBufferPosition>`, `Read<GBufferNormal>`, `Read<GBufferAlbedo>`, `Read<GradientBackground>`, `Read<TerrainTLAS>`, `Modify<Swapchain>`
   - **Stage 2 (Forward Translucent Pass)**:
     - `WaterNode`: `Read<GBufferPosition>`, `Read<GBufferDepth>`, `Read<GBufferAlbedo>`, `Modify<Swapchain>`
     - Renders translucent screen-space water plane at $Y=0.0$ with depth-based extinction, wave normals, Fresnel, specular lighting, and alpha blending over deferred shading output.

4. **Physical Execution Backend (`PhysicalExecutionBackend`, `PhysicalRegistry`)**:
   - Automatically provisions textures/buffers with temporal lifetime VMA aliasing.
   - Translates barriers via `BarrierTranslator` using Vulkan 1.3 `vkCmdPipelineBarrier2`.
   - Handles dynamic rendering `vkCmdBeginRendering`/`vkCmdEndRendering` using tracked image layouts and `loadOp` (`eClear` for fresh writes, `eLoad` for `Modify<K>` accesses).

### Testing & Verification

- FrameGraph standalone unit tests: `make graph-test`
- Full test suite: `cmake --build build --target tests && ctest --test-dir build --output-on-failure`
- Headless engine verification: `./build/bin/sandbox --headless 10`
