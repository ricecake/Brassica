# Cookbook: Shader Usage

Brassica uses a unified `Shader` class that handles loading, hot-reloading, and metadata for both Graphics and Compute pipelines.

## Shader Loading and Hot Reloading
The `ShaderWatcher` automatically monitors the `shaders/` directory. When a shader file is modified, the `Shader` object is updated, and the `PipelineLibrary` will re-resolve any pipelines using that shader.

- **Loading**: Use `Shader::Load("path/to/shader.frag")` to create a shader object.
- **Hot Reload**: The `Shader` class handles the internal state. You don't need to do anything special; the next time `BindGraphics` or `BindCompute` is called, it will use the new pipeline.

## Descriptor Sets (The "Convention")
Brassica uses a standard descriptor set layout for every pass:
- **Set 0**: Frame Data (Camera, Time, Frame Index). Always bound.
- **Set 1**: Global Bindless Resources (Textures, Buffers). Always bound.
- **Set 2+**: Node-specific "Extra Sets" (e.g. Atomic Counters, specific material data).

When using `BindGraphics` or `BindCompute`, the `ExtraSets` struct is used to define these additional sets.

## Push Constants
Push constants are the preferred method for high-frequency updates (e.g. per-object transforms, small arrays of light data).

- **Graphics**: Use `BindGraphics<PushT>(..., push, pushStage)`.
- **Compute**: Use `BindCompute(..., push)`.

The `PushT` type must be a POD (Plain Old Data) struct.

## Bindless Access
To access resources in shaders:
1. Ensure the resource is registered in the `ResourceCatalog`.
2. Use `ctx.resources.IndexOf<K>()` in your C++ code to get the bindless index.
3. Pass that index to your shader via Push Constants or a Storage Buffer.

## Example: Compute Shader with Bindless Texture
```cpp
// C++ Side
struct MyPushData {
    uint32_t inputTextureIndex;
    float time;
};

auto push = MyPushData{ ctx.resources.IndexOf<InputTexture>(), ctx.frameContext.time };
BindCompute(ctx, m_library, m_shader, push);

// GLSL Side
layout(set = 1, binding = 0) uniform sampler2D globalTextures[]; // Bindless
layout(push_constant) uniform PushData {
    uint inputTextureIndex;
    float time;
} push;

void main() {
    uint texIdx = push.inputTextureIndex;
    vec4 color = texture(globalTextures[texIdx], uv);
    // ...
}
```
