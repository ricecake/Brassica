# Cookbook: Adding a New Render Pass

To add a new rendering pass to Brassica, you should follow the node-based pattern. This ensures the graph can automatically handle memory barriers and phase ordering.

## Steps

### 1. Define a new ResourceKey
If your pass produces or consumes a new type of resource, define it in the `brassica::graph` namespace.

```cpp
// Example: A new noise texture for a specific effect
struct NoiseTextureKey {
    static constexpr Lifetime kLifetime = Lifetime::Transient;
    ResourceDesc Desc(const FrameContext& ctx) const {
        ResourceDesc d{};
        d.width = 512;
        d.height = 512;
        d.formatCode = ...; // Your format
        return d;
    }
};
```

### 2. Implement the Node
Create a class that satisfies the `BuilderNode` concept. This means providing a `Setup` method that takes a `SetupBuilder` and a `FrameContext`, and an `Execute` method that takes a `NodeContext`.

```cpp
class MyNewPassNode {
public:
    // Setup is where you declare your resource dependencies and configuration.
    // The Graph uses this to build the dependency graph and provide a Recipe.
    void Setup(graph::SetupBuilder& builder, const graph::FrameContext& ctx) {
        builder.Domain(graph::ExecutionDomain::Graphics);
        builder.Active(true);
        
        // Declare dependencies
        builder.Create<NoiseTextureKey>(); 
        // You can also use Read<K>(), Write<K>(), Modify<K>(), etc.
    }

    // Execute is where you perform the actual rendering commands.
    void Execute(graph::NodeContext& ctx) {
        // Get the pipeline
        auto pipeline = BindGraphics(ctx, m_library, m_stages, m_state, m_colors, m_push, m_stage, m_depth);
        
        if (pipeline.pipeline) {
            // Perform your draw calls here
            // Use ctx.resources.IndexOf<K>() to get bindless indices
        }
    }

private:
    PipelineLibrary& m_library;
    // ... other member variables
};
```

### 3. Register the Node in the Graph
Add the node to your `GraphBuilder`. The builder handles the type-erasure into a `NodeHandle`.

```cpp
graphBuilder.AddNode<MyNewPassNode>(...);
```

## Best Practices
- **Use Helpers**: Always use `BindGraphics` or `BindCompute` instead of raw `vkCmdBindPipeline` calls to ensure descriptor sets and push constants are handled consistently.
- **Be Explicit**: Declare every resource in your `Setup` call. The `CheckDeclared` guard will warn you if you miss one.
- **Phase Selection**: If your pass is a "heavy" compute task that doesn't touch the GBuffer, consider putting it in a custom phase to allow the scheduler to run it in parallel with other tasks.
- **HostWriteNodes**: For data that needs to be sent from the CPU to the GPU (like constant buffers or staging buffers), use the `HostWriteNode` or `PredefinedBufferNode` helpers. They automatically handle the `Transfer` domain and staging logic.
