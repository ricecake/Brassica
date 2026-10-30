# Cookbook: Understanding the Graph & Resource System

Brassica's core innovation is how it handles resource dependencies and execution. This document explains the interplay between Keys, Catalogs, and Recipes.

## ResourceKeys & History
Every resource in the engine is identified by a **Key**.
- **Plain Keys**: (e.g., `GBufferAlbedo`) represent resources that are persistent or updated every frame.
- **History Keys**: (e.g., `History<GBufferAlbedo>`) represent data from the previous frame. 
  - The graph uses these to create "temporal edges". If Node B reads `History<K>`, the graph knows it must run *after* the node that wrote `K` in the previous frame.

## The Resource Catalog
The **Catalog** is a persistent registry of "Resource Types" that have configuration.
- If a key has configuration (e.g. `TerrainMinMaxTexture` has a `gridDim`), it is a **CatalogKey**.
- The engine owns exactly *one* instance of each CatalogKey.
- **Publishing**: Systems (like the TerrainManager) publish configuration to the catalog.
- **Retrieval**: Any node that needs that resource calls `catalog.Get<T>()`. This ensures that all nodes see the same configuration for a given resource type.

## The Recipe & Setup
The `Recipe` is the "contract" for a node in a specific frame.
1. **Setup**: When the graph is built, it calls `Node::Setup(SetupBuilder&, const FrameContext&)`.
2. **Builder**: The `SetupBuilder` records every `Create<K>`, `Read<K>`, and `Write<K>` call.
3. **Lowering**: The engine "lowers" these calls into a `Recipe`.
   - It identifies which keys are inputs and which are outputs.
   - It determines the `ExecutionDomain` (Graphics, Compute, etc.).
   - It determines if the node is `Active` for this frame.

## ResourceRealizations
This is the bridge between the *Graph* (which knows Keys) and the *Renderer* (which needs pointers).
- For every `ResourceKey` in a node's contract, the graph produces a `ResourceRealization`.
- A realization contains:
  - The specific memory handle (from the `PhysicalResourceRegistry`).
  - The access type (Read, Write, ReadWrite).
  - Per-frame data (e.g. ClearColor).
- The **Barrier Batcher** uses these realizations to calculate the minimal set of memory barriers needed to transition resources between nodes.

## Example Flow
1. **Initialization**: `TerrainManager` calls `catalog.Get<TerrainTexture>().gridDim = 64;`.
2. **Frame Start**: `GraphBuilder` is called.
3. **Setup**: `TerrainNode` calls `builder.Create<TerrainTexture>()`.
4. **Lowering**: The Graph sees `TerrainTexture` is a CatalogKey. It fetches the `Desc` from the Catalog (using the `gridDim = 64` config).
5. **Compilation**: The Graph sees that `TerrainNode` produces `TerrainTexture` and `ShadowNode` reads `History<TerrainTexture>`. It schedules them correctly.
6. **Execution**: `TerrainNode` executes. It provides a `ResourceRealization` for the texture. `ShadowNode` then receives that realization and uses it to bind its shaders.
