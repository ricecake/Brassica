# Brassica Architecture Design Document

## Overview
Brassica is a high-performance, Vulkan-based game engine designed with a flexible, graph-based rendering architecture. It emphasizes a clean separation between high-level engine logic and low-level graphics API calls, using a node-based system to manage complex frame dependencies and execution ordering.

## Core Architecture

### 1. Node-Based Rendering Graph
The engine's rendering pipeline is constructed as a Directed Acyclic Graph (DAG) of "Nodes". Each node represents a discrete rendering task (e.g., "Shadow Pass", "Deferred Shading", "Water Rendering").

- **Execution Order**: Ordering is determined by two mechanisms:
  - **Resource Dependencies**: Explicit Read/Write dependencies on `ResourceKey` types. The graph compiler uses these to inject necessary memory barriers.
  - **Phases**: Coarse-grained bands (e.g., `Prepare`, `GBuffer`, `Lighting`, `Composite`, `ToneMap`, `Overlay`) provide high-level grouping and allow for easier insertion of new passes without complex dependency rewiring.
- **Recipe System**: Each node declares a `Recipe` every frame. The recipe defines its execution domain (Graphics, Compute, Transfer, Host), activity state, and the specific `ResourceRealizations` (keys with per-frame metadata) it interacts with.

### 2. Resource Management & Ownership
Brassica uses a unique "Resource-Type-Ownership" model to handle persistent vs. transient data.

- **ResourceKey**: A type-level identity for a resource. Keys can be wrapped in `History<K>` to represent data from previous frames.
- **ResourceCatalog**: A persistent registry of "owning" resource types. If a resource type has configuration (like a texture's grid dimensions), it is stored in the catalog once. This allows configuration to be "published" once and shared across all nodes that use that resource type.
- **ResourceRealizations**: Every frame, the graph converts abstract `ResourceKeys` into `ResourceRealizations`. This step resolves:
  - Temporal targets (e.g., finding the specific buffer instance for `History<K>`).
  - Per-frame metadata (e.g., clear colors, dynamic resolution).
  - Accessibility (e.g., is this resource currently active in the scene?).

### 3. Execution Model
Nodes are executed by a scheduler that translates the graph into a sequence of command buffer recording calls.

- **NodeContext**: Every node's `Execute` function receives a `NodeContext`. This object abstracts the Vulkan backend, providing:
  - **Bindless Access**: A `ResourceServices` object for high-level lookups (`IndexOf`, `StorageIndexOf`).
  - **Host-to-Device Transfer**: A high-level `Write<K>` and `WriteSpan<K>` API that handles staging buffers and memory copies automatically.
  - **Pipeline Binding**: Helpers like `BindGraphics` and `BindCompute` that handle the boilerplate of descriptor set binding, push constants, and viewport setup.
- **Barrier Batching**: Instead of issuing barriers per-access (which is inefficient), the graph compiler batches barriers per-resource per-execution-slot, merging overlapping requirements into a single `vkCmdPipelineBarrier` call.

### 4. Pipeline Library
The `PipelineLibrary` manages the creation and caching of Vulkan pipelines. It takes a `GraphicsPipelineRequest` (or `ComputePipelineRequest`) and returns a `ResolvedPipeline`. It uses a hash-based cache to ensure that identical state requests are only built once, significantly reducing state change overhead.

## Development Workflow
1. **Define Keys**: Create a new `ResourceKey` in the appropriate namespace.
2. **Define Nodes**: Implement a class that satisfies the Node interface, defining its `Recipe` and `Execute` logic.
3. **Register in Graph**: Add the node to the engine's graph builder, defining its resource dependencies.
4. **Publish Config**: If the resource is a "Catalog" type, use the `ResourceCatalog` to set its configuration during initialization.

## Key Idioms
- **Data-Oriented**: Prefer `ResourceRealizations` and `PushConstants` over high-level objects to keep data contiguous and cache-friendly.
- **Type-Safety**: Use `ResourceRef` constraints to ensure that only valid, declared resources can be accessed during execution.
- **Separation of Concerns**: Keep `Node::Execute` free of Vulkan-specific calls. Use the provided `NodeContext` helpers instead.
