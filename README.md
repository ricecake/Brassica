# Brassica

Brassica is a high-performance, Vulkan-based game engine featuring a sophisticated node-based rendering graph, a persistent resource catalog, and a focus on clean, data-oriented architecture.

## Features
- **Graph-Based Renderer**: A dynamic execution graph that handles complex resource dependencies, automated barrier batching, and coarse-grained phase ordering.
- **Resource Catalog**: A system for persistent resource type ownership, allowing configuration to be shared across the engine without redundant allocations.
- **Type-Safe Resource Keys**: Compile-time validation of resource access via `ResourceRef` constraints.
- **Bindless Architecture**: Native support for bindless textures and buffers with a high-level `ResourceServices` abstraction.
- **High-Performance Pipeline Caching**: A robust system for resolving and caching Vulkan pipelines based on state requests.
- **Vulkan 1.3+**: Modern Vulkan features like dynamic rendering and push constants are first-class citizens.

## Architecture Overview
Brassica separates the "What" from the "How". 
- **What**: The **Rendering Graph** and **Resource Catalog** define the intent of the scene (which textures are used, what order passes run in).
- **How**: The **Execution Backend** and **NodeContext** translate that intent into low-level Vulkan commands.

## Getting Started
*(Instructions for building and running the engine would go here)*

## Project Structure
- `apps/`: Sample applications and tools.
- `include/`: Core engine headers, organized by system (render, graph, types, etc.).
- `src/`: Core engine implementation.
- `shaders/`: GLSL/SPIR-V shader source files.
- `external/`: Third-party dependencies (Eigen, Vulkan headers, etc.).
- `tools/`: Internal engine tools.

## Development
Brassica uses C++23 and requires a modern compiler (Clang 15+, MSVC 19.35+, or GCC 13+).

For a deep dive into the architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
