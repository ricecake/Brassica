#pragma once

// Model-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure).

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	// Stage 1 of the GLTF/GLB pipeline (GltfModel/GltfModelManager): one asset's vertex/meshlet
	// data, host-uploaded once at load time via graph::PredefinedBufferNode. ConfiguredExternalBuffer,
	// same reasoning as MaterialBuffer (passes/material/resources.hpp): byte size depends on which
	// asset was loaded, so there's no type-level kByteSize constant -- GltfModelManager's
	// PredefinedBufferNode members publish the real (data-size-derived) byteSize onto each key's
	// catalog instance directly. eShaderDeviceAddress on top of ExternalBuffer's usual
	// eStorageBuffer|eTransferDst: read via buffer_reference (device address in a push constant,
	// not a bindless SSBO index -- no generic bindless-buffer GLSL array exists in this codebase
	// yet, see shaders/bindless.glsl), same as the foliage archetype buffers above.
	struct GltfVertexBuffer: ConfiguredExternalBuffer<GltfVertexBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	};
	struct GltfMeshletVertexBuffer: ConfiguredExternalBuffer<GltfMeshletVertexBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};
	struct GltfMeshletTriangleBuffer: ConfiguredExternalBuffer<GltfMeshletTriangleBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};
	struct GltfMeshletBuffer: ConfiguredExternalBuffer<GltfMeshletBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// Stage 4 (GPU skinning): one mat4 per joint, in skin-joint order (GltfModel::EvaluatePose's
	// own output order), recomputed and re-uploaded every frame as the current animation plays --
	// unlike the other four GLTF buffers above, this one's *contents* change every frame even
	// though its size doesn't, same PredefinedBufferNode mechanism either way. Same
	// ConfiguredExternalBuffer reasoning as the others: joint count depends on which asset's skin
	// was loaded.
	struct GltfJointMatrixBuffer: ConfiguredExternalBuffer<GltfJointMatrixBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

} // namespace brassica
