#pragma once

// Material-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). MaterialBuffer catalog-converted
// in S2 once MaterialManager had a place to publish byteSize (RegisterBufferNode, below).

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	// byteSize is genuine runtime config MaterialManager publishes (RegisterBufferNode,
	// MaterialManager.cpp) from m_materials.size() * sizeof(MaterialData) every time it builds
	// the frame graph -- not a type-level constant, since the real material count has no fixed
	// upper bound yet (see ConfiguredExternalBuffer's own comment: this is its reference user).
	// kUsage matches what the old hand-rolled StagedStorageBufferDesc(byteSize) implied
	// (Node.hpp's StagedStorageBufferDesc: eStorageBuffer | eTransferDst).
	struct MaterialBuffer: ConfiguredExternalBuffer<MaterialBuffer> {
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

} // namespace brassica
