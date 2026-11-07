#pragma once

// Entity-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"
#include "types/EntityRenderComponent.hpp"

namespace brassica {

	// Fixed capacity (1024 instances). eShaderDeviceAddress is on top of FixedBuffer's usual
	// eStorageBuffer|eTransferDst -- EntityNode reads this buffer by BDA, not a bindless index
	// (types/EntityRenderComponent.hpp's EntityPushConstants).
	struct EntityInstanceBuffer: FixedBuffer<EntityInstanceBuffer> {
		static constexpr std::uint64_t kByteSize = sizeof(EntityInstanceData) * 1024;
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::Mapped;
	};

	// One MeshTasksIndirectCommand.
	struct EntityIndirectBuffer: FixedBuffer<EntityIndirectBuffer> {
		static constexpr std::uint64_t kByteSize = sizeof(MeshTasksIndirectCommand);
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer;
		static constexpr graph::HostAccess kHostAccess = graph::HostAccess::Mapped;
	};
} // namespace brassica
