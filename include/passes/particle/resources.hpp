#pragma once

// Particle-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). S2 first catalog-converted the two
// indirect-command buffers (genuinely fixed-size, every toucher already agreed on the same
// desc) and ParticleTypeBuffer (compile-time kMaxTypes). The remaining five -- ParticleBuffer,
// both Alive buffers, both grid buffers -- stayed plain tags longer: each toucher's own
// maxParticles/gridSize was an independent per-node-instance field with no shared
// config-publishing owner (unlike TerrainGenNode::numLODs). Now resolved by having each
// buffer's real producer(s) publish byteSize onto the catalog instance directly inside Setup()
// (via ServiceLocator's graph::PhysicalResourceRegistry -- the same idiom these nodes already
// use for IParticleManager, see each producer's own Setup() comment).
//
// The whole cluster is no longer dormant: every node in passes/particle/nodes/ is now
// BRASSICA_REGISTER_NODE'd, gated at runtime by IParticleManager::IsEnabled()
// (ParticleManagerState::enabled, already save/load-capable via ManagerBase's generic
// reflection-driven config) -- the only reason registration was ever commented out was the
// particles being visually distracting during other work, not a missing mechanism.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"
#include "types/Particle.hpp"

namespace brassica {

	namespace detail {
		// ParticleBuffer and both grid buffers share this exact usage -- plain
		// StorageBufferDesc's shape (PhysicalResource.hpp), nothing extra.
		struct ParticleScratchBufferShape {
			static constexpr vk::BufferUsageFlags kUsage =
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
		};

		// Both Alive buffers add eTransferSrc on top of ParticleScratchBufferShape -- folded in
		// here instead of ParticleLivenessNode::Setup's old per-instance
		// aliveDesc.usageMask |= eTransferSrc union.
		struct ParticleAliveBufferShape {
			static constexpr vk::BufferUsageFlags kUsage =
				ParticleScratchBufferShape::kUsage | vk::BufferUsageFlagBits::eTransferSrc;
		};

		// Not a base itself -- both particle indirect-command buffers below share this exact
		// shape (FixedBuffer itself, resources/ResourceTypes.hpp).
		struct ParticleIndirectShape {
			static constexpr std::uint64_t kByteSize = sizeof(ParticleIndirectCommand);
			static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
				vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer |
				vk::BufferUsageFlagBits::eTransferSrc;
		};
	} // namespace detail

	// Producers: ParticleBehaviorNode and ParticleLivenessNode (both Modify<>, each publishing
	// from their own maxParticles field -- whichever runs later in schedule order wins the
	// catalog instance for this frame, same "last writer wins" semantics the old .Desc()
	// escape-hatch collapse already had for this key). See ConfiguredBuffer's own comment
	// (resources/ResourceTypes.hpp) for why Lifetime::Transient + persistent=true, not
	// Lifetime::Persistent.
	struct ParticleBuffer: ConfiguredBuffer<ParticleBuffer>, detail::ParticleScratchBufferShape {};

	// Catalog-converted (S2): kMaxTypes is a compile-time constant, not a config field --
	// sgreenhusted's call: the real particle-type count has no owner that would ever need to
	// change it (confirmed: particle.mesh is the only shader that ever reads uParticleTypes[]),
	// so unlike MaterialBuffer (sized by MaterialManager once model loading exists) this doesn't
	// need a manager-published field. 16 matches ParticleTypeBufferNode's own existing default-
	// constructor seed count exactly. Lifetime::External: its real content comes from a host
	// upload (ParticleTypeBufferNode, a PredefinedBufferNode) with no per-frame regeneration
	// concept -- same category as FoliageBiomeTableTexture/MaterialBuffer.
	struct ParticleTypeBuffer: ExternalBuffer<ParticleTypeBuffer> {
		static constexpr std::uint32_t kMaxTypes = 16;
		static constexpr std::uint64_t kByteSize = kMaxTypes * sizeof(ParticleType);
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

	// Sole producer: ParticleLivenessNode (its own maxParticles field, the sole Create<> of both
	// Alive buffers). See ParticleBuffer's identical Lifetime::Transient + persistent=true
	// reasoning above.
	struct AboveWaterParticleAliveBuffer: ConfiguredBuffer<AboveWaterParticleAliveBuffer>,
										   detail::ParticleAliveBufferShape {};

	// Sole producer: ParticleLivenessNode. See AboveWaterParticleAliveBuffer's identical
	// reasoning.
	struct UnderwaterParticleAliveBuffer: ConfiguredBuffer<UnderwaterParticleAliveBuffer>,
										   detail::ParticleAliveBufferShape {};

	// Catalog-converted (S2): one ParticleIndirectCommand, same desc every one of its 5 touchers
	// (Reset/Liveness/Behavior/both render nodes) used to hand-build identically.
	struct UnderwaterParticleIndirectBuffer: FixedBuffer<UnderwaterParticleIndirectBuffer>,
											  detail::ParticleIndirectShape {};

	// Catalog-converted (S2): see UnderwaterParticleIndirectBuffer's identical reasoning.
	struct AboveWaterParticleIndirectBuffer: FixedBuffer<AboveWaterParticleIndirectBuffer>,
											  detail::ParticleIndirectShape {};

	// Producers: ParticleResetNode (Create<>, sole owner the frame the cluster (re-)enables) and
	// ParticleGridBuildNode (Modify<> every active frame after that) -- both publish from their
	// own gridSize field. See ParticleBuffer's identical reasoning above.
	struct ParticleGridHeadsBuffer: ConfiguredBuffer<ParticleGridHeadsBuffer>,
									 detail::ParticleScratchBufferShape {};

	// Sole producer: ParticleGridBuildNode (its own maxParticles field, the sole Create<>). See
	// ParticleBuffer's identical reasoning above.
	struct ParticleGridNextBuffer: ConfiguredBuffer<ParticleGridNextBuffer>,
									detail::ParticleScratchBufferShape {};

	using ParticleIndirectBuffer = AboveWaterParticleIndirectBuffer;

} // namespace brassica
