#pragma once

#include <vulkan/vulkan.hpp>

#include "graph/Execution.hpp"

// The physical layer's one seam for turning an abstract (AccessKind, ExecutionDomain, usage)
// triple into concrete vk::ImageLayout / vk::PipelineStageFlags2 / vk::AccessFlags2 values.
// Execution.hpp/Graph.hpp/Node.hpp stay Vulkan-free by design and treat formatCode/usageMask as
// opaque (see Execution.hpp's ResourceDesc comment) -- this is where that opacity gets resolved.
// Included directly against <vulkan/vulkan.hpp> rather than VulkanSeam.hpp, since none of this
// needs VMA and it lets tests/test_resource_state.cpp exercise these rules with no GPU present.
//
// Load-bearing assumption, stated up front because it is easy to violate by accident later: the
// resource state PhysicalTexture/PhysicalBuffer track (see PhysicalResource.hpp) is a single
// linear timeline. That is only correct because record order == submission order on one queue
// with one command buffer -- true today (the engine has exactly one graphicsQueue, and
// FRAME_OVERLAP's two frames record identical sequences in order, gated by a timeline
// semaphore). The instant anything is submitted to a second queue or a secondary command
// buffer, this tracking is wrong and needs revisiting.

namespace brassica::graph {

	// -- format helpers ---------------------------------------------------------------
	// Previously duplicated (and subtly wrong for combined depth-stencil formats -- eD24UnormS8Uint
	// mapped to eDepth only) at BarrierTranslator.hpp, PhysicalExecutionBackend.hpp, and
	// PhysicalResource.hpp. One definition here, used by all three.

	inline bool IsDepthFormat(vk::Format format) {
		switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32:
			return true;
		default:
			return false;
		}
	}

	inline bool HasStencilComponent(vk::Format format) {
		switch (format) {
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eS8Uint:
			return true;
		default:
			return false;
		}
	}

	inline vk::ImageAspectFlags AspectFor(vk::Format format) {
		if (format == vk::Format::eS8Uint) {
			return vk::ImageAspectFlagBits::eStencil;
		}
		if (IsDepthFormat(format)) {
			vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eDepth;
			if (HasStencilComponent(format)) {
				aspect |= vk::ImageAspectFlagBits::eStencil;
			}
			return aspect;
		}
		return vk::ImageAspectFlagBits::eColor;
	}

	// -- state derivation ---------------------------------------------------------------

	// What a resource must look like for a given access, before/after a barrier. .layout is
	// meaningless for buffers (DeriveBufferState leaves it default-constructed / eUndefined);
	// callers must not read it in that case.
	struct ResourceState {
		vk::ImageLayout         layout{};
		vk::PipelineStageFlags2 stage{};
		vk::AccessFlags2        access{};
	};

	inline vk::PipelineStageFlags2 ShaderStageForDomain(ExecutionDomain domain) {
		switch (domain) {
		case ExecutionDomain::Graphics:
			// eAllGraphics, not eFragmentShader: this engine dispatches through task/mesh
			// shaders (TerrainPass's drawMeshTasksEXT), so naming the fragment stage
			// specifically would silently miss any Graphics-domain node that touches a
			// resource from a task/mesh/vertex stage instead.
			return vk::PipelineStageFlagBits2::eAllGraphics;
		case ExecutionDomain::Compute:
			return vk::PipelineStageFlagBits2::eComputeShader;
		case ExecutionDomain::Transfer:
			return vk::PipelineStageFlagBits2::eAllTransfer;
		case ExecutionDomain::Host:
			return vk::PipelineStageFlagBits2::eHost;
		}
		// Unreachable -- every ExecutionDomain value is handled above. Kept as a defensive
		// fallback rather than an assert, consistent with this header's rule that nothing ever
		// returns a zero/no-op stage mask (that's Bug 1 this header exists to fix).
		return vk::PipelineStageFlagBits2::eAllCommands;
	}

	// The write-side access bits that matter to this header's callers: everything the three
	// Derive*State functions below can produce, plus eMemoryWrite, which PhysicalResource.hpp
	// uses as the conservative "something wrote here, but I don't know exactly what" seed for
	// an aliased-and-reused block or an imported resource with hasDefinedContents=true. Extend
	// this list if a new write case is added to either place -- it is intentionally not a
	// kitchen-sink of every AccessFlagBits2 write bit Vulkan defines.
	inline bool IsWrite(vk::AccessFlags2 access) {
		constexpr vk::AccessFlags2 kWriteBits = vk::AccessFlagBits2::eColorAttachmentWrite |
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eShaderStorageWrite |
			vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eMemoryWrite |
			vk::AccessFlagBits2::eAccelerationStructureWriteKHR;
		return static_cast<bool>(access & kWriteBits);
	}

	// Derives the (layout, stage, access) a Graphics/Compute/Transfer/Host-domain node needs for
	// a given access to an image with the given usage/format. Exhaustive over every
	// (AccessKind, usage-bit) combination this codebase's consumers actually exercise, with a
	// defensive non-zero fallback for the rest -- a zero return here is exactly Bug 1
	// (BarrierTranslator casting an all-zero MemoryBarrier into an invalid newLayout=eUndefined).
	inline ResourceState
	DeriveImageState(AccessKind access, ExecutionDomain domain, vk::ImageUsageFlags usage, vk::Format format) {
		// Mirrors detail::BuildImageCreateInfo's default (PhysicalResource.hpp) so derivation and
		// creation can never disagree about what an unset usage mask means.
		if (!usage) {
			usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment;
		}

		const bool isWrite = access == AccessKind::Write || access == AccessKind::ReadWrite;
		const bool isReadWrite = access == AccessKind::ReadWrite;

		if (isWrite) {
			if (IsDepthFormat(format) && (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)) {
				vk::AccessFlags2 acc = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
				if (isReadWrite) {
					acc |= vk::AccessFlagBits2::eDepthStencilAttachmentRead;
				}
				return ResourceState{
					vk::ImageLayout::eDepthStencilAttachmentOptimal,
					vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
					acc,
				};
			}

			if ((usage & vk::ImageUsageFlagBits::eColorAttachment) && domain == ExecutionDomain::Graphics) {
				vk::AccessFlags2 acc = vk::AccessFlagBits2::eColorAttachmentWrite;
				if (isReadWrite) {
					acc |= vk::AccessFlagBits2::eColorAttachmentRead;
				}
				return ResourceState{
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					acc,
				};
			}

			if (usage & vk::ImageUsageFlagBits::eStorage) {
				vk::AccessFlags2 acc = vk::AccessFlagBits2::eShaderStorageWrite;
				if (isReadWrite) {
					acc |= vk::AccessFlagBits2::eShaderStorageRead;
				}
				return ResourceState{vk::ImageLayout::eGeneral, ShaderStageForDomain(domain), acc};
			}

			// Defensive fallback: a write usage this table doesn't model yet (e.g. a
			// TransferDst-only target). Never return zeros -- fall back to a broad-but-valid
			// transfer-write shape rather than silently dropping the memory dependency.
			vk::AccessFlags2 acc = vk::AccessFlagBits2::eTransferWrite;
			if (isReadWrite) {
				acc |= vk::AccessFlagBits2::eTransferRead;
			}
			return ResourceState{vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands, acc};
		}

		// Read. Storage is checked before Sampled deliberately: a combined Storage|Sampled
		// resource (e.g. ComputeStorageImageDesc) must use eGeneral, since a storage-image
		// descriptor requires it and a sampler can legally read from eGeneral, but the reverse
		// isn't true. AccessKind can't distinguish "read via imageLoad" from "read via a
		// sampler" for such a resource (a known limitation, see the migration plan's Known
		// Limitations list), so both access bits are set conservatively -- broader access masks
		// are always safe in Vulkan, narrower-than-actual ones are not.
		if (usage & vk::ImageUsageFlagBits::eStorage) {
			return ResourceState{
				vk::ImageLayout::eGeneral,
				ShaderStageForDomain(domain),
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead,
			};
		}

		if (usage & vk::ImageUsageFlagBits::eSampled) {
			return ResourceState{
				vk::ImageLayout::eShaderReadOnlyOptimal,
				ShaderStageForDomain(domain),
				vk::AccessFlagBits2::eShaderSampledRead,
			};
		}

		// Defensive fallback: a read usage this table doesn't model yet (e.g. TransferSrc-only).
		return ResourceState{
			vk::ImageLayout::eGeneral,
			vk::PipelineStageFlagBits2::eAllCommands,
			vk::AccessFlagBits2::eMemoryRead,
		};
	}

	// Buffer counterpart. .layout on the returned ResourceState is meaningless -- buffers have
	// no layout -- and left default-constructed; callers must not read it.
	inline ResourceState DeriveBufferState(AccessKind access, ExecutionDomain domain, vk::BufferUsageFlags usage) {
		const bool isWrite = access == AccessKind::Write || access == AccessKind::ReadWrite;
		const bool isReadWrite = access == AccessKind::ReadWrite;

		if (usage & vk::BufferUsageFlagBits::eUniformBuffer) {
			// Uniform buffers in this engine are host-written, persistently mapped, shader-read
			// (mirrors the old FrameGraphUBO::preRead/preWrite, src/passes/PassResource.cpp) --
			// there is no node-authored write path yet, so Write/ReadWrite just falls back to
			// the host-write shape rather than modeling a case nothing exercises.
			if (isWrite) {
				return ResourceState{{}, vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostWrite};
			}
			return ResourceState{{}, ShaderStageForDomain(domain), vk::AccessFlagBits2::eUniformRead};
		}

		if (usage & vk::BufferUsageFlagBits::eIndirectBuffer) {
			if (isWrite) {
				vk::AccessFlags2 acc = vk::AccessFlagBits2::eShaderStorageWrite;
				if (isReadWrite) {
					acc |= vk::AccessFlagBits2::eShaderStorageRead;
				}
				return ResourceState{{}, ShaderStageForDomain(domain), acc};
			}
			return ResourceState{
				{},
				vk::PipelineStageFlagBits2::eDrawIndirect,
				vk::AccessFlagBits2::eIndirectCommandRead,
			};
		}

		if (usage & vk::BufferUsageFlagBits::eStorageBuffer) {
			if (isWrite) {
				vk::AccessFlags2 acc = vk::AccessFlagBits2::eShaderStorageWrite;
				if (isReadWrite) {
					acc |= vk::AccessFlagBits2::eShaderStorageRead;
				}
				return ResourceState{{}, ShaderStageForDomain(domain), acc};
			}
			return ResourceState{{}, ShaderStageForDomain(domain), vk::AccessFlagBits2::eShaderStorageRead};
		}

		// Defensive fallback: a usage this table doesn't model yet. Never return zeros.
		if (isWrite) {
			vk::AccessFlags2 acc = vk::AccessFlagBits2::eTransferWrite;
			if (isReadWrite) {
				acc |= vk::AccessFlagBits2::eTransferRead;
			}
			return ResourceState{{}, vk::PipelineStageFlagBits2::eAllCommands, acc};
		}
		return ResourceState{{}, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eTransferRead};
	}

	// Acceleration-structure counterpart. .layout is meaningless here too (an AS has no
	// layout), same as DeriveBufferState. No usage flags to branch on -- an AS is only ever
	// built (Write, by the AS-build stage) or read (via ray query, from whichever shader stage
	// the consuming node's domain implies). See PhysicalAccelerationStructure's class comment
	// (PhysicalResource.hpp) for why this is the only Derive*State function with no usage
	// parameter at all.
	inline ResourceState DeriveAccelerationStructureState(AccessKind access, ExecutionDomain domain) {
		switch (access) {
		case AccessKind::Write:
			return ResourceState{
				{},
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
				vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
			};
		case AccessKind::Read:
			return ResourceState{
				{},
				ShaderStageForDomain(domain),
				vk::AccessFlagBits2::eAccelerationStructureReadKHR,
			};
		case AccessKind::ReadWrite:
			return ResourceState{
				{},
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | ShaderStageForDomain(domain),
				vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
					vk::AccessFlagBits2::eAccelerationStructureReadKHR,
			};
		}
		// Unreachable -- every AccessKind value is handled above. Defensive fallback, never zero.
		return ResourceState{
			{},
			vk::PipelineStageFlagBits2::eAllCommands,
			vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		};
	}

} // namespace brassica::graph
