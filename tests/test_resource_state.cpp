#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/ResourceState.hpp"

using namespace brassica::graph;

// This is the barrier logic's first real coverage: DeriveImageState/DeriveBufferState are pure
// functions with no device dependency, so unlike everything downstream in PhysicalRegistry.hpp/
// PhysicalExecutionBackend.hpp, they run and assert on every machine, GPU or not. The single
// most important property asserted throughout is the Bug 1 regression -- these functions must
// never hand BarrierTranslator a zero-filled state, since that's what turns into an invalid
// newLayout=eUndefined or a memory-visibility-free eNone access mask.

namespace {

	void CheckNeverZero(const ResourceState& s, bool isBuffer = false) {
		if (!isBuffer) {
			CHECK(s.layout != vk::ImageLayout::eUndefined);
		}
		CHECK(s.stage != vk::PipelineStageFlags2{});
		CHECK(s.access != vk::AccessFlags2{});
	}

} // namespace

TEST_CASE("AspectFor and IsDepthFormat handle depth, combined depth-stencil, stencil-only, and color") {
	CHECK(IsDepthFormat(vk::Format::eD32Sfloat));
	CHECK(IsDepthFormat(vk::Format::eD24UnormS8Uint));
	CHECK(IsDepthFormat(vk::Format::eD16UnormS8Uint));
	CHECK(IsDepthFormat(vk::Format::eD32SfloatS8Uint));
	CHECK(IsDepthFormat(vk::Format::eX8D24UnormPack32));
	CHECK_FALSE(IsDepthFormat(vk::Format::eR8G8B8A8Unorm));
	CHECK_FALSE(IsDepthFormat(vk::Format::eS8Uint));

	CHECK(AspectFor(vk::Format::eD32Sfloat) == vk::ImageAspectFlagBits::eDepth);
	CHECK(AspectFor(vk::Format::eR8G8B8A8Unorm) == vk::ImageAspectFlagBits::eColor);
	CHECK(AspectFor(vk::Format::eS8Uint) == vk::ImageAspectFlagBits::eStencil);

	// The bug all three duplicated old sites shared: a combined depth-stencil format must carry
	// both aspect bits, not just eDepth.
	CHECK(
		AspectFor(vk::Format::eD24UnormS8Uint) == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
	);
	CHECK(
		AspectFor(vk::Format::eD16UnormS8Uint) == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
	);
	CHECK(
		AspectFor(vk::Format::eD32SfloatS8Uint) == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
	);
	// No stencil component -- depth-only aspect even though the format has padding stencil bits.
	CHECK(AspectFor(vk::Format::eX8D24UnormPack32) == vk::ImageAspectFlagBits::eDepth);
}

TEST_CASE("DeriveImageState: depth attachment write/read-write") {
	auto write = DeriveImageState(
		AccessKind::Write,
		ExecutionDomain::Graphics,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::Format::eD32Sfloat
	);
	CHECK(write.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal);
	CHECK(
		write.stage ==
		(vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests)
	);
	CHECK(write.access == vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
	CheckNeverZero(write);

	auto readWrite = DeriveImageState(
		AccessKind::ReadWrite,
		ExecutionDomain::Graphics,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::Format::eD32Sfloat
	);
	CHECK(
		readWrite.access ==
		(vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eDepthStencilAttachmentRead)
	);
}

TEST_CASE("DeriveImageState: color attachment write matches real consumers (GradientPass/TerrainPass/DeferredPass)") {
	auto state = DeriveImageState(
		AccessKind::Write,
		ExecutionDomain::Graphics,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::Format::eR16G16B16A16Sfloat
	);
	CHECK(state.layout == vk::ImageLayout::eColorAttachmentOptimal);
	// WSI coupling (see Engine.cpp's swapchainSemaphore wait at eColorAttachmentOutput): a color
	// write must derive exactly this stage, not something broader like eAllCommands, or the
	// first-use transition could race ahead of swapchain acquire.
	CHECK(state.stage == vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	CHECK(state.access == vk::AccessFlagBits2::eColorAttachmentWrite);
	CheckNeverZero(state);
}

TEST_CASE("DeriveImageState: compute storage write/read (AtmosphereLUT transmittance/multiscattering shape)") {
	auto write = DeriveImageState(
		AccessKind::Write,
		ExecutionDomain::Compute,
		vk::ImageUsageFlagBits::eStorage,
		vk::Format::eR32G32B32A32Sfloat
	);
	CHECK(write.layout == vk::ImageLayout::eGeneral);
	CHECK(write.stage == vk::PipelineStageFlagBits2::eComputeShader);
	CHECK(write.access == vk::AccessFlagBits2::eShaderStorageWrite);

	auto read = DeriveImageState(
		AccessKind::Read,
		ExecutionDomain::Compute,
		vk::ImageUsageFlagBits::eStorage,
		vk::Format::eR32G32B32A32Sfloat
	);
	CHECK(read.layout == vk::ImageLayout::eGeneral);
	CheckNeverZero(read);
}

TEST_CASE("DeriveImageState: sampled read uses eShaderReadOnlyOptimal, storage|sampled read stays eGeneral") {
	auto sampledOnly = DeriveImageState(
		AccessKind::Read,
		ExecutionDomain::Graphics,
		vk::ImageUsageFlagBits::eSampled,
		vk::Format::eR16G16B16A16Sfloat
	);
	CHECK(sampledOnly.layout == vk::ImageLayout::eShaderReadOnlyOptimal);
	CHECK(sampledOnly.access == vk::AccessFlagBits2::eShaderSampledRead);

	// A resource that's both Storage and Sampled (ComputeStorageImageDesc) must resolve to
	// eGeneral on read, never eShaderReadOnlyOptimal -- a storage-image descriptor requires it,
	// and a sampler can legally read from eGeneral, but not the reverse.
	auto storageAndSampled = DeriveImageState(
		AccessKind::Read,
		ExecutionDomain::Compute,
		vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
		vk::Format::eR32G32B32A32Sfloat
	);
	CHECK(storageAndSampled.layout == vk::ImageLayout::eGeneral);
	CheckNeverZero(storageAndSampled);
}

TEST_CASE(
	"DeriveImageState: never returns eUndefined or a zero access mask across the full access x domain x usage sweep"
) {
	const vk::ImageUsageFlags usages[] = {
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageUsageFlagBits::eStorage,
		vk::ImageUsageFlagBits::eSampled,
		vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
		vk::ImageUsageFlagBits::eTransferDst, // deliberately unmodeled -- exercises the fallback
	};
	const AccessKind      accesses[] = {AccessKind::Read, AccessKind::Write, AccessKind::ReadWrite};
	const ExecutionDomain domains[] = {
		ExecutionDomain::Graphics,
		ExecutionDomain::Compute,
		ExecutionDomain::Transfer,
		ExecutionDomain::Host,
	};
	const vk::Format formats[] = {vk::Format::eR16G16B16A16Sfloat, vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint};

	for (auto usage : usages) {
		for (auto access : accesses) {
			for (auto domain : domains) {
				for (auto format : formats) {
					CheckNeverZero(DeriveImageState(access, domain, usage, format));
				}
			}
		}
	}
}

TEST_CASE("DeriveBufferState: uniform, storage, indirect, and the unmodeled fallback never return zeros") {
	CheckNeverZero(
		DeriveBufferState(AccessKind::Read, ExecutionDomain::Graphics, vk::BufferUsageFlagBits::eUniformBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Write, ExecutionDomain::Graphics, vk::BufferUsageFlagBits::eUniformBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Read, ExecutionDomain::Compute, vk::BufferUsageFlagBits::eStorageBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Write, ExecutionDomain::Compute, vk::BufferUsageFlagBits::eStorageBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Read, ExecutionDomain::Graphics, vk::BufferUsageFlagBits::eIndirectBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Write, ExecutionDomain::Graphics, vk::BufferUsageFlagBits::eIndirectBuffer),
		true
	);
	CheckNeverZero(
		DeriveBufferState(AccessKind::Read, ExecutionDomain::Transfer, vk::BufferUsageFlagBits::eTransferDst),
		true
	);

	auto uniformRead = DeriveBufferState(
		AccessKind::Read,
		ExecutionDomain::Graphics,
		vk::BufferUsageFlagBits::eUniformBuffer
	);
	CHECK(uniformRead.access == vk::AccessFlagBits2::eUniformRead);

	auto indirectRead = DeriveBufferState(
		AccessKind::Read,
		ExecutionDomain::Graphics,
		vk::BufferUsageFlagBits::eIndirectBuffer
	);
	CHECK(indirectRead.stage == vk::PipelineStageFlagBits2::eDrawIndirect);
	CHECK(indirectRead.access == vk::AccessFlagBits2::eIndirectCommandRead);
}

TEST_CASE("IsWrite recognizes every write-side access this header can produce, and no read-side access") {
	CHECK(IsWrite(vk::AccessFlagBits2::eColorAttachmentWrite));
	CHECK(IsWrite(vk::AccessFlagBits2::eDepthStencilAttachmentWrite));
	CHECK(IsWrite(vk::AccessFlagBits2::eShaderStorageWrite));
	CHECK(IsWrite(vk::AccessFlagBits2::eTransferWrite));
	CHECK(IsWrite(vk::AccessFlagBits2::eHostWrite));
	// A ReadWrite access sets both bits; IsWrite must still see it as a write.
	CHECK(
		IsWrite(vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eDepthStencilAttachmentRead)
	);
	// eMemoryWrite is the conservative seed PhysicalResource.hpp uses for a reused aliased block
	// or an imported resource with hasDefinedContents=true (PhysicalResource.hpp's aliased/
	// imported constructors) -- if IsWrite didn't recognize it, the barrier skip-rule could
	// wrongly skip a resource's first real barrier after import or alias reuse.
	CHECK(IsWrite(vk::AccessFlagBits2::eMemoryWrite));

	CHECK_FALSE(IsWrite(vk::AccessFlagBits2::eShaderSampledRead));
	CHECK_FALSE(IsWrite(vk::AccessFlagBits2::eColorAttachmentRead));
	CHECK_FALSE(IsWrite(vk::AccessFlagBits2::eUniformRead));
	CHECK_FALSE(IsWrite(vk::AccessFlags2{}));
}

TEST_CASE("End-to-end swapchain chain: Modify<Swapchain> derives a valid first-use transition") {
	// Mirrors what the migration's DeferredNode will declare: Modify<Swapchain> is a
	// ReadWrite access to a freshly re-imported (eUndefined, never-written) color attachment.
	auto state = DeriveImageState(
		AccessKind::ReadWrite,
		ExecutionDomain::Graphics,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::Format::eB8G8R8A8Unorm
	);
	CHECK(state.layout == vk::ImageLayout::eColorAttachmentOptimal);
	CHECK(state.stage == vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	CHECK(state.access == (vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead));

	// This is exactly what Engine.cpp's existing hardcoded present barrier assumes as oldLayout
	// (src/Engine.cpp:826-831) -- confirming the present barrier stays valid without being
	// touched, once the graph path derives the same layout by the same rule.
	CHECK(state.layout == vk::ImageLayout::eColorAttachmentOptimal);
}

TEST_CASE("DeriveAccelerationStructureState: build-write, ray-query-read, and read-write never return zeros") {
	auto write = DeriveAccelerationStructureState(AccessKind::Write, ExecutionDomain::Graphics);
	CHECK(write.stage == vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR);
	CHECK(write.access == vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
	CheckNeverZero(write, true); // no layout for an AS, same as a buffer

	// Matches how TerrainTLASNode/DeferredNode will declare it: DeferredNode's ray query runs
	// from the fragment shader, so a Graphics-domain Read must resolve to eAllGraphics (which
	// subsumes eFragmentShader), not something narrower.
	auto read = DeriveAccelerationStructureState(AccessKind::Read, ExecutionDomain::Graphics);
	CHECK(read.stage == vk::PipelineStageFlagBits2::eAllGraphics);
	CHECK(read.access == vk::AccessFlagBits2::eAccelerationStructureReadKHR);
	CheckNeverZero(read, true);

	auto readWrite = DeriveAccelerationStructureState(AccessKind::ReadWrite, ExecutionDomain::Compute);
	CHECK(readWrite.access == (vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eAccelerationStructureReadKHR));
	CheckNeverZero(readWrite, true);
}

TEST_CASE("IsWrite recognizes eAccelerationStructureWriteKHR") {
	// Without this, EmitAccelerationStructureBarrier's read-after-read skip rule (BarrierTranslator.hpp)
	// would wrongly skip a resource's first real barrier after an AS build.
	CHECK(IsWrite(vk::AccessFlagBits2::eAccelerationStructureWriteKHR));
	CHECK_FALSE(IsWrite(vk::AccessFlagBits2::eAccelerationStructureReadKHR));
}
