#include "passes/model/DecorAccelerationStructure.hpp"

#include <algorithm>
#include <cstring>

#include "graph/Graph.hpp"
#include "graph/ResourceCatalog.hpp"
#include "passes/model/GltfTypes.hpp"
#include "passes/model/ModelRegistry.hpp"

namespace brassica {

	void* DecorAccelerationStructure::CreateBuffer(
		vk::DeviceSize size, vk::BufferUsageFlags usage, BufferResource& res, bool hostMapped
	) {
		VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufInfo.size = size;
		bufInfo.usage = static_cast<VkBufferUsageFlags>(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
		if (m_concurrentFamilies.size() > 1) {
			bufInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			bufInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(m_concurrentFamilies.size());
			bufInfo.pQueueFamilyIndices = m_concurrentFamilies.data();
		}

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		if (hostMapped) {
			allocInfo.flags =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		}

		VkBuffer          vkBuf = VK_NULL_HANDLE;
		VmaAllocationInfo allocResInfo{};
		if (vmaCreateBuffer(m_allocator, &bufInfo, &allocInfo, &vkBuf, &res.allocation, &allocResInfo) !=
		    VK_SUCCESS) {
			return nullptr;
		}
		res.buffer = vkBuf;
		vk::BufferDeviceAddressInfo addrInfo{};
		addrInfo.setBuffer(res.buffer);
		res.deviceAddress = m_device.getBufferAddress(addrInfo);
		res.mapped = allocResInfo.pMappedData;
		return res.mapped;
	}

	void DecorAccelerationStructure::DestroyBuffer(BufferResource& res) {
		if (res.buffer && res.allocation) {
			vmaDestroyBuffer(m_allocator, res.buffer, res.allocation);
		}
		res = BufferResource{};
	}

	void DecorAccelerationStructure::DestroyTemplates() {
		for (TemplateEntry& entry : m_templates) {
			if (entry.blas) {
				m_device.destroyAccelerationStructureKHR(entry.blas, nullptr, *m_dls);
			}
			DestroyBuffer(entry.blasBuffer);
		}
		m_templates.clear();
	}

	void DecorAccelerationStructure::Destroy() {
		if (!m_allocator) {
			return;
		}
		DestroyTemplates();
		DestroyBuffer(m_vertexBuffer);
		DestroyBuffer(m_flatIndexBuffer);
		DestroyBuffer(m_scratchBuffer);

		if (m_tlas) {
			m_device.destroyAccelerationStructureKHR(m_tlas, nullptr, *m_dls);
			m_tlas = nullptr;
		}
		DestroyBuffer(m_tlasBuffer);
		DestroyBuffer(m_tlasScratchBuffer);
	}

	// kDecorCapacity instances, eInstances geometry -- sizing needs no real data pointers (per
	// spec, getAccelerationStructureBuildSizesKHR ignores every device-address member of
	// pGeometries, only geometryCount/type/mode/flags and the primitive count matter), so this
	// can run once, eagerly, from Init() with no dependency on ModelRegistry content at all.
	void DecorAccelerationStructure::InitTLAS() {
		vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
		instancesData.setArrayOfPointers(VK_FALSE);

		vk::AccelerationStructureGeometryDataKHR geomData{};
		geomData.setInstances(instancesData);

		vk::AccelerationStructureGeometryKHR geometry{};
		geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
		geometry.setGeometry(geomData);

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
		buildInfo.setGeometries(geometry);

		std::uint32_t                              instanceCount = kDecorCapacity;
		vk::AccelerationStructureBuildSizesInfoKHR sizeInfo{};
		m_device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice, &buildInfo, &instanceCount, &sizeInfo, *m_dls
		);

		CreateBuffer(
			sizeInfo.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, m_tlasBuffer, false
		);

		vk::AccelerationStructureCreateInfoKHR createInfo{};
		createInfo.setBuffer(m_tlasBuffer.buffer);
		createInfo.setSize(sizeInfo.accelerationStructureSize);
		createInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		m_tlas = m_device.createAccelerationStructureKHR(createInfo, nullptr, *m_dls);

		CreateBuffer(sizeInfo.buildScratchSize, vk::BufferUsageFlagBits::eStorageBuffer, m_tlasScratchBuffer, false);
	}

	void DecorAccelerationStructure::RegisterBufferNode(
		graph::Graph& frameGraph, graph::ResourceCatalog& catalog, const ModelRegistry& registry
	) {
		// Sized off registry.ModelCount() directly, not off whatever RebuildIfNeeded last wrote --
		// models are registered synchronously at Engine::Init (before frame 1's graph even builds),
		// but RebuildIfNeeded only ever runs from DecorTemplateBlasNode::Execute(), one full frame
		// after this call. Deriving byteSize from m_blasAddressesForGpu.size() directly left it at
		// 0 on frame 1 (default-constructed, empty) -- a real, confirmed-on-real-hardware
		// vkUpdateDescriptorSets validation error ("range is not VK_WHOLE_SIZE and is zero"),
		// same class of bug ModelInstanceWriteBuffer's own fixed-capacity byteSize (not derived
		// from this frame's write count -- ModelInstanceManager.cpp's own comment) already avoids.
		// New entries default to (0,0): the shader already treats that as "template not built
		// yet," not an error (model_decor_tlas_instances.comp's own comment).
		std::uint32_t modelCount = registry.ModelCount();
		if (m_blasAddressesForGpu.size() != modelCount) {
			m_blasAddressesForGpu.resize(modelCount, glm::uvec2(0, 0));
			m_blasAddressBufNode.SetData(m_blasAddressesForGpu);
		}
		catalog.Get<DecorTemplateBlasAddressBuffer>().byteSize = m_blasAddressesForGpu.size() * sizeof(glm::uvec2);
		frameGraph.RegisterRef(m_blasAddressBufNode);
	}

	void DecorAccelerationStructure::BuildTLAS(vk::CommandBuffer cmd, vk::DeviceAddress instanceBufferAddress) {
		if (!m_tlas) {
			return;
		}

		vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
		instancesData.setArrayOfPointers(VK_FALSE);
		instancesData.setData(instanceBufferAddress);

		vk::AccelerationStructureGeometryDataKHR geomData{};
		geomData.setInstances(instancesData);

		vk::AccelerationStructureGeometryKHR geometry{};
		geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
		geometry.setGeometry(geomData);

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
		buildInfo.setGeometries(geometry);
		buildInfo.setDstAccelerationStructure(m_tlas);
		buildInfo.setScratchData(m_tlasScratchBuffer.deviceAddress);

		vk::AccelerationStructureBuildRangeInfoKHR range{};
		range.setPrimitiveCount(kDecorCapacity);
		range.setPrimitiveOffset(0);
		range.setFirstVertex(0);
		range.setTransformOffset(0);

		const vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
		cmd.buildAccelerationStructuresKHR(1, &buildInfo, &pRange, *m_dls);

		vk::MemoryBarrier barrier{
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR
		};
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::DependencyFlags{},
			barrier,
			nullptr,
			nullptr
		);
	}

	void DecorAccelerationStructure::RebuildIfNeeded(vk::CommandBuffer cmd, const ModelRegistry& registry) {
		if (!m_allocator || !m_dls || registry.GetVersion() == m_lastSeenVersion) {
			return;
		}
		m_lastSeenVersion = registry.GetVersion();

		DestroyTemplates();
		DestroyBuffer(m_vertexBuffer);
		DestroyBuffer(m_flatIndexBuffer);
		DestroyBuffer(m_scratchBuffer);

		std::span<const std::uint32_t> flatIndices = registry.GetFlatTriangleIndices();
		std::span<const GltfVertex>    vertices = registry.GetVertices();
		std::uint32_t                  modelCount = registry.ModelCount();
		m_blasAddressesForGpu.assign(modelCount, glm::uvec2(0, 0));
		if (flatIndices.empty() || vertices.empty() || modelCount == 0) {
			m_blasAddressBufNode.SetData(m_blasAddressesForGpu);
			return;
		}

		// A private copy, not the shared catalog GltfVertexBuffer -- see GetVertices()'s own
		// comment (ModelRegistry.hpp) for why this must not be read via that buffer's device
		// address.
		void* vertexMapped = CreateBuffer(
			vertices.size() * sizeof(GltfVertex),
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			m_vertexBuffer,
			true
		);
		if (vertexMapped) {
			std::memcpy(vertexMapped, vertices.data(), vertices.size_bytes());
		}

		void* mapped = CreateBuffer(
			flatIndices.size() * sizeof(std::uint32_t),
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			m_flatIndexBuffer,
			true
		);
		if (mapped) {
			std::memcpy(mapped, flatIndices.data(), flatIndices.size_bytes());
		}

		m_templates.assign(modelCount, TemplateEntry{});

		// Pass 1: query build sizes and create every template's BLAS object (cheap, no scratch
		// needed yet) -- deferred until this loop's own max is known, same "share one scratch
		// buffer across sequential builds" shape TerrainAccelerationStructure's BLAS-then-TLAS
		// build used, just generalized to N independent BLASes instead of one.
		struct PendingBuild {
			std::uint32_t                              modelIndex{0};
			vk::AccelerationStructureGeometryKHR       geometry{};
			vk::AccelerationStructureBuildRangeInfoKHR range{};
			std::uint32_t                              triangleCount{0};
		};
		std::vector<PendingBuild> pending;
		vk::DeviceSize            maxScratch = 0;

		for (std::uint32_t m = 0; m < modelCount; ++m) {
			ModelFlatIndexRange range = registry.GetFlatIndexRange(m);
			std::uint32_t       vertexBase = registry.GetVertexBase(m);
			std::uint32_t       vertexCount = registry.GetVertexCount(m);
			if (range.indexCount < 3 || vertexCount == 0) {
				continue;
			}
			std::uint32_t triangleCount = range.indexCount / 3u;

			vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
			triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
			triangles.setVertexData(
				m_vertexBuffer.deviceAddress + static_cast<vk::DeviceSize>(vertexBase) * sizeof(GltfVertex)
			);
			triangles.setVertexStride(sizeof(GltfVertex));
			triangles.setMaxVertex(vertexCount - 1);
			triangles.setIndexType(vk::IndexType::eUint32);
			triangles.setIndexData(
				m_flatIndexBuffer.deviceAddress + static_cast<vk::DeviceSize>(range.indexBase) * sizeof(std::uint32_t)
			);

			vk::AccelerationStructureGeometryDataKHR geomData{};
			geomData.setTriangles(triangles);

			vk::AccelerationStructureGeometryKHR geometry{};
			geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
			geometry.setGeometry(geomData);
			geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

			vk::AccelerationStructureBuildGeometryInfoKHR sizeQueryInfo{};
			sizeQueryInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
			sizeQueryInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
			sizeQueryInfo.setGeometries(geometry);

			vk::AccelerationStructureBuildSizesInfoKHR sizeInfo{};
			m_device.getAccelerationStructureBuildSizesKHR(
				vk::AccelerationStructureBuildTypeKHR::eDevice, &sizeQueryInfo, &triangleCount, &sizeInfo, *m_dls
			);
			maxScratch = std::max(maxScratch, sizeInfo.buildScratchSize);

			TemplateEntry& entry = m_templates[m];
			CreateBuffer(
				sizeInfo.accelerationStructureSize,
				vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
				entry.blasBuffer,
				false
			);

			vk::AccelerationStructureCreateInfoKHR createInfo{};
			createInfo.setBuffer(entry.blasBuffer.buffer);
			createInfo.setSize(sizeInfo.accelerationStructureSize);
			createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
			entry.blas = m_device.createAccelerationStructureKHR(createInfo, nullptr, *m_dls);

			vk::AccelerationStructureBuildRangeInfoKHR range_{};
			range_.setPrimitiveCount(triangleCount);
			range_.setPrimitiveOffset(0);
			range_.setFirstVertex(0);
			range_.setTransformOffset(0);

			pending.push_back(PendingBuild{
				.modelIndex = m,
				.geometry = geometry,
				.range = range_,
				.triangleCount = triangleCount,
			});
		}

		if (pending.empty()) {
			m_blasAddressBufNode.SetData(m_blasAddressesForGpu);
			return;
		}

		CreateBuffer(maxScratch, vk::BufferUsageFlagBits::eStorageBuffer, m_scratchBuffer, false);

		// Pass 2: record every build, one at a time against the shared scratch buffer -- a
		// memory barrier between each (not just at the end) because every build both reads and
		// writes the same scratch region, same hazard TerrainAccelerationStructure's own
		// BLAS-then-TLAS sequential build on one shared scratch buffer guarded against.
		for (const PendingBuild& build : pending) {
			TemplateEntry& entry = m_templates[build.modelIndex];

			vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
			buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
			buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
			buildInfo.setGeometries(build.geometry);
			buildInfo.setDstAccelerationStructure(entry.blas);
			buildInfo.setScratchData(m_scratchBuffer.deviceAddress);

			const vk::AccelerationStructureBuildRangeInfoKHR* pRange = &build.range;
			cmd.buildAccelerationStructuresKHR(1, &buildInfo, &pRange, *m_dls);

			vk::MemoryBarrier barrier{
				vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureWriteKHR
			};
			cmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				vk::DependencyFlags{},
				barrier,
				nullptr,
				nullptr
			);

			vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{};
			addrInfo.setAccelerationStructure(entry.blas);
			entry.blasAddress = m_device.getAccelerationStructureAddressKHR(addrInfo, *m_dls);
			m_blasAddressesForGpu[build.modelIndex] = glm::uvec2(
				static_cast<std::uint32_t>(entry.blasAddress & 0xFFFFFFFFu),
				static_cast<std::uint32_t>(entry.blasAddress >> 32u)
			);
		}

		m_blasAddressBufNode.SetData(m_blasAddressesForGpu);

		// One more pass->read barrier for whatever reads these addresses next this frame (the
		// decor TLAS instance-authoring pass, once it exists) -- builds already barrier against
		// each other above, but not yet against a later consumer outside this loop.
		vk::MemoryBarrier finalBarrier{
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR
		};
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::DependencyFlags{},
			finalBarrier,
			nullptr,
			nullptr
		);
	}

} // namespace brassica
