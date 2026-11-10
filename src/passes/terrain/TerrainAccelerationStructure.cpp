#include "passes/terrain/TerrainAccelerationStructure.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "constants.h"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

namespace brassica {

	struct TerrainAABBPushConstants {
		glm::uvec2 aabbBufferAddr;
		glm::uvec2 padding{0, 0};
		glm::uvec4 gridParams;
	};

	void TerrainAccelerationStructure::DestroyAccelerationStructures() {
		if (!allocator)
			return;

		auto destroyAS = [this](vk::AccelerationStructureKHR& as, BufferResource& buf) {
			if (as) {
				device.destroyAccelerationStructureKHR(as, nullptr, dls);
				as = nullptr;
			}
			if (buf.buffer && buf.allocation) {
				vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
				buf.buffer = nullptr;
				buf.allocation = VK_NULL_HANDLE;
				buf.deviceAddress = 0;
			}
		};

		auto destroyBuf = [this](BufferResource& buf) {
			if (buf.buffer && buf.allocation) {
				vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
				buf.buffer = nullptr;
				buf.allocation = VK_NULL_HANDLE;
				buf.deviceAddress = 0;
			}
		};

		destroyAS(tlas, tlasBuffer);
		destroyAS(blas, blasBuffer);
		for (auto& entry : m_chunkBlasEntries) {
			destroyAS(entry.blas, entry.storage);
			entry.built = false;
		}
		m_chunkBlasEntries.clear();
		destroyBuf(aabbBuffer);
		destroyBuf(chunkIndexBuffer);
		destroyBuf(instanceBuffer);
		destroyBuf(scratchBuffer);
	}

	void TerrainAccelerationStructure::InitAccelerationStructures() {
		if (!allocator || blas != VK_NULL_HANDLE)
			return;

		auto createBuffer =
			[this](vk::DeviceSize size, vk::BufferUsageFlags usage, BufferResource& res, bool hostMapped = false) {
				if (res.buffer != VK_NULL_HANDLE) {
					return static_cast<void*>(nullptr);
				}
				VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
				bufInfo.size = size;
				bufInfo.usage = static_cast<VkBufferUsageFlags>(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
				// aabbBuffer/instanceBuffer/scratchBuffer/blasBuffer/tlasBuffer all go through
			    // here -- TerrainTLAS and its backing buffer are out-of-registry (RegisterImported-
			    // AccelerationStructure, not Provision*), so this is the one place to apply the
			    // design plan's S4 CONCURRENT-vs-EXCLUSIVE call for them.
				if (m_concurrentFamilies.size() > 1) {
					bufInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
					bufInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(m_concurrentFamilies.size());
					bufInfo.pQueueFamilyIndices = m_concurrentFamilies.data();
				}

				VmaAllocationCreateInfo allocInfo{};
				allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
				if (hostMapped) {
					allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						VMA_ALLOCATION_CREATE_MAPPED_BIT;
				}

				VkBuffer          vkBuf = VK_NULL_HANDLE;
				VmaAllocationInfo allocResInfo{};
				if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &vkBuf, &res.allocation, &allocResInfo) ==
			        VK_SUCCESS) {
					res.buffer = vkBuf;
					vk::BufferDeviceAddressInfo addrInfo{};
					addrInfo.setBuffer(res.buffer);
					res.deviceAddress = device.getBufferAddress(addrInfo);
					return allocResInfo.pMappedData;
				}
				return static_cast<void*>(nullptr);
			};

		uint32_t maxAABBs = constants::Class::Terrain::DefaultMaxLODs * constants::Class::Terrain::MeshletCountPerLOD;
		vk::DeviceSize aabbBufferSize = sizeof(std::uint64_t) + sizeof(VkAabbPositionsKHR) * maxAABBs;

		createBuffer(
			aabbBufferSize,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
				vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			aabbBuffer,
			false
		);

		vk::AccelerationStructureGeometryAabbsDataKHR aabbGeomData{};
		aabbGeomData.setData(aabbBuffer.deviceAddress + sizeof(std::uint64_t));
		aabbGeomData.setStride(sizeof(VkAabbPositionsKHR));

		vk::AccelerationStructureGeometryDataKHR geomData{};
		geomData.setAabbs(aabbGeomData);

		vk::AccelerationStructureGeometryKHR geometry{};
		geometry.setGeometryType(vk::GeometryTypeKHR::eAabbs);
		geometry.setGeometry(geomData);
		geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

		vk::AccelerationStructureBuildGeometryInfoKHR blasBuildInfo{};
		blasBuildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
		blasBuildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
		blasBuildInfo.setGeometries(geometry);

		uint32_t                                   primitiveCount = maxAABBs;
		vk::AccelerationStructureBuildSizesInfoKHR blasSizeInfo{};
		device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,
			&blasBuildInfo,
			&primitiveCount,
			&blasSizeInfo,
			dls
		);

		createBuffer(
			blasSizeInfo.accelerationStructureSize,
			vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
			blasBuffer,
			false
		);

		vk::AccelerationStructureCreateInfoKHR blasCreateInfo{};
		blasCreateInfo.setBuffer(blasBuffer.buffer);
		blasCreateInfo.setSize(blasSizeInfo.accelerationStructureSize);
		blasCreateInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
		blas = device.createAccelerationStructureKHR(blasCreateInfo, nullptr, dls);

		vk::AccelerationStructureDeviceAddressInfoKHR blasAddrInfo{};
		blasAddrInfo.setAccelerationStructure(blas);
		vk::DeviceAddress blasAddress = device.getAccelerationStructureAddressKHR(blasAddrInfo, dls);

		VkAccelerationStructureInstanceKHR instanceData{};
		instanceData.transform.matrix[0][0] = 1.0f;
		instanceData.transform.matrix[1][1] = 1.0f;
		instanceData.transform.matrix[2][2] = 1.0f;
		instanceData.instanceCustomIndex = 0;
		instanceData.mask = 0xFF;
		instanceData.instanceShaderBindingTableRecordOffset = 0;
		instanceData.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instanceData.accelerationStructureReference = blasAddress;

		// Size instanceBuffer for up to MaxAllocatedChunks
		uint32_t maxChunks = constants::Class::Terrain::ChunkManager::MaxAllocatedChunks;
		void* instMapped = createBuffer(
			sizeof(VkAccelerationStructureInstanceKHR) * maxChunks,
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			instanceBuffer,
			true
		);

		// Chunk BLAS sizing: 1 chunk has 256 vertices and 450 triangles (1350 indices)
		constexpr uint32_t vertsPerChunk = constants::Class::Terrain::ChunkManager::VerticesPerChunk;
		constexpr uint32_t chunkVertexStride = sizeof(glm::vec4) * 3; // 48 bytes (TerrainChunkVertexGPU)
		constexpr uint32_t chunkPrimCount = constants::Class::Terrain::ChunkManager::TrianglesPerChunk; // 450

		vk::AccelerationStructureGeometryTrianglesDataKHR chunkTriangles{};
		chunkTriangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
		chunkTriangles.setVertexData(vk::DeviceAddress(0)); // dummy address for sizing
		chunkTriangles.setVertexStride(chunkVertexStride);
		chunkTriangles.setMaxVertex(vertsPerChunk);
		chunkTriangles.setIndexType(vk::IndexType::eUint32);
		chunkTriangles.setIndexData(vk::DeviceAddress(0)); // dummy address for sizing

		vk::AccelerationStructureGeometryDataKHR chunkGeomData{};
		chunkGeomData.setTriangles(chunkTriangles);

		vk::AccelerationStructureGeometryKHR chunkGeom{};
		chunkGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
		chunkGeom.setGeometry(chunkGeomData);
		chunkGeom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

		vk::AccelerationStructureBuildGeometryInfoKHR chunkBlasBuildInfo{};
		chunkBlasBuildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
		chunkBlasBuildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
		chunkBlasBuildInfo.setGeometries(chunkGeom);

		vk::AccelerationStructureBuildSizesInfoKHR chunkBlasSizeInfo{};
		device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,
			&chunkBlasBuildInfo,
			&chunkPrimCount,
			&chunkBlasSizeInfo,
			dls
		);
		m_chunkBlasSize = chunkBlasSizeInfo.accelerationStructureSize;
		m_chunkScratchSize = chunkBlasSizeInfo.buildScratchSize;

		// TLAS sizing for max chunks
		vk::AccelerationStructureGeometryInstancesDataKHR maxTlasInstancesData{};
		maxTlasInstancesData.setArrayOfPointers(VK_FALSE);
		maxTlasInstancesData.setData(instanceBuffer.deviceAddress);

		vk::AccelerationStructureGeometryDataKHR maxTlasGeomData{};
		maxTlasGeomData.setInstances(maxTlasInstancesData);

		vk::AccelerationStructureGeometryKHR maxTlasGeom{};
		maxTlasGeom.setGeometryType(vk::GeometryTypeKHR::eInstances);
		maxTlasGeom.setGeometry(maxTlasGeomData);

		vk::AccelerationStructureBuildGeometryInfoKHR maxTlasBuildInfo{};
		maxTlasBuildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		maxTlasBuildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
		maxTlasBuildInfo.setGeometries(maxTlasGeom);

		vk::AccelerationStructureBuildSizesInfoKHR maxTlasSizeInfo{};
		device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,
			&maxTlasBuildInfo,
			&maxChunks,
			&maxTlasSizeInfo,
			dls
		);

		createBuffer(
			maxTlasSizeInfo.accelerationStructureSize,
			vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
			tlasBuffer,
			false
		);

		vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{};
		tlasCreateInfo.setBuffer(tlasBuffer.buffer);
		tlasCreateInfo.setSize(maxTlasSizeInfo.accelerationStructureSize);
		tlasCreateInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		tlas = device.createAccelerationStructureKHR(tlasCreateInfo, nullptr, dls);

		vk::DeviceSize scratchSize = std::max({blasSizeInfo.buildScratchSize, m_chunkScratchSize, maxTlasSizeInfo.buildScratchSize});
		createBuffer(scratchSize, vk::BufferUsageFlagBits::eStorageBuffer, scratchBuffer, false);

		// Pre-size chunk BLAS entries vector
		m_chunkBlasEntries.resize(constants::Class::Terrain::ChunkManager::MaxAllocatedChunks);

		// Shared chunk index buffer for 15x15 quads (1350 uint32 indices)
		constexpr uint32_t quadsPerRow = constants::Class::Terrain::ChunkManager::ChunkDimension - 1; // 15
		constexpr uint32_t indicesCount = constants::Class::Terrain::ChunkManager::IndicesPerChunk; // 1350
		std::vector<uint32_t> indices;
		indices.reserve(indicesCount);
		for (uint32_t r = 0; r < quadsPerRow; ++r) {
			for (uint32_t c = 0; c < quadsPerRow; ++c) {
				uint32_t v0 = r * constants::Class::Terrain::ChunkManager::ChunkDimension + c;
				uint32_t v1 = r * constants::Class::Terrain::ChunkManager::ChunkDimension + (c + 1);
				uint32_t v2 = (r + 1) * constants::Class::Terrain::ChunkManager::ChunkDimension + c;
				uint32_t v3 = (r + 1) * constants::Class::Terrain::ChunkManager::ChunkDimension + (c + 1);

				// Matching orientation used in terrain_chunk.mesh
				indices.push_back(v0);
				indices.push_back(v2);
				indices.push_back(v1);

				indices.push_back(v1);
				indices.push_back(v2);
				indices.push_back(v3);
			}
		}

		void* idxMapped = createBuffer(
			sizeof(uint32_t) * indices.size(),
			vk::BufferUsageFlagBits::eIndexBuffer |
				vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
				vk::BufferUsageFlagBits::eShaderDeviceAddress,
			chunkIndexBuffer,
			true
		);
		if (idxMapped) {
			std::memcpy(idxMapped, indices.data(), sizeof(uint32_t) * indices.size());
		}
	}

	void TerrainAccelerationStructure::BuildOrUpdate(
		vk::CommandBuffer        cmd,
		const glm::vec3&         cameraPos,
		float                    baseTexelSize,
		uint32_t                 numLODs,
		render::PipelineLibrary* pipelineLibrary,
		ComputeShader*           aabbShader,
		vk::DescriptorSet        frameSet,
		vk::DescriptorSet        globalSet,
		vk::DescriptorSetLayout  frameSetLayout,
		vk::DescriptorSetLayout  globalSetLayout,
		const glm::uvec4&        gridParams
	) {
		// Deprecated stub preserved for interface compatibility.
		// Ray query acceleration structures are now maintained from terrain chunk SSBOs via BuildOrUpdateFromChunks.
		(void)cmd;
		(void)cameraPos;
		(void)baseTexelSize;
		(void)numLODs;
		(void)pipelineLibrary;
		(void)aabbShader;
		(void)frameSet;
		(void)globalSet;
		(void)frameSetLayout;
		(void)globalSetLayout;
		(void)gridParams;
	}

	void TerrainAccelerationStructure::BuildOrUpdateFromChunks(
		vk::CommandBuffer cmd,
		vk::DeviceAddress vertexBufferAddr,
		const std::vector<uint32_t>& newPhysicalSlots,
		const std::vector<uint32_t>& activePhysicalSlots
	) {
		if (allocator == VK_NULL_HANDLE || vertexBufferAddr == 0 || !m_enabled) {
			return;
		}

		InitAccelerationStructures();

		auto createBuffer =
			[this](vk::DeviceSize size, vk::BufferUsageFlags usage, BufferResource& res, bool hostMapped = false) {
				if (res.buffer != VK_NULL_HANDLE) {
					return static_cast<void*>(nullptr);
				}
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
					allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						VMA_ALLOCATION_CREATE_MAPPED_BIT;
				}

				VkBuffer          vkBuf = VK_NULL_HANDLE;
				VmaAllocationInfo allocResInfo{};
				if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &vkBuf, &res.allocation, &allocResInfo) ==
			        VK_SUCCESS) {
					res.buffer = vkBuf;
					vk::BufferDeviceAddressInfo addrInfo{};
					addrInfo.setBuffer(res.buffer);
					res.deviceAddress = device.getBufferAddress(addrInfo);
					return allocResInfo.pMappedData;
				}
				return static_cast<void*>(nullptr);
			};

		constexpr uint32_t bytesPerVertex = sizeof(glm::vec4) * 3;
		constexpr uint32_t vertsPerChunk = constants::Class::Terrain::ChunkManager::VerticesPerChunk;
		constexpr uint32_t chunkVertexStride = bytesPerVertex;
		constexpr uint32_t trisPerChunk = constants::Class::Terrain::ChunkManager::TrianglesPerChunk; // 450

		bool anyBlasBuilt = false;

		// Build BLAS for newly allocated/generated chunk physical slots
		for (uint32_t slot : newPhysicalSlots) {
			if (slot >= m_chunkBlasEntries.size()) continue;

			auto& entry = m_chunkBlasEntries[slot];
			if (!entry.blas) {
				createBuffer(
					m_chunkBlasSize,
					vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
					entry.storage,
					false
				);

				vk::AccelerationStructureCreateInfoKHR createInfo{};
				createInfo.setBuffer(entry.storage.buffer);
				createInfo.setSize(m_chunkBlasSize);
				createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
				entry.blas = device.createAccelerationStructureKHR(createInfo, nullptr, dls);

				vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{};
				addrInfo.setAccelerationStructure(entry.blas);
				entry.address = device.getAccelerationStructureAddressKHR(addrInfo, dls);
			}

			vk::DeviceAddress chunkVertAddr = vertexBufferAddr + static_cast<vk::DeviceSize>(slot * vertsPerChunk * chunkVertexStride);

			vk::AccelerationStructureGeometryTrianglesDataKHR chunkTriangles{};
			chunkTriangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
			chunkTriangles.setVertexData(chunkVertAddr);
			chunkTriangles.setVertexStride(chunkVertexStride);
			chunkTriangles.setMaxVertex(vertsPerChunk);
			chunkTriangles.setIndexType(vk::IndexType::eUint32);
			chunkTriangles.setIndexData(chunkIndexBuffer.deviceAddress);

			vk::AccelerationStructureGeometryDataKHR geomData{};
			geomData.setTriangles(chunkTriangles);

			vk::AccelerationStructureGeometryKHR geom{};
			geom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
			geom.setGeometry(geomData);
			geom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

			vk::AccelerationStructureBuildGeometryInfoKHR bInfo{};
			bInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
			bInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
			bInfo.setDstAccelerationStructure(entry.blas);
			bInfo.setScratchData(scratchBuffer.deviceAddress);
			bInfo.setGeometries(geom);

			vk::AccelerationStructureBuildRangeInfoKHR range{};
			range.setPrimitiveCount(trisPerChunk);
			range.setPrimitiveOffset(0);
			range.setFirstVertex(0);
			range.setTransformOffset(0);

			const vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
			cmd.buildAccelerationStructuresKHR(1, &bInfo, &pRange, dls);

			// Synchronize scratch buffer between sequential chunk BLAS builds
			vk::MemoryBarrier barrier{
				vk::AccessFlagBits::eAccelerationStructureWriteKHR,
				vk::AccessFlagBits::eAccelerationStructureWriteKHR | vk::AccessFlagBits::eAccelerationStructureReadKHR
			};
			cmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				vk::DependencyFlags{},
				barrier,
				nullptr,
				nullptr
			);

			entry.built = true;
			anyBlasBuilt = true;
		}

		// Rebuild TLAS only when required by chunk generation (new BLAS built or active chunk set changed)
		if (anyBlasBuilt || activePhysicalSlots != m_lastActiveSlots) {
			m_lastActiveSlots = activePhysicalSlots;

			std::vector<VkAccelerationStructureInstanceKHR> instances;
			instances.reserve(activePhysicalSlots.size());

			for (uint32_t slot : activePhysicalSlots) {
				if (slot < m_chunkBlasEntries.size() && m_chunkBlasEntries[slot].built && m_chunkBlasEntries[slot].address != 0) {
					VkAccelerationStructureInstanceKHR inst{};
					inst.transform.matrix[0][0] = 1.0f;
					inst.transform.matrix[1][1] = 1.0f;
					inst.transform.matrix[2][2] = 1.0f;
					inst.instanceCustomIndex = slot;
					inst.mask = 0xFF;
					inst.instanceShaderBindingTableRecordOffset = 0;
					inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
					inst.accelerationStructureReference = m_chunkBlasEntries[slot].address;
					instances.push_back(inst);
				}
			}

			if (instances.empty()) {
				VkAccelerationStructureInstanceKHR dummyInst{};
				dummyInst.transform.matrix[0][0] = 1.0f;
				dummyInst.transform.matrix[1][1] = 1.0f;
				dummyInst.transform.matrix[2][2] = 1.0f;
				dummyInst.mask = 0x0;
				instances.push_back(dummyInst);
			}

			// Copy instances to mapped instanceBuffer
			void* mapped = nullptr;
			vmaMapMemory(allocator, instanceBuffer.allocation, &mapped);
			if (mapped) {
				std::memcpy(mapped, instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instances.size());
				vmaUnmapMemory(allocator, instanceBuffer.allocation);
			}

			vk::AccelerationStructureGeometryInstancesDataKHR tlasInstancesData{};
			tlasInstancesData.setArrayOfPointers(VK_FALSE);
			tlasInstancesData.setData(instanceBuffer.deviceAddress);

			vk::AccelerationStructureGeometryDataKHR tlasGeomData{};
			tlasGeomData.setInstances(tlasInstancesData);

			vk::AccelerationStructureGeometryKHR tlasGeometry{};
			tlasGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
			tlasGeometry.setGeometry(tlasGeomData);

			vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{};
			tlasBuildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
			tlasBuildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
			tlasBuildInfo.setGeometries(tlasGeometry);
			tlasBuildInfo.setDstAccelerationStructure(tlas);
			tlasBuildInfo.setScratchData(scratchBuffer.deviceAddress);

			vk::AccelerationStructureBuildRangeInfoKHR tlasRange{};
			tlasRange.setPrimitiveCount(static_cast<uint32_t>(instances.size()));
			tlasRange.setPrimitiveOffset(0);
			tlasRange.setFirstVertex(0);
			tlasRange.setTransformOffset(0);

			const vk::AccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;
			cmd.buildAccelerationStructuresKHR(1, &tlasBuildInfo, &pTlasRange, dls);
		}
	}

} // namespace brassica
