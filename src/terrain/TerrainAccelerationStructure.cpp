#include "terrain/TerrainAccelerationStructure.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace brassica {

	void TerrainAccelerationStructure::DestroyAccelerationStructures() {
		if (!lastAllocator)
			return;

		auto destroyAS = [this](vk::AccelerationStructureKHR& as, BufferResource& buf) {
			if (as) {
				device.destroyAccelerationStructureKHR(as, nullptr, dls);
				as = nullptr;
			}
			if (buf.buffer && buf.allocation) {
				vmaDestroyBuffer(lastAllocator, buf.buffer, buf.allocation);
				buf.buffer = nullptr;
				buf.allocation = VK_NULL_HANDLE;
				buf.deviceAddress = 0;
			}
		};

		auto destroyBuf = [this](BufferResource& buf) {
			if (buf.buffer && buf.allocation) {
				vmaDestroyBuffer(lastAllocator, buf.buffer, buf.allocation);
				buf.buffer = nullptr;
				buf.allocation = VK_NULL_HANDLE;
				buf.deviceAddress = 0;
			}
		};

		destroyAS(tlas, tlasBuffer);
		destroyAS(blas, blasBuffer);
		destroyBuf(aabbBuffer);
		destroyBuf(instanceBuffer);
		destroyBuf(scratchBuffer);
	}

	void TerrainAccelerationStructure::BuildOrUpdate(
		VmaAllocator     allocator,
		const glm::vec3& cameraPos,
		float            baseTexelSize,
		uint32_t         numLODs
	) {
		if (allocator == VK_NULL_HANDLE)
			return;
		lastAllocator = allocator;

		if (tlas && glm::distance(cameraPos, lastASCameraPos) < 16.0f) {
			return; // Rebuild AS only when camera moves across a grid cell threshold
		}
		lastASCameraPos = cameraPos;

		// Generate distance-aware AABBs for the terrain grid chunks.
		// For points/AABBs close to the camera, resolution is finer (e.g., 32 world units per AABB).
		// For points/AABBs further from the camera (shadow caster point distance), resolution is coarser (64, 128,
		// etc.).
		std::vector<VkAabbPositionsKHR> aabbs;

		uint32_t meshletsPerRow = 16;
		for (uint32_t lod = 0; lod < numLODs; ++lod) {
			float     baseMeshletSize = 32.0f;
			float     meshletSize = baseMeshletSize * std::pow(2.0f, std::min(0.0f, static_cast<float>(lod - 1)));
			glm::vec2 cameraSnap = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / meshletSize) * meshletSize;

			for (uint32_t row = 0; row < meshletsPerRow; ++row) {
				for (uint32_t col = 0; col < meshletsPerRow; ++col) {
					glm::vec3 minB(
						cameraSnap.x +
							(static_cast<float>(col) - static_cast<float>(meshletsPerRow) * 0.5f) * meshletSize,
						-500.0f,
						cameraSnap.y +
							(static_cast<float>(row) - static_cast<float>(meshletsPerRow) * 0.5f) * meshletSize
					);
					glm::vec3 maxB = minB + glm::vec3(meshletSize, 2000.0f, meshletSize);

					// Radial ring check matching task shader to only generate AABBs for active LOD regions
					if (lod > 0) {
						float prevMeshletSize = baseMeshletSize * std::pow(2.0f, static_cast<float>(lod - 1));
						float safeInnerRadius = (static_cast<float>(meshletsPerRow) * 0.5f - 1.0f) * prevMeshletSize;
						glm::vec2 maxOffset = glm::max(
							glm::abs(glm::vec2(minB.x, minB.z) - glm::vec2(cameraPos.x, cameraPos.z)),
							glm::abs(glm::vec2(maxB.x, maxB.z) - glm::vec2(cameraPos.x, cameraPos.z))
						);
						float maxDistToCam = glm::length(maxOffset);
						if (maxDistToCam < safeInnerRadius) {
							continue; // Region covered by finer LOD
						}
					}

					VkAabbPositionsKHR aabb{};
					aabb.minX = minB.x;
					aabb.minY = minB.y;
					aabb.minZ = minB.z;
					aabb.maxX = maxB.x;
					aabb.maxY = maxB.y;
					aabb.maxZ = maxB.z;

					aabbs.push_back(aabb);
				}
			}
		}

		if (aabbs.empty())
			return;

		// Ensure GPU has finished reading/using previous TLAS before destroying or updating
		device.waitIdle();

		DestroyAccelerationStructures();

		// Helper to create Vulkan memory buffer with device address flag
		auto createBuffer =
			[this,
			 allocator](vk::DeviceSize size, vk::BufferUsageFlags usage, BufferResource& res, bool hostMapped = false) {
				VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
				bufInfo.size = size;
				bufInfo.usage = static_cast<VkBufferUsageFlags>(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);

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

		// 1. Upload AABBs to GPU Buffer
		vk::DeviceSize aabbBufferSize = sizeof(VkAabbPositionsKHR) * aabbs.size();
		void*          aabbMapped = createBuffer(
			aabbBufferSize,
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			aabbBuffer,
			true
		);
		if (aabbMapped) {
			std::memcpy(aabbMapped, aabbs.data(), aabbBufferSize);
		}

		// 2. Build BLAS
		vk::AccelerationStructureGeometryAabbsDataKHR aabbGeomData{};
		aabbGeomData.setData(aabbBuffer.deviceAddress);
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

		uint32_t                                   primitiveCount = static_cast<uint32_t>(aabbs.size());
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

		// Get BLAS device address
		vk::AccelerationStructureDeviceAddressInfoKHR blasAddrInfo{};
		blasAddrInfo.setAccelerationStructure(blas);
		vk::DeviceAddress blasAddress = device.getAccelerationStructureAddressKHR(blasAddrInfo, dls);

		// 3. Build TLAS Instance
		VkAccelerationStructureInstanceKHR instanceData{};
		instanceData.transform.matrix[0][0] = 1.0f;
		instanceData.transform.matrix[1][1] = 1.0f;
		instanceData.transform.matrix[2][2] = 1.0f;
		instanceData.instanceCustomIndex = 0;
		instanceData.mask = 0xFF;
		instanceData.instanceShaderBindingTableRecordOffset = 0;
		instanceData.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instanceData.accelerationStructureReference = blasAddress;

		void* instMapped = createBuffer(
			sizeof(VkAccelerationStructureInstanceKHR),
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			instanceBuffer,
			true
		);
		if (instMapped) {
			std::memcpy(instMapped, &instanceData, sizeof(VkAccelerationStructureInstanceKHR));
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

		uint32_t                                   tlasInstanceCount = 1;
		vk::AccelerationStructureBuildSizesInfoKHR tlasSizeInfo{};
		device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,
			&tlasBuildInfo,
			&tlasInstanceCount,
			&tlasSizeInfo,
			dls
		);

		createBuffer(
			tlasSizeInfo.accelerationStructureSize,
			vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
			tlasBuffer,
			false
		);

		vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{};
		tlasCreateInfo.setBuffer(tlasBuffer.buffer);
		tlasCreateInfo.setSize(tlasSizeInfo.accelerationStructureSize);
		tlasCreateInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
		tlas = device.createAccelerationStructureKHR(tlasCreateInfo, nullptr, dls);

		// Allocate Scratch Buffer for build commands
		vk::DeviceSize scratchSize = std::max(blasSizeInfo.buildScratchSize, tlasSizeInfo.buildScratchSize);
		createBuffer(scratchSize, vk::BufferUsageFlagBits::eStorageBuffer, scratchBuffer, false);

		// Execute Acceleration Structure Build Commands using Command Pool / Queue
		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eTransient);
		vk::CommandPool tempPool = device.createCommandPool(poolInfo);

		vk::CommandBufferAllocateInfo cmdAlloc{};
		cmdAlloc.setCommandPool(tempPool);
		cmdAlloc.setCommandBufferCount(1);
		vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc).front();

		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		// Build BLAS
		blasBuildInfo.setDstAccelerationStructure(blas);
		blasBuildInfo.setScratchData(scratchBuffer.deviceAddress);

		vk::AccelerationStructureBuildRangeInfoKHR blasRange{};
		blasRange.setPrimitiveCount(primitiveCount);
		blasRange.setPrimitiveOffset(0);
		blasRange.setFirstVertex(0);
		blasRange.setTransformOffset(0);

		const vk::AccelerationStructureBuildRangeInfoKHR* pBlasRange = &blasRange;
		cmd.buildAccelerationStructuresKHR(1, &blasBuildInfo, &pBlasRange, dls);

		// Memory barrier between BLAS build and TLAS build
		vk::MemoryBarrier barrier{
			vk::AccessFlagBits::eAccelerationStructureWriteKHR,
			vk::AccessFlagBits::eAccelerationStructureReadKHR
		};
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::DependencyFlags{},
			barrier,
			nullptr,
			nullptr
		);

		// Build TLAS
		tlasBuildInfo.setDstAccelerationStructure(tlas);
		tlasBuildInfo.setScratchData(scratchBuffer.deviceAddress);

		vk::AccelerationStructureBuildRangeInfoKHR tlasRange{};
		tlasRange.setPrimitiveCount(1);
		tlasRange.setPrimitiveOffset(0);
		tlasRange.setFirstVertex(0);
		tlasRange.setTransformOffset(0);

		const vk::AccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;
		cmd.buildAccelerationStructuresKHR(1, &tlasBuildInfo, &pTlasRange, dls);

		cmd.end();

		// Submit command buffer synchronously
		vk::Queue      queue = device.getQueue(0, 0);
		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(cmd);
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();

		device.destroyCommandPool(tempPool);
	}

} // namespace brassica
