#include "terrain/TerrainAccelerationStructure.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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
		destroyBuf(aabbBuffer);
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

		uint32_t       maxAABBs = 12 * 16 * 16;
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

		vk::DeviceSize scratchSize = std::max(blasSizeInfo.buildScratchSize, tlasSizeInfo.buildScratchSize);
		createBuffer(scratchSize, vk::BufferUsageFlagBits::eStorageBuffer, scratchBuffer, false);
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
		(void)baseTexelSize;
		if (allocator == VK_NULL_HANDLE || !pipelineLibrary || !aabbShader)
			return;

		if (tlas && glm::distance(cameraPos, lastASCameraPos) < 16.0f) {
			return;
		}
		lastASCameraPos = cameraPos;

		InitAccelerationStructures();

		uint32_t       maxAABBs = numLODs * 16 * 16;
		vk::DeviceSize aabbBufferSize = sizeof(std::uint64_t) + sizeof(VkAabbPositionsKHR) * maxAABBs;

		cmd.fillBuffer(aabbBuffer.buffer, 0, sizeof(std::uint64_t), 0);

		vk::BufferMemoryBarrier fillBarrier{
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			aabbBuffer.buffer,
			0,
			aabbBufferSize
		};
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eComputeShader,
			vk::DependencyFlags{},
			nullptr,
			fillBarrier,
			nullptr
		);

		std::array<vk::DescriptorSetLayout, 2> setLayouts{frameSetLayout, globalSetLayout};
		std::array<vk::PushConstantRange, 1>   pushConstantRanges{
			vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainAABBPushConstants)}
		};

		render::ComputePipelineRequest request{
			.shader = aabbShader,
			.setLayouts = setLayouts,
			.pushConstantRanges = pushConstantRanges,
		};
		render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

		if (resolved.pipeline) {
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
		}

		std::array<vk::DescriptorSet, 2> boundSets{frameSet, globalSet};
		if (boundSets[0] && boundSets[1]) {
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
		}

		TerrainAABBPushConstants push{
			.aabbBufferAddr = glm::uvec2(
				static_cast<uint32_t>(aabbBuffer.deviceAddress & 0xFFFFFFFFu),
				static_cast<uint32_t>(aabbBuffer.deviceAddress >> 32u)
			),
			.padding = glm::uvec2(0),
			.gridParams = gridParams
		};
		cmd.pushConstants(
			resolved.layout,
			vk::ShaderStageFlagBits::eCompute,
			0,
			sizeof(TerrainAABBPushConstants),
			&push
		);

		uint32_t groupCount = (maxAABBs + 63) / 64;
		cmd.dispatch(groupCount, 1, 1);

		vk::BufferMemoryBarrier aabbBarrier{
			vk::AccessFlagBits::eShaderWrite,
			vk::AccessFlagBits::eAccelerationStructureReadKHR,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			aabbBuffer.buffer,
			0,
			aabbBufferSize
		};
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::DependencyFlags{},
			nullptr,
			aabbBarrier,
			nullptr
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
		blasBuildInfo.setDstAccelerationStructure(blas);
		blasBuildInfo.setScratchData(scratchBuffer.deviceAddress);

		uint32_t                                   primitiveCount = maxAABBs;
		vk::AccelerationStructureBuildRangeInfoKHR blasRange{};
		blasRange.setPrimitiveCount(primitiveCount);
		blasRange.setPrimitiveOffset(0);
		blasRange.setFirstVertex(0);
		blasRange.setTransformOffset(0);

		const vk::AccelerationStructureBuildRangeInfoKHR* pBlasRange = &blasRange;
		cmd.buildAccelerationStructuresKHR(1, &blasBuildInfo, &pBlasRange, dls);

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

		vk::AccelerationStructureDeviceAddressInfoKHR blasAddrInfo{};
		blasAddrInfo.setAccelerationStructure(blas);
		vk::DeviceAddress blasAddress = device.getAccelerationStructureAddressKHR(blasAddrInfo, dls);

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
		tlasRange.setPrimitiveCount(1);
		tlasRange.setPrimitiveOffset(0);
		tlasRange.setFirstVertex(0);
		tlasRange.setTransformOffset(0);

		const vk::AccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;
		cmd.buildAccelerationStructuresKHR(1, &tlasBuildInfo, &pTlasRange, dls);
	}

} // namespace brassica
