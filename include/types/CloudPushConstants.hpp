#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	struct CloudBakePushConstants {
		std::uint32_t weatherStorageIdx{0};
		std::uint32_t minMaxStorageIdx{0};
		std::uint32_t volumeStorageIdx{0};
		float         worldScale{1.0f};
		float         cloudCoverage{0.35f};
		float         cloudThickness{1500.0f};
		float         time{0.0f};
		std::uint32_t padding{0};
	};

	struct CloudBoundingPushConstants {
		std::uint32_t depthTextureIdx{0};
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t boundingStorageIdx{0};
		float         cloudMaxRayDistance{175000.0f};
		float         renderScale{1.0f};
		float         worldScale{1.0f};
		float         cloudCoverage{0.35f};
		float         cloudAltitude{2000.0f};
		float         cloudThickness{1500.0f};
		float         time{0.0f};
		std::uint32_t frameIndex{0};
		std::uint32_t padding{0};
	};

	struct CloudShadowBakePushConstants {
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t shadowMapStorageIdx{0};
		std::uint32_t frameIndex{0};
		float         worldScale{1.0f};
		float         cloudAltitude{2000.0f};
		float         cloudThickness{1500.0f};
		std::uint32_t pad0{0};
		std::uint32_t pad1{0};
		glm::mat4     lightSpaceMatrix{1.0f};
		glm::mat4     invLightSpaceMatrix{1.0f};
		glm::vec3     primaryLightDir{0.0f, 1.0f, 0.0f};
		std::uint32_t pad2{0};
	};

	struct CloudTileSchedulerPushConstants {
		std::uint32_t boundingMapIdx{0};
		std::uint32_t errorMapStorageIdx{0};
		std::uint64_t tileQueueBufferAddr{0};
		std::uint64_t indirectDispatchBufferAddr{0};
		std::int32_t  pass{0};
		float         priorityErrorWeight{2.5f};
		float         priorityGradWeight{2.0f};
		float         priorityAgeWeight{0.05f};
		float         priorityNeighborErrorWeight{1.5f};
		float         priorityNeighborGradWeight{1.0f};
		float         priorityThreshold{0.05f};
		std::uint32_t padding{0};
	};

	struct CloudRenderPushConstants {
		std::uint32_t depthTextureIdx{0};
		std::uint32_t boundingMapIdx{0};
		std::uint32_t weatherMinMaxIdx{0};
		std::uint32_t weatherTextureIdx{0};
		std::uint32_t volume3DIdx{0};
		std::uint32_t packedColorStorageIdx{0};
		std::uint32_t packedDepthStorageIdx{0};
		std::uint32_t packedVelocityStorageIdx{0};
		std::uint32_t errorMapStorageIdx{0};
		std::uint32_t pad0{0};
		std::uint64_t tileQueueBufferAddr{0};
		float         cloudMaxRayDistance{175000.0f};
		float         renderScale{1.0f};
		float         worldScale{1.0f};
		std::int32_t  cloudMinSamples{32};
		std::int32_t  cloudMaxSamples{96};
		float         cloudExtinction{0.372f};
		float         deltaTime{0.016f};
		float         cloudAltitude{2000.0f};
		float         cloudThickness{1500.0f};
		float         cloudDensity{0.100f};
		float         cloudCoverage{0.35f};
		glm::vec3     cloudExtinctionColor{1.0f, 1.0f, 1.0f};
		float         pad1{0.0f};
		glm::vec3     cloudAlbedo{0.85f, 0.85f, 0.85f};
		float         pad2{0.0f};
	};

	struct CloudTemporalPushConstants {
		std::uint32_t packedColorIdx{0};
		std::uint32_t packedDepthIdx{0};
		std::uint32_t packedVelocityIdx{0};
		std::uint32_t boundingMapIdx{0};
		std::uint32_t historyColorIdx{0};
		std::uint32_t historyDepthIdx{0};
		std::uint32_t historyMomentsIdx{0};
		std::uint32_t colorStorageIdx{0};
		std::uint32_t depthStorageIdx{0};
		std::uint32_t momentsStorageIdx{0};
		std::uint32_t errorMapStorageIdx{0};
		std::uint32_t pad0{0};
		std::uint64_t tileQueueBufferAddr{0};
		float         cloudTemporalGamma{1.1f};
		float         cloudMaxHistoryLength{32.0f};
		float         cloudMaxRayDistance{175000.0f};
		float         renderScale{1.0f};
		float         deltaTime{0.016f};
		std::int32_t  enableTemporal{1};
		std::int32_t  hasHistory{0};
		std::int32_t  useTileQueue{1};
	};

	struct CloudSpatialFilterPushConstants {
		std::uint32_t cloudColorIdx{0};
		std::uint32_t cloudDepthIdx{0};
		std::uint32_t cloudMomentsIdx{0};
		std::uint32_t errorMapIdx{0};
		std::uint32_t filteredColorStorageIdx{0};
		std::int32_t  stepSize{1};
		std::int32_t  passIndex{0};
		float         phiLuma{20.0f};
		float         phiDensity{0.05f};
		float         phiDepth{1.5f};
		float         svgfHistoryBoost{4.0f};
		float         svgfHistoryThreshold{10.0f};
	};

} // namespace brassica
