#pragma once

// Volumetrics-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	namespace detail {
		// Not a base itself -- VolumetricCascade*Scattering/Extinction all share this exact
		// shape, but each still needs to be its own real type (FixedImage3D's CRTP needs
		// Derived == the type being defined), so this only exists to avoid repeating the same
		// five constants six times.
		struct VolumetricFroxelShape {
			static constexpr std::uint32_t kWidth = 128;
			static constexpr std::uint32_t kHeight = 96;
			static constexpr std::uint32_t kDepth = 64;
			static constexpr vk::Format    kFormat = vk::Format::eR16G16B16A16Sfloat;
			static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		};
	} // namespace detail

	struct VolumetricCascade0Scattering: FixedImage3D<VolumetricCascade0Scattering>, detail::VolumetricFroxelShape {};

	struct VolumetricCascade0Extinction: FixedImage3D<VolumetricCascade0Extinction>, detail::VolumetricFroxelShape {};

	struct VolumetricCascade1Scattering: FixedImage3D<VolumetricCascade1Scattering>, detail::VolumetricFroxelShape {};

	struct VolumetricCascade1Extinction: FixedImage3D<VolumetricCascade1Extinction>, detail::VolumetricFroxelShape {};

	struct VolumetricCascade2Scattering: FixedImage3D<VolumetricCascade2Scattering>, detail::VolumetricFroxelShape {};

	struct VolumetricCascade2Extinction: FixedImage3D<VolumetricCascade2Extinction>, detail::VolumetricFroxelShape {};

} // namespace brassica
