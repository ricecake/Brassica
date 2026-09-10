#pragma once

#include "vulkan/vulkan.hpp"

// Vulkan-Hpp portability shim for the dispatcher-related classes that
// KhronosGroup/Vulkan-Hpp#1983 ("Move all the dispatcher related classes into namespace
// detail") relocated from vk:: into vk::detail::. Verified against upstream history: that PR's
// merge commit (ed3cf7a) lands between tagged releases v1.3.300 (still top-level
// vk::DispatchLoaderDynamic) and v1.3.301 (moved to vk::detail::DispatchLoaderDynamic, with no
// backward-compatible alias kept at the old location) -- a hard break, not a deprecation window.
//
// Use brassica::DispatchLoaderDynamic everywhere instead of naming either spelling directly, so
// this codebase builds against Vulkan-Hpp on either side of that break.
namespace brassica {

#if VK_HEADER_VERSION >= 301
	using DispatchLoaderDynamic = vk::detail::DispatchLoaderDynamic;
#else
	using DispatchLoaderDynamic = vk::DispatchLoaderDynamic;
#endif

} // namespace brassica
