#pragma once

// The physical layer's one point of contact with real Vulkan headers. Everything downstream of
// this file (PhysicalResource.hpp, PhysicalRegistry.hpp, PhysicalExecutionBackend.hpp,
// BarrierTranslator.hpp) requires a real Vulkan SDK to compile -- unlike the declarative layer
// (Execution.hpp and below), which stays Vulkan-free on purpose. See Engine.hpp for the engine's
// own use of the same headers.

#include <vk_mem_alloc.h>

#include <vulkan/vulkan.hpp>
