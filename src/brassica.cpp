#include "brassica.hpp"

// 1. Logging & Profiling
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

// 2. Math & ECS
#include <glm/glm.hpp>
#include <entt/entt.hpp>

// 3. Vulkan & Windowing
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

// 4. Assets & Rendering
#include <fastgltf/core.hpp>
#include <shaderc/shaderc.hpp>
#include <meshoptimizer.h>
#include <fg/FrameGraph.hpp>

// 5. UI
#include <imgui.h>
#include <ImGuizmo.h>
#include <imgui_entt_entity_editor.hpp>

// 6. Physics & Audio
#include <Jolt/Jolt.h>
#include <Jolt/Core/Memory.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/RegisterTypes.h>
#include <miniaudio.h>

// 7. Threading
#include <TaskScheduler.h>

namespace brassica {

void InitializeCore() {
    ZoneScoped; // Tracy profiler macro to test Tracy linkage

    // Test Logging (fmt & spdlog)
    fmt::print("Bootstrapping Brassica Engine...\n");
    spdlog::info("Logger linked and ready.");

    // Test Math & ECS (glm & entt)
    glm::vec3 testVec(1.0f, 2.0f, 3.0f);
    entt::registry registry;
    auto entity = registry.create();

    // Test Threading (enkiTS)
    enki::TaskScheduler scheduler;
    scheduler.Initialize();
    spdlog::info("enkiTS initialized with {} threads.", scheduler.GetNumTaskThreads());

    // Test Physics Linking (Jolt)
    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
    spdlog::info("Jolt Physics registered.");

    // Test Windowing Linking (GLFW)
    if (glfwInit()) {
        spdlog::info("GLFW linked and initialized.");
        glfwTerminate();
    }

    // Test Shader Compilation Linking (shaderc)
    shaderc::Compiler compiler;
    if (compiler.IsValid()) {
        spdlog::info("Shaderc compiler linked.");
    }

    // Test Asset Linking (fastgltf)
    fastgltf::Parser parser;
    spdlog::info("fastgltf parser linked.");

    // Test vk-bootstrap
    vkb::InstanceBuilder builder;
    auto inst_ret = builder.set_app_name("Brassica Test")
                           .request_validation_layers(false)
                           .use_default_debug_messenger()
                           .build();
    if (inst_ret) {
         spdlog::info("vk-bootstrap linked.");
    }
}

} // namespace brassica
