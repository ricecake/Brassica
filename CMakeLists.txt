cmake_minimum_required(VERSION 3.20)
project(Brassica VERSION 1.0 LANGUAGES CXX)

# 1. C++ Standard and Flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Get the current git hash and branch
execute_process(COMMAND git rev-parse --abbrev-ref HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
endif()
if(NOT GIT_BRANCH)
    set(GIT_BRANCH "unknown")
endif()
configure_file(include/build_info.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/build_info.h)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

option(BRASSICA_BUILD_SHARED "Build brassica as a shared library" OFF)
if(BRASSICA_BUILD_SHARED)
    set(BRASSICA_LIB_TYPE SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    set(BRASSICA_LIB_TYPE STATIC)
endif()

# 2. Dependency Discovery
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# 3. Submodule Configurations
# Disable testing/examples globally for all submodules to speed up build
set(BUILD_TESTING OFF CACHE BOOL "Disable submodule tests" FORCE)

# JoltPhysics specific configuration
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(USE_SSE4_2 ON CACHE BOOL "Enable SSE4.2 for Jolt" FORCE) # Standard baseline for modern CPUs

# Shaderc specific configuration
set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)

set(FG_BUILD_TEST OFF CACHE BOOL "" FORCE)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

# Add well-behaved CMake submodules
add_subdirectory(external/glfw)
add_subdirectory(external/fmt)
add_subdirectory(external/spdlog)
add_subdirectory(external/enkits)
add_subdirectory(external/fastgltf)
add_subdirectory(external/JoltPhysics/Build)
add_subdirectory(external/vk-bootstrap)
add_subdirectory(external/shaderc)
add_subdirectory(external/meshoptimizer)
add_subdirectory(external/FrameGraph)
add_subdirectory(external/tracy)
add_subdirectory(external/VulkanMemoryAllocator)

set(IMGUI_DIR external/imgui)
set(IMGUIZMO_DIR external/ImGuizmo)

# Safely locate ImGuizmo sources (avoids example folders)
file(GLOB IMGUIZMO_SOURCES
    "${IMGUIZMO_DIR}/*.cpp"
    "${IMGUIZMO_DIR}/src/*.cpp"
)

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUIZMO_SOURCES}
)

# Export ImGuizmo headers alongside ImGui
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUIZMO_DIR}
    ${IMGUIZMO_DIR}/src
)

target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# 4. Define the Library
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
add_library(brassica ${BRASSICA_LIB_TYPE} ${LIB_SOURCES})

target_compile_definitions(brassica PUBLIC GLM_ENABLE_EXPERIMENTAL BRASSICA_BUILD_DIR="${CMAKE_BINARY_DIR}")
target_compile_options(brassica PRIVATE -march=native -ffast-math)

# Interface/Header-only includes mapped from your structure
target_include_directories(brassica
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        external/glm
        external/entt/single_include
        external/stb
        external/miniaudio
        external/imgui_entt_entity_editor
)

# 5. Link Dependencies
target_link_libraries(brassica
    PUBLIC
        Vulkan::Vulkan
        glfw
        imgui
        fmt::fmt
        spdlog::spdlog
        enkiTS
        fastgltf::fastgltf
        Jolt
        vk-bootstrap
        shaderc_shared
        meshoptimizer
        FrameGraph
        TracyClient
        Threads::Threads
        VulkanMemoryAllocator
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
    target_link_libraries(brassica PRIVATE stdc++exp)
endif()

# 6. Build Apps, Examples, and Tests
file(GLOB APP_DIRS "apps/*")
foreach(APP_DIR ${APP_DIRS})
    if(IS_DIRECTORY ${APP_DIR})
        get_filename_component(APP_NAME ${APP_DIR} NAME)
        file(GLOB APP_SOURCES "${APP_DIR}/main.cpp" "${APP_DIR}/src/*.cpp")
        if(APP_SOURCES)
            add_executable(${APP_NAME} ${APP_SOURCES})
            target_link_libraries(${APP_NAME} PRIVATE brassica)
            target_include_directories(${APP_NAME} PRIVATE "${APP_DIR}/include")
            set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
        endif()
    endif()
endforeach()

# Tests
enable_testing()
add_subdirectory(external/doctest)

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE brassica doctest::doctest)
    set_target_properties(${TEST_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
