cmake_minimum_required(VERSION 3.20)
project(Brassica VERSION 1.0 LANGUAGES CXX)

# 1. C++ Standard and Flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Corresponds to -std=gnu++23
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(STATUS "LTO/IPO is not supported: ${error}")
endif()

# Get the current git hash and branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
endif()

if(NOT GIT_BRANCH)
    set(GIT_BRANCH "unknown")
endif()

set(BRASSICA_GIT_IDENT "${GIT_BRANCH}-${GIT_HASH}")

configure_file(include/build_info.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/build_info.h)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()


option(ENABLE_PROFILING "Enable profiling instrumentation" OFF)
if(ENABLE_PROFILING)
    add_definitions(-DPROFILING_ENABLED)
endif()

option(BRASSICA_BUILD_SHARED "Build brassica as a shared library" OFF)

if(BRASSICA_BUILD_SHARED)
    set(BRASSICA_LIB_TYPE SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    set(BRASSICA_LIB_TYPE STATIC)
endif()

option(BRASSICA_USE_SPIRV "Compile and use SPIR-V shaders" OFF)
if(BRASSICA_USE_SPIRV)
    add_definitions(-DBRASSICA_USE_SPIRV)
endif()

set(IMGUI_DIR external/imgui)
set(STB_DIR "${CMAKE_SOURCE_DIR}/external/stb")

# 2. Dependency Discovery
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

find_package(Backtrace REQUIRED)

# 1. Define ImGui FIRST so FastNoise2 can see it
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

# 2. Tell FastNoise2 where the ImGui headers are explicitly
set(IMGUI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${IMGUI_DIR}" CACHE PATH "" FORCE)

get_directory_property(OLD_COMPILE_OPTIONS COMPILE_OPTIONS)
add_compile_options(-w) # Disable all warnings for GCC/Clang for external libraries
set(BUILD_TESTING OFF CACHE BOOL "Disable submodule tests" FORCE)
add_subdirectory(external/task-thread-pool)
add_subdirectory(external/poolSTL)
add_subdirectory(external/libmorton)
add_subdirectory(external/meshoptimizer)

# Bonxai - header-only voxel grid library (only need core, not ROS/map modules)
set(BONXAI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/Bonxai/bonxai_core/include")

set_directory_properties(PROPERTIES COMPILE_OPTIONS "${OLD_COMPILE_OPTIONS}")

# 3. Define the Library
# Finds all .cpp files in src/
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")

add_library(brassica ${BRASSICA_LIB_TYPE} ${LIB_SOURCES})
target_compile_definitions(brassica PUBLIC GLM_ENABLE_EXPERIMENTAL)
target_compile_definitions(brassica PUBLIC BRASSICA_BUILD_DIR="${CMAKE_BINARY_DIR}")

target_compile_options(brassica PRIVATE -march=native -ffast-math)


target_include_directories(brassica
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        external/include
        external/libmorton/include
        ${BONXAI_INCLUDE_DIR}
        ${IMGUI_DIR}/
        ${IMGUI_DIR}/backends
        ${STB_DIR}/
        ${ASSIMP_INCLUDE_DIR}
        ${BULLET_INCLUDE_DIRS}
        external/eigen
)

# 4. Platform-Specific Linking
target_link_libraries(brassica
    PUBLIC
        OpenGL::GL
        GLEW::GLEW
        glfw
        imgui
        task-thread-pool::task-thread-pool
        poolSTL::poolSTL
        ${ASSIMP_LIBRARIES}
        ZLIB::ZLIB
        libmorton::libmorton
        meshoptimizer
        Threads::Threads
        glm::glm
)

# Link GCC-specific backtrace libraries manually if necessary
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
        target_link_libraries(brassica PRIVATE stdc++exp)
    else()
        target_link_libraries(brassica PRIVATE stdc++_libbacktrace)
    endif()
endif()

# 5. Build Examples
file(GLOB EXAMPLE_DIRS "examples/*")

foreach(EXAMPLE_DIR ${EXAMPLE_DIRS})
    if(IS_DIRECTORY ${EXAMPLE_DIR})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_DIR} NAME)

        # Find all source files in the example's directory
        file(GLOB EXAMPLE_SOURCES
            "${EXAMPLE_DIR}/main.cpp"
            "${EXAMPLE_DIR}/src/*.cpp"
        )

        if(EXAMPLE_SOURCES)
            add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCES})

            target_link_libraries(${EXAMPLE_NAME} PRIVATE brassica)
            target_include_directories(${EXAMPLE_NAME} PRIVATE "${EXAMPLE_DIR}/include")

            target_compile_definitions(${EXAMPLE_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
            set_target_properties(${EXAMPLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
        endif()
    endif()
endforeach()

# Build Applications
file(GLOB APP_DIRS "app/*")

foreach(APP_DIR ${APP_DIRS})
    if(IS_DIRECTORY ${APP_DIR})
        get_filename_component(APP_NAME ${APP_DIR} NAME)

        file(GLOB APP_SOURCES
            "${APP_DIR}/main.cpp"
            "${APP_DIR}/src/*.cpp"
        )

        if(APP_SOURCES)
            add_executable(${APP_NAME} ${APP_SOURCES})
            target_link_libraries(${APP_NAME} PRIVATE brassica)
            target_include_directories(${APP_NAME} PRIVATE "${APP_DIR}/include")
            target_compile_definitions(${APP_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
            set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
        endif()
    endif()
endforeach()

# Copy assets and shaders to the root of the build directory
set(EXTRA_SHADER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/lygia"
)

set(COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/assets
    ${CMAKE_BINARY_DIR}/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${CMAKE_BINARY_DIR}/shaders
)

foreach(DIR ${EXTRA_SHADER_INCLUDE_DIRS})
    get_filename_component(DIR_NAME ${DIR} NAME)
    list(APPEND COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${DIR}
        ${CMAKE_BINARY_DIR}/shaders/${DIR_NAME}
    )
endforeach()

add_custom_command(TARGET brassica POST_BUILD ${COPY_COMMANDS})

# Find the clang-format executable
find_program(CLANG_FORMAT_EXE NAMES clang-format)

if(CLANG_FORMAT_EXE)
    # Define the files to be formatted, matching your Makefile's scope
    file(GLOB_RECURSE FORMAT_SOURCES
        "src/*.cpp" "src/*.h" "src/*.hpp"
        "include/*.cpp" "include/*.h" "include/*.hpp"
        "examples/*.cpp" "examples/*.h" "examples/*.hpp"
        "shaders/*"
    )

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${FORMAT_SOURCES}
        COMMENT "Running clang-format on all source files..."
    )
else()
    message(WARNING "clang-format not found. The 'format' target will not be available.")
endif()

# 6. Build Tests
# Note: Tests are included in the default build ('all') to ensure they are available
# for 'ctest' in CI environments, avoiding "executable not found" errors.
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Compile-only check target
# Using STATIC library ensures it compiles but doesn't link into a binary.
# Linking to brassica ensures it inherits all necessary include paths and definitions.
add_library(check STATIC ${LIB_SOURCES})
target_compile_definitions(check PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(check PRIVATE brassica)
set_target_properties(check PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Build Tools
file(GLOB TOOL_SOURCES "tools/*.cpp")

foreach(TOOL_SOURCE ${TOOL_SOURCES})
    get_filename_component(TOOL_NAME ${TOOL_SOURCE} NAME_WE)

    add_executable(${TOOL_NAME} ${TOOL_SOURCE})
    target_link_libraries(${TOOL_NAME} PRIVATE brassica)
    add_dependencies(check ${TOOL_NAME})
endforeach()

# Find glslangValidator
find_program(GLSLANG_VALIDATOR_EXE NAMES glslangValidator)

if(GLSLANG_VALIDATOR_EXE)
    file(GLOB_RECURSE SHADER_FILES
        "shaders/*.vert"
        "shaders/*.frag"
        "shaders/*.geom"
        "shaders/*.comp"
        "shaders/*.tcs"
        "shaders/*.tes"
    )

    # Filter out files that are known to be include-only or don't have a main()
    list(REMOVE_ITEM SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/visual_effects.frag")
    list(REMOVE_ITEM SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/visual_effects.vert")

    set(VALIDATION_OUTPUTS "")
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)

        # Map extensions for glslangValidator compatibility
        set(VAL_EXT ${SHADER_EXT})
        if(SHADER_EXT STREQUAL ".tcs")
            set(VAL_EXT ".tesc")
        elseif(SHADER_EXT STREQUAL ".tes")
            set(VAL_EXT ".tese")
        endif()

        # Determine preprocessed path
        file(RELATIVE_PATH REL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${SHADER_FILE})
        get_filename_component(REL_DIR ${REL_SHADER} DIRECTORY)
        get_filename_component(BASE_NAME ${REL_SHADER} NAME_WE)

        set(PREPROCESSED_FILE "${CMAKE_BINARY_DIR}/shaders_preprocessed/${REL_DIR}/${BASE_NAME}${VAL_EXT}")
        set(STAMP_FILE "${PREPROCESSED_FILE}.validated")

        add_custom_command(
            OUTPUT ${PREPROCESSED_FILE}
            COMMAND shader_preprocessor ${SHADER_FILE} ${PREPROCESSED_FILE}
            DEPENDS shader_preprocessor ${SHADER_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Preprocessing shader ${REL_SHADER}"
        )

        if(BRASSICA_USE_SPIRV)
            set(SPV_FILE "${PREPROCESSED_FILE}.spv")
            add_custom_command(
                OUTPUT ${STAMP_FILE} ${SPV_FILE}
                COMMAND ${GLSLANG_VALIDATOR_EXE} -G --auto-map-locations --auto-map-bindings ${PREPROCESSED_FILE} -o ${SPV_FILE}
                COMMAND ${CMAKE_COMMAND} -E touch ${STAMP_FILE}
                DEPENDS ${PREPROCESSED_FILE}
                COMMENT "Compiling shader ${REL_SHADER} to SPIR-V"
            )
        else()
            add_custom_command(
                OUTPUT ${STAMP_FILE}
                COMMAND ${GLSLANG_VALIDATOR_EXE} ${PREPROCESSED_FILE}
                COMMAND ${CMAKE_COMMAND} -E touch ${STAMP_FILE}
                DEPENDS ${PREPROCESSED_FILE}
                COMMENT "Validating shader ${REL_SHADER}"
            )
        endif()

        list(APPEND VALIDATION_OUTPUTS ${STAMP_FILE})
    endforeach()

    add_custom_target(glcheck DEPENDS ${VALIDATION_OUTPUTS})
else()
    message(WARNING "glslangValidator not found. 'glcheck' target will be empty.")
    add_custom_target(glcheck)
endif()

# Build Tests
file(GLOB TEST_SOURCES "tests/*.cpp")
set(ALL_TEST_TARGETS "")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Special case for collision_test.cpp to keep it named test_collision
    if(TEST_NAME STREQUAL "collision_test")
        set(TEST_NAME "test_collision")
    endif()

    file(READ ${TEST_SOURCE} TEST_CONTENT)

    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE brassica)

    # Automatically detect and link GTest
    if(TEST_CONTENT MATCHES "gtest/gtest.h")
        target_link_libraries(${TEST_NAME} PRIVATE GTest::GTest)
        if(NOT TEST_CONTENT MATCHES "int main")
            target_link_libraries(${TEST_NAME} PRIVATE GTest::Main)
        endif()
    endif()

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        EXCLUDE_FROM_ALL TRUE
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    list(APPEND ALL_TEST_TARGETS ${TEST_NAME})
endforeach()

# Group all tests into a single target for convenience
add_custom_target(tests)
if(ALL_TEST_TARGETS)
    add_dependencies(tests ${ALL_TEST_TARGETS})
endif()
