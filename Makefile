BUILD_DIR = build
CONFIG = RelWithDebInfo
APP = sandbox

.PHONY: all clean format run clean-build test profile setup-deps graph-test graph-check

all:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --parallel

# Run once after pulling submodules
setup-deps:
	cd external/shaderc && ./utils/git-sync-deps

format:
	@cmake -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) --target format

format-shaders:
	@cmake -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) --target format-shaders

test:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --target tests
	@cd $(BUILD_DIR) && ctest --output-on-failure

run: all
	@./$(BUILD_DIR)/bin/$(APP)

profile:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) -DENABLE_PROFILING=ON
	@cmake --build $(BUILD_DIR) --parallel

clean-build:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --parallel --clean-first

clean:
	rm -rf $(BUILD_DIR)

# Include paths tests/graph/test_graph.cpp needs to parse on its own, without configuring the
# whole CMake project: the engine headers it pulls in (ArgparseManager/IManager/ManagerReflection/
# PhysicalResource/VulkanSeam/...) reach argparse, glm, VMA and real vulkan.hpp (found here via
# Homebrew's MoltenVK SDK, not a vendored copy -- adjust if Vulkan isn't installed at this path).
# This is NOT the "Vulkan-free" build the file's own header comment once promised: ResourceKeys.hpp
# and TerrainMapExporter.hpp (both genuinely used by later TEST_CASEs in this file) unconditionally
# include vulkan/vulkan.hpp, so there's no way to syntax-check the real file without it.
GRAPH_CXXFLAGS = -std=c++23 \
	-Iinclude \
	-Iexternal/entt/src \
	-Iexternal/doctest \
	-Iexternal/spdlog/include \
	-Iexternal/imgui \
	-Iexternal/argparse/include \
	-Iexternal/glm \
	-Iexternal/VulkanMemoryAllocator/include \
	-I/opt/homebrew/include

# Compile-time assertions only, no link/run step. Fastest inner loop while iterating on the
# declaration/validation layers.
graph-check:
	@$(CXX) $(GRAPH_CXXFLAGS) -fsyntax-only tests/graph/test_graph.cpp

# Builds and runs just the graph-layer test binary through the real CMake project (needed to
# actually run it, not just syntax-check it -- graph-check above is the fast no-link loop).
graph-test:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --target test_graph
	@cd $(BUILD_DIR) && ctest -R test_graph --output-on-failure
