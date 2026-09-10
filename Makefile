BUILD_DIR = build
CONFIG = RelWithDebInfo
APP = sandbox

GRAPH_CXXFLAGS = -std=c++23 -Wall -Wextra -Wpedantic -Iinclude -Iexternal/doctest -Iexternal/entt/src

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

# Frame graph skeleton (include/graph/): single TU, zero link deps, no CMake -- deliberate,
# since configuring the root project at all requires Vulkan (see CMakeLists.txt). The two
# -I flags are the Vulkan-free guarantee: there is no path by which a vulkan/ or fg/ include
# could resolve.
graph-test:
	@mkdir -p $(BUILD_DIR)/bin
	@$(CXX) $(GRAPH_CXXFLAGS) -o $(BUILD_DIR)/bin/test_graph tests/graph/test_graph.cpp
	@$(CXX) $(GRAPH_CXXFLAGS) -o $(BUILD_DIR)/bin/test_physical_backend tests/graph/test_physical_backend.cpp
	@$(BUILD_DIR)/bin/test_graph
	@$(BUILD_DIR)/bin/test_physical_backend

# Compile-time assertions only, no link/run step. Fastest inner loop while iterating on the
# declaration/validation layers.
graph-check:
	@$(CXX) $(GRAPH_CXXFLAGS) -fsyntax-only tests/graph/test_graph.cpp
