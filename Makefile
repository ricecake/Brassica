BUILD_DIR = build
# CONFIG = Release
CONFIG = RelWithDebInfo

.PHONY: all clean format run clean-build

all:
	rm -rf $(BUILD_DIR)/shaders/
	cd external/shaderc && ./utils/git-sync-deps
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --parallel

# Bridges the wrapper to the CMake 'format' target we created earlier
format:
	@cmake -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) --target format

test:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --target tests
	@cd $(BUILD_DIR) && ctest --output-on-failure

run: all
	@./$(BUILD_DIR)/$(X)

profile:
	rm -rf $(BUILD_DIR)/shaders/
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) -DENABLE_PROFILING=ON
	@cmake --build $(BUILD_DIR) --parallel

clean-build:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)
	@cmake --build $(BUILD_DIR) --parallel --clean-first

clean:
	rm -rf $(BUILD_DIR)
