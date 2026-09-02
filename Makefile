BUILD_DIR = build
CONFIG = RelWithDebInfo
APP = sandbox

.PHONY: all clean format run clean-build test profile setup-deps

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