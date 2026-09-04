#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "ShaderWatcher.hpp"

TEST_CASE("ShaderWatcher NormalizePath") {
	std::string rawPath = "shaders/../shaders/gradient.vert";
	std::string norm = brassica::ShaderWatcher::NormalizePath(rawPath);
	CHECK(!norm.empty());
	CHECK(norm.find('\\') == std::string::npos);
}

TEST_CASE("ShaderWatcher File Registration and Callback Invocation") {
	std::filesystem::path testDir = std::filesystem::current_path() / "test_shader_dir";
	std::filesystem::create_directories(testDir);

	std::filesystem::path shaderFile = testDir / "test.vert";
	{
		std::ofstream out(shaderFile);
		out << "// initial shader code\n";
	}

	brassica::ShaderWatcher watcher;
	REQUIRE(watcher.WatchDirectory(testDir.string()));

	bool callbackFired = false;
	watcher.RegisterFile(shaderFile.string(), [&]() {
		callbackFired = true;
	});

	// Wait briefly to ensure file watcher is active
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	// Modify the shader file
	{
		std::ofstream out(shaderFile, std::ios::app);
		out << "// modified line\n";
	}

	// Poll until callback fires or timeout occurs (up to 2 seconds)
	auto start = std::chrono::steady_clock::now();
	while (!callbackFired && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		watcher.ProcessPendingReloads(vk::Device{nullptr});
	}

	watcher.StopWatching();
	std::filesystem::remove_all(testDir);

	CHECK(callbackFired);
}

TEST_CASE("ShaderWatcher Included File Registration and Reload") {
	std::filesystem::path testDir = std::filesystem::current_path() / "test_inc_watch_dir";
	std::filesystem::create_directories(testDir);

	std::filesystem::path incFile = testDir / "common.glsl";
	{
		std::ofstream out(incFile);
		out << "// common glsl v1\n";
	}

	std::filesystem::path vertFile = testDir / "main.vert";
	{
		std::ofstream out(vertFile);
		out << "#version 460\n";
		out << "#include \"common.glsl\"\n";
		out << "void main() {}\n";
	}

	brassica::VertexShader shader;
	REQUIRE(shader.LoadFromFile(vertFile.string()));

	// Verify shader tracked the included file
	CHECK(shader.GetIncludedFiles().size() == 2);

	brassica::ShaderWatcher watcher;
	REQUIRE(watcher.WatchDirectory(testDir.string()));

	bool reloadFired = false;
	watcher.RegisterShader(&shader, [&]() {
		reloadFired = true;
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	// Modify the INCLUDED file (common.glsl)
	{
		std::ofstream out(incFile, std::ios::app);
		out << "// common glsl modified\n";
	}

	auto start = std::chrono::steady_clock::now();
	while (!reloadFired && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		watcher.ProcessPendingReloads(vk::Device{nullptr});
	}

	watcher.StopWatching();
	std::filesystem::remove_all(testDir);

	CHECK(reloadFired);
}
