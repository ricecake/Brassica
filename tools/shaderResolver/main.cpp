#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include "spdlog/spdlog.h"

#include "Shader.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
	argparse::ArgumentParser program("shaderResolver", "1.0.0");

	program.add_argument("-i", "--input")
		.help("Path to input shader file or directory")
		.default_value(std::string(""));

	program.add_argument("-o", "--output")
		.help("Path to output directory or file")
		.default_value(std::string(""));

	program.add_argument("pos_input")
		.help("Positional input shader file or directory")
		.default_value(std::string(""))
		.remaining();

	try {
		program.parse_args(argc, argv);
	} catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	std::string inputPath = program.get<std::string>("-i");
	std::string outputPath = program.get<std::string>("-o");

	if (program.is_used("pos_input")) {
		auto posArgs = program.get<std::vector<std::string>>("pos_input");
		if (inputPath.empty() && !posArgs.empty()) {
			inputPath = posArgs[0];
		}
		if (outputPath.empty() && posArgs.size() > 1) {
			outputPath = posArgs[1];
		}
	}

	if (inputPath.empty() || outputPath.empty()) {
		spdlog::error(
			"Usage: shaderResolver -i <input_file_or_dir> -o <output_dir> (or shaderResolver <input> "
			"<output>)"
		);
		std::cerr << program;
		return 1;
	}

	fs::path inFsPath(inputPath);
	if (!fs::exists(inFsPath)) {
		spdlog::error("Input path does not exist: {}", inputPath);
		return 1;
	}

	bool   isSuccess = true;
	size_t successCount = 0;
	size_t failureCount = 0;

	if (fs::is_directory(inFsPath)) {
		for (const auto& entry : fs::recursive_directory_iterator(inFsPath)) {
			if (!entry.is_regular_file()) {
				continue;
			}

			fs::path filePath = entry.path();
			if (filePath.extension() == ".glsl") {
				spdlog::info("Skipping include file during directory scan: {}", filePath.string());
				continue;
			}

			if (brassica::Shader::ResolveShader(filePath.string(), outputPath)) {
				++successCount;
			} else {
				++failureCount;
				isSuccess = false;
			}
		}
	} else {
		if (brassica::Shader::ResolveShader(inputPath, outputPath)) {
			++successCount;
		} else {
			++failureCount;
			isSuccess = false;
		}
	}

	if (!isSuccess || successCount == 0) {
		spdlog::error(
			"Shader resolution completed with errors (Successes: {}, Failures: {})",
			successCount,
			failureCount
		);
		return 1;
	}

	spdlog::info("Shader resolution succeeded (Processed {} shader(s))", successCount);
	return 0;
}
