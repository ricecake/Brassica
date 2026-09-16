#include "ArgparseManager.hpp"

#include "spdlog/spdlog.h"

namespace brassica {

	ArgparseManager::ArgparseManager() : m_parser("BrassicaEngine", "1.0.0") {}

	ArgparseManager::ArgparseManager(const std::string& programName, const std::string& version)
		: m_parser(programName, version) {}

	void ArgparseManager::SetupArguments() {
		m_parser.add_argument("--headless", "-headless")
			.help("Run engine in headless mode without window display")
			.default_value(false)
			.implicit_value(true);

		m_parser.add_argument("--frames", "-frames")
			.help("Maximum number of frames to run before exiting")
			.default_value(0)
			.scan<'i', int>();

		m_parser.add_argument("--config", "-config")
			.help("Path to configuration file")
			.default_value(std::string("config.ini"));

		m_parser.add_argument("--app", "-app")
			.help("Application name for scoped configurations")
			.default_value(std::string("Sandbox"));

		m_parser.add_argument("positional_frames")
			.help("Optional positional frame count")
			.default_value(std::vector<int>{})
			.scan<'i', int>()
			.nargs(0, 1);
	}

	void ArgparseManager::Initialize() {
		if (m_initialized)
			return;

		SetupArguments();
		m_initialized = true;
	}

	void ArgparseManager::Shutdown() {
		m_initialized = false;
	}

	bool ArgparseManager::Parse(int argc, char** argv) {
		Initialize();

		try {
			m_parser.parse_args(argc, argv);
			m_argsParsed = true;
			return true;
		} catch (const std::exception& err) {
			spdlog::warn("ArgparseManager failed to parse command line args: {}", err.what());
			return false;
		}
	}

	bool ArgparseManager::Parse(const std::vector<std::string>& args) {
		Initialize();

		try {
			m_parser.parse_args(args);
			m_argsParsed = true;
			return true;
		} catch (const std::exception& err) {
			spdlog::warn("ArgparseManager failed to parse command line args: {}", err.what());
			return false;
		}
	}

	bool ArgparseManager::GetHeadless() const {
		if (!m_argsParsed)
			return false;
		return m_parser.get<bool>("--headless");
	}

	uint32_t ArgparseManager::GetMaxFrames() const {
		if (!m_argsParsed)
			return 0;
		int frames = m_parser.get<int>("--frames");
		if (frames > 0) {
			return static_cast<uint32_t>(frames);
		}

		try {
			auto posFrames = m_parser.get<std::vector<int>>("positional_frames");
			if (!posFrames.empty() && posFrames[0] > 0) {
				return static_cast<uint32_t>(posFrames[0]);
			}
		} catch (...) {
		}

		return 0u;
	}

	std::string ArgparseManager::GetConfigFile() const {
		if (!m_argsParsed)
			return "config.ini";
		return m_parser.get<std::string>("--config");
	}

	std::string ArgparseManager::GetAppName() const {
		if (!m_argsParsed)
			return "Sandbox";
		return m_parser.get<std::string>("--app");
	}

} // namespace brassica
