#pragma once

#include <memory>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>

#include "IManager.hpp"

namespace brassica {

	class ArgparseManager: public IManager {
	public:
		ArgparseManager();
		explicit ArgparseManager(const std::string& programName, const std::string& version = "1.0.0");
		~ArgparseManager() override = default;

		void Initialize() override;
		void Shutdown() override;

		bool Parse(int argc, char** argv);
		bool Parse(const std::vector<std::string>& args);

		[[nodiscard]] bool        GetHeadless() const;
		[[nodiscard]] uint32_t    GetMaxFrames() const;
		[[nodiscard]] std::string GetConfigFile() const;
		[[nodiscard]] std::string GetAppName() const;

		argparse::ArgumentParser&       GetParser() { return m_parser; }
		const argparse::ArgumentParser& GetParser() const { return m_parser; }

	private:
		void SetupArguments();

		argparse::ArgumentParser m_parser;
		bool                     m_argsParsed{false};
	};

} // namespace brassica
