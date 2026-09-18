#include "Engine.hpp"
#include <brassica.hpp>

int main(int argc, char** argv) {
	brassica::InitializeCore();

	brassica::EngineOptions options = brassica::EngineOptions::FromArgs(argc, argv);

	brassica::Engine engine;
	engine.Init(options);
	engine.Run();
	engine.Cleanup();

	return 0;
}
