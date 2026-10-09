#include "BallSystemHandler.hpp"
#include "CowSystemHandler.hpp"
#include "Engine.hpp"
#include "OzzCylinderSystemHandler.hpp"
#include <brassica.hpp>

int main(int argc, char** argv) {
	brassica::InitializeCore();

	brassica::EngineOptions options = brassica::EngineOptions::FromArgs(argc, argv);

	brassica::Engine engine;
	engine.AddSystemHandler<brassica::BallSystemHandler>();
	engine.AddSystemHandler<brassica::OzzCylinderSystemHandler>();
	engine.AddSystemHandler<brassica::CowSystemHandler>();
	engine.Init(options);
	engine.Run();
	engine.Cleanup();

	return 0;
}
