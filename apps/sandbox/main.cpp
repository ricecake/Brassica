#include "Engine.hpp"
#include <brassica.hpp>

int main() {
	brassica::InitializeCore();

	brassica::Engine engine;
	engine.Init();
	engine.Run();
	engine.Cleanup();

	return 0;
}
