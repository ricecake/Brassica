#include "doctest/doctest.h"
#include "Resources.hpp"

TEST_CASE("Resources Enums and Options Setup") {
	brassica::TextureOptions options{};
	options.format = vk::Format::eR8G8B8A8Unorm;
	options.depth = 1;
	options.mipLevels = 4;

	CHECK(options.format == vk::Format::eR8G8B8A8Unorm);
	CHECK(options.depth == 1);
	CHECK(options.mipLevels == 4);

	brassica::StagingResource staging{};
	staging.completionValue = 42;
	CHECK(staging.completionValue == 42);
}

TEST_CASE("GlobalDescriptorPool Options") {
	brassica::GlobalDescriptorPool::CapacityOptions opts{};
	opts.maxSets = 1000;
	opts.combinedImageSamplers = 2000;
	opts.allowUpdateAfterBind = true;

	CHECK(opts.maxSets == 1000);
	CHECK(opts.combinedImageSamplers == 2000);
	CHECK(opts.allowUpdateAfterBind == true);
}
