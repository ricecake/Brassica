#pragma once

// Umbrella header pulling in every graph resource-key tag in the engine -- core cross-feature
// keys plus every render feature's own passes/<feature>/resources.hpp (design plan's
// restructure, S1). Intended for the small number of genuinely cross-cutting nodes that touch
// keys from three or more features (DeferredNode, TerrainNode, FoliageNode, the particle
// cluster, ...) -- a node that only touches one or two features' keys should include those
// specific headers directly instead of this umbrella.

#include "resources/core.hpp"

#include "passes/atmosphere/resources.hpp"
#include "passes/biome/resources.hpp"
#include "passes/clouds/resources.hpp"
#include "passes/entity/resources.hpp"
#include "passes/foliage/resources.hpp"
#include "passes/lighting/resources.hpp"
#include "passes/material/resources.hpp"
#include "passes/model/resources.hpp"
#include "passes/particle/resources.hpp"
#include "passes/postprocess/resources.hpp"
#include "passes/scene/resources.hpp"
#include "passes/terrain/resources.hpp"
#include "passes/volumetrics/resources.hpp"
