#include "model/ProceduralArchetypeCatalog.hpp"

#include "model/ManifoldBranchingTreeAsset.hpp"
#include "model/ManifoldRockAsset.hpp"
#include "model/ModelRegistry.hpp"
#include "model/ProceduralPlantGenerator.hpp"

namespace brassica {

	namespace {

		// Trees: tall, mostly-vertical trunk, moderate branch spread, rigid trunk tapering to
		// flexible tips (TreeGenParams' own defaults already land close to "medium" -- these
		// three presets scale around that).
		TreeGenParams SmallTree() {
			TreeGenParams p;
			p.seed = 101;
			p.maxDepth = 2;
			p.minBranchesPerNode = 2;
			p.maxBranchesPerNode = 2;
			p.trunkLength = 1.6f;
			p.trunkRadiusLow = 0.14f;
			p.trunkRadiusHigh = 0.10f;
			p.maxJoints = 12;
			return p;
		}

		TreeGenParams MediumTree() {
			TreeGenParams p;
			p.seed = 102;
			p.maxDepth = 3;
			p.minBranchesPerNode = 2;
			p.maxBranchesPerNode = 3;
			p.trunkLength = 2.5f;
			p.trunkRadiusLow = 0.22f;
			p.trunkRadiusHigh = 0.16f;
			p.maxJoints = 20;
			return p;
		}

		TreeGenParams LargeTree() {
			TreeGenParams p;
			p.seed = 103;
			p.maxDepth = 3;
			p.minBranchesPerNode = 3;
			p.maxBranchesPerNode = 3;
			p.trunkLength = 3.3f;
			p.trunkRadiusLow = 0.30f;
			p.trunkRadiusHigh = 0.22f;
			p.maxJoints = 24;
			return p;
		}

		// Bushes: short trunk, more branches per node, a much wider pitch range (closer to
		// horizontal than a tree's own mostly-vertical spread), shallower depth -- the same
		// generator, just parametrized bushier rather than a separate mechanism.
		TreeGenParams BushVariantA() {
			TreeGenParams p;
			p.seed = 201;
			p.maxDepth = 2;
			p.minBranchesPerNode = 3;
			p.maxBranchesPerNode = 4;
			p.trunkLength = 0.35f;
			p.trunkRadiusLow = 0.08f;
			p.trunkRadiusHigh = 0.06f;
			p.lengthFalloff = 0.7f;
			p.radiusFalloff = 0.7f;
			p.pitchDegreesMin = 45.0f;
			p.pitchDegreesMax = 85.0f;
			p.baseRigidity = 0.6f; // bushes sway more overall than a tree's own rigid trunk
			p.rigidityFalloffPerLevel = 0.25f;
			p.maxJoints = 18;
			return p;
		}

		TreeGenParams BushVariantB() {
			TreeGenParams p = BushVariantA();
			p.seed = 202;
			p.trunkLength = 0.5f;
			p.maxBranchesPerNode = 5;
			p.maxJoints = 22;
			return p;
		}

		RockGenParams SmallRock() {
			RockGenParams p;
			p.seed = 301;
			p.baseRadius = 0.35f;
			p.clusterCount = 1;
			p.noiseAmplitude = 0.25f;
			return p;
		}

		RockGenParams MediumRock() {
			RockGenParams p;
			p.seed = 302;
			p.baseRadius = 0.6f;
			p.clusterCount = 2;
			p.noiseAmplitude = 0.3f;
			return p;
		}

		RockGenParams LargeRock() {
			RockGenParams p;
			p.seed = 303;
			p.baseRadius = 0.9f;
			p.clusterCount = 3;
			p.noiseAmplitude = 0.35f;
			return p;
		}

		// Flowers: a short, unbranched stem capped with a real flower head (ManifoldFlowerAsset.hpp
		// via ProceduralPlantGenerator's customShape hook), not a bare disembodied head -- a
		// placeable decor archetype needs a stem to root it to the ground the same way tree/bush
		// do. branchChance=0 since a single flower doesn't need side branches; flowerChance=1
		// guarantees the stem's one terminal tip actually grows a head.
		PlantGenParams SmallFlower() {
			PlantGenParams p;
			p.seed = 401;
			p.maxSegments = 4;
			p.segmentLength = 0.10f;
			p.directionJitterDegrees = 8.0f;
			p.branchChance = 0.0f;
			p.trunkRadiusLow = 0.012f;
			p.trunkRadiusHigh = 0.008f;
			p.baseRigidity = 0.3f; // flowers sway more than a woody stem
			p.rigidityFalloffPerSegment = 0.05f;
			p.circularSegments = 8;
			p.maxJoints = 6;
			p.enableFlowers = true;
			p.flowerChance = 1.0f;
			p.flowerSize = 0.10f;
			return p;
		}

		PlantGenParams MediumFlower() {
			PlantGenParams p = SmallFlower();
			p.seed = 402;
			p.maxSegments = 5;
			p.segmentLength = 0.14f;
			p.flowerSize = 0.15f;
			return p;
		}

		PlantGenParams LargeFlower() {
			PlantGenParams p = SmallFlower();
			p.seed = 403;
			p.maxSegments = 6;
			p.segmentLength = 0.18f;
			p.flowerSize = 0.22f;
			return p;
		}

	} // namespace

	ProceduralArchetypeIndices RegisterProceduralArchetypes(ModelRegistry& registry) {
		ProceduralArchetypeIndices indices;

		indices.treeBase = registry.RegisterAsset(BuildProceduralBranchingTree(SmallTree()));
		registry.RegisterAsset(BuildProceduralBranchingTree(MediumTree()));
		registry.RegisterAsset(BuildProceduralBranchingTree(LargeTree()));

		indices.bushBase = registry.RegisterAsset(BuildProceduralBranchingTree(BushVariantA()));
		registry.RegisterAsset(BuildProceduralBranchingTree(BushVariantB()));

		indices.rockBase = registry.RegisterAsset(BuildProceduralRock(SmallRock()));
		registry.RegisterAsset(BuildProceduralRock(MediumRock()));
		registry.RegisterAsset(BuildProceduralRock(LargeRock()));

		indices.flowerBase = registry.RegisterAsset(BuildProceduralPlant(SmallFlower()));
		registry.RegisterAsset(BuildProceduralPlant(MediumFlower()));
		registry.RegisterAsset(BuildProceduralPlant(LargeFlower()));

		return indices;
	}

} // namespace brassica
