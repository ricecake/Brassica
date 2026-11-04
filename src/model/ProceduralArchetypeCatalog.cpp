#include "model/ProceduralArchetypeCatalog.hpp"

#include "model/ManifoldRockAsset.hpp"
#include "model/ModelRegistry.hpp"
#include "model/ProceduralPlantGenerator.hpp"

namespace brassica {

	namespace {

		// Trees: tall, mostly-vertical trunk, moderate branch spread, rigid trunk tapering to
		// flexible tips -- now via BuildProceduralPlant (curved branches via the Catmull-Rom
		// warp, real flared junctions, same single generator Bush below also uses with bushier
		// parameters) instead of the old BuildProceduralBranchingTree's straight segments.
		// maxSegments/segmentLength/branchChance here are this generator's own stand-ins for the
		// old maxDepth/trunkLength -- there's no direct 1:1 field mapping (growth here is one
		// short jittered segment at a time, not one branch-level at a time), so these presets
		// are a fresh tuning aimed at reproducing the old presets' own relative scale (small/
		// medium/large) rather than a mechanical field-for-field translation. Not yet confirmed
		// on real hardware at these specific values -- flag for a visual check.
		PlantGenParams SmallTree() {
			PlantGenParams p;
			p.seed = 101;
			p.maxSegments = 14;
			p.segmentLength = 0.35f;
			p.directionJitterDegrees = 10.0f;
			p.branchChance = 0.3f;
			p.minBranchesPerNode = 2;
			p.maxBranchesPerNode = 2;
			p.trunkRadiusLow = 0.14f;
			p.trunkRadiusHigh = 0.11f;
			p.radiusFalloffPerSegment = 0.92f;
			p.rigidityFalloffPerSegment = 0.07f;
			p.circularSegments = 10;
			p.maxJoints = 12;
			return p;
		}

		PlantGenParams MediumTree() {
			PlantGenParams p;
			p.seed = 102;
			p.maxSegments = 24;
			p.segmentLength = 0.42f;
			p.directionJitterDegrees = 10.0f;
			p.branchChance = 0.3f;
			p.minBranchesPerNode = 2;
			p.maxBranchesPerNode = 3;
			p.trunkRadiusLow = 0.22f;
			p.trunkRadiusHigh = 0.16f;
			p.radiusFalloffPerSegment = 0.94f;
			p.rigidityFalloffPerSegment = 0.045f;
			p.circularSegments = 10;
			p.maxJoints = 20;
			return p;
		}

		PlantGenParams LargeTree() {
			PlantGenParams p;
			p.seed = 103;
			p.maxSegments = 30;
			p.segmentLength = 0.48f;
			p.directionJitterDegrees = 9.0f;
			p.branchChance = 0.32f;
			p.minBranchesPerNode = 3;
			p.maxBranchesPerNode = 3;
			p.trunkRadiusLow = 0.30f;
			p.trunkRadiusHigh = 0.22f;
			p.radiusFalloffPerSegment = 0.945f;
			p.rigidityFalloffPerSegment = 0.04f;
			p.circularSegments = 10;
			p.maxJoints = 24;
			return p;
		}

		// Bushes: short trunk, more branches per node, a much wider pitch range (closer to
		// horizontal than a tree's own mostly-vertical spread), higher branchChance (splits
		// sooner/more often) -- the same generator as Tree above, just parametrized bushier
		// rather than a separate mechanism (same reasoning the old presets already used).
		PlantGenParams BushVariantA() {
			PlantGenParams p;
			p.seed = 201;
			p.maxSegments = 20;
			p.segmentLength = 0.14f;
			p.directionJitterDegrees = 16.0f;
			p.branchChance = 0.5f;
			p.minBranchesPerNode = 3;
			p.maxBranchesPerNode = 4;
			p.branchPitchMinDegrees = 45.0f;
			p.branchPitchMaxDegrees = 85.0f;
			p.trunkRadiusLow = 0.08f;
			p.trunkRadiusHigh = 0.06f;
			p.radiusFalloffPerSegment = 0.85f;
			p.baseRigidity = 0.6f; // bushes sway more overall than a tree's own rigid trunk
			p.rigidityFalloffPerSegment = 0.10f;
			p.circularSegments = 8;
			p.maxJoints = 18;
			return p;
		}

		PlantGenParams BushVariantB() {
			PlantGenParams p = BushVariantA();
			p.seed = 202;
			p.maxSegments = 24;
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

		indices.treeBase = registry.RegisterAsset(BuildProceduralPlant(SmallTree()));
		registry.RegisterAsset(BuildProceduralPlant(MediumTree()));
		registry.RegisterAsset(BuildProceduralPlant(LargeTree()));

		indices.bushBase = registry.RegisterAsset(BuildProceduralPlant(BushVariantA()));
		registry.RegisterAsset(BuildProceduralPlant(BushVariantB()));

		indices.rockBase = registry.RegisterAsset(BuildProceduralRock(SmallRock()));
		registry.RegisterAsset(BuildProceduralRock(MediumRock()));
		registry.RegisterAsset(BuildProceduralRock(LargeRock()));

		indices.flowerBase = registry.RegisterAsset(BuildProceduralPlant(SmallFlower()));
		registry.RegisterAsset(BuildProceduralPlant(MediumFlower()));
		registry.RegisterAsset(BuildProceduralPlant(LargeFlower()));

		return indices;
	}

} // namespace brassica
