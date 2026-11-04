#include "model/ProceduralArchetypeCatalog.hpp"

#include "model/ManifoldBranchingTreeAsset.hpp"
#include "model/ManifoldRockAsset.hpp"
#include "model/ModelRegistry.hpp"

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

		return indices;
	}

} // namespace brassica
