#include "passes/model/ProceduralArchetypeCatalog.hpp"

#include "passes/material/MaterialManager.hpp"
#include "passes/model/ManifoldPalmAsset.hpp"
#include "passes/model/ManifoldPineAsset.hpp"
#include "passes/model/ManifoldRockAsset.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/ProceduralPlantGenerator.hpp"
#include "types/MaterialData.hpp"

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
			p.maxSegments = 18;       // Now represents depth. 18 segments is plenty.
			p.segmentLength = 0.3f;   // Longer runs between joints (18 * 0.8m = ~14.4m tall)
			p.directionJitterDegrees = 9.0f;
			p.branchChance = 0.32f;
			p.minBranchesPerNode = 2; // Reduced slightly to prevent exponential explosion
			p.maxBranchesPerNode = 3;
			p.trunkRadiusLow = 0.15f; // 1.2m thick base
			p.trunkRadiusHigh = 0.10f;
			p.radiusFalloffPerSegment = 0.945f;
			p.rigidityFalloffPerSegment = 0.04f;
			p.circularSegments = 10;
			p.maxJoints = 64;         // IMPORTANT: Must be raised so the tree isn't truncated
			return p;
		}

		PlantGenParams MediumTree() {
			PlantGenParams p;
			p.seed = 102;
			p.maxSegments = 18;       // Now represents depth. 18 segments is plenty.
			p.segmentLength = 0.4f;   // Longer runs between joints (18 * 0.8m = ~14.4m tall)
			p.directionJitterDegrees = 9.0f;
			p.branchChance = 0.32f;
			p.minBranchesPerNode = 2; // Reduced slightly to prevent exponential explosion
			p.maxBranchesPerNode = 3;
			p.trunkRadiusLow = 0.30f; // 1.2m thick base
			p.trunkRadiusHigh = 0.20f;
			p.radiusFalloffPerSegment = 0.945f;
			p.rigidityFalloffPerSegment = 0.04f;
			p.circularSegments = 10;
			p.maxJoints = 64;         // IMPORTANT: Must be raised so the tree isn't truncated
			return p;
		}

		PlantGenParams LargeTree() {
			PlantGenParams p;
			p.seed = 103;
			p.maxSegments = 18;       // Now represents depth. 18 segments is plenty.
			p.segmentLength = 0.8f;   // Longer runs between joints (18 * 0.8m = ~14.4m tall)
			p.directionJitterDegrees = 9.0f;
			p.branchChance = 0.32f;
			p.minBranchesPerNode = 2; // Reduced slightly to prevent exponential explosion
			p.maxBranchesPerNode = 3;
			p.trunkRadiusLow = 0.60f; // 1.2m thick base
			p.trunkRadiusHigh = 0.45f;
			p.radiusFalloffPerSegment = 0.945f;
			p.rigidityFalloffPerSegment = 0.04f;
			p.circularSegments = 10;
			p.maxJoints = 64;         // IMPORTANT: Must be raised so the tree isn't truncated
			return p;
		}

		// Pine: NOT a parametrization of BuildProceduralPlant's branching-chain generator --
		// sgreenhusted confirmed (after seeing it in practice) that trying to fake a conifer
		// silhouette out of the same branch-pitch mechanism trees/bushes use was fundamentally
		// the wrong shape, and asked for a real port of the pre-conversion look instead. See
		// ManifoldPineAsset.hpp's own comment: a plain tapered trunk (no branching at all) plus
		// stacked cone-shaped foliage tiers, matching the old GPU-emitted `emitPineTreeItem`
		// (shaders/foliage.mesh, deleted around commit 676d60d) silhouette directly.
		PineGenParams Pine() {
			PineGenParams p;
			p.seed = 501;
			p.trunkSegments = 4;
			p.trunkHeight = 3.6f;
			p.trunkRadiusLow = 0.26f;
			p.trunkRadiusHigh = 0.1f;
			p.trunkBaseRigidity = 0.95f;
			p.tierCount = 3;
			p.tierBaseRadius = 0.85f;
			p.tierRadiusFalloff = 0.70f;
			p.tierHeight = 2.15f;
			p.tierOverlap = 0.35f;
			p.tierBaseRigidity = 0.98f;
			p.circularSegments = 10;
			return p;
		}

		// Palm: NOT a parametrization of BuildProceduralPlant's branching-chain generator --
		// sgreenhusted clarified the earlier branchChance=0 trunk + big leaf-cluster preset read
		// as a diminutive tuft standing in for a frond crown, and asked for a real upgraded port
		// of the pre-conversion look instead (recognizable gently-curving trunk, long drooping
		// fronds), same reasoning Pine already got its own dedicated generator for. See
		// ManifoldPalmAsset.hpp's own comment: a round, quadratically-bending trunk plus
		// frondCount long tapering fronds, matching the old GPU-emitted `emitPalmTreeItem`
		// (shaders/foliage_emit.glsl, removed by commit dd4478871274e88ad87b18986978186fd25ed0f2)
		// silhouette directly.
		PalmGenParams Palm() {
			PalmGenParams p;
			p.seed = 502;
			p.trunkSegments = 5;
			p.trunkHeight = 12.0f;
			p.trunkRadiusLow = 0.32f;
			p.trunkRadiusHigh = 0.27f;
			p.trunkBendAmount = 0.8f;
			p.trunkBaseRigidity = 0.9f;
			p.frondCount = 5;
			p.frondLength = 2.7f;
			p.frondWidth = 0.75f;
			p.frondDroop = 0.9f;
			p.frondSegments = 4;
			p.frondRigidity = 0.5f;
			p.circularSegments = 10;
			return p;
		}

		// Cactus: thick, rigid, mostly-unbranched column (low branchChance) with a couple of
		// thick upward-curving arms (branchPitch skewed near-vertical, not horizontal) --
		// aiming for a saguaro-like silhouette. No leaves. True ribbing (per-angle radius
		// variation around the tube's circular cross-section) isn't supported by this
		// cylinder/curved-tube pipeline, so the "ribbed" part of the original ask is dropped
		// here, not faked -- flagged as a real scope cut, not a silent guess.
		PlantGenParams Cactus() {
			PlantGenParams p;
			p.seed = 503;
			p.maxSegments = 16;
			p.segmentLength = 0.35f;
			p.directionJitterDegrees = 3.0f;
			p.branchChance = 0.18f;
			p.minBranchesPerNode = 1;
			p.maxBranchesPerNode = 2;
			p.branchPitchMinDegrees = 15.0f;
			p.branchPitchMaxDegrees = 40.0f;
			p.trunkRadiusLow = 0.22f;
			p.trunkRadiusHigh = 0.19f;
			p.radiusFalloffPerSegment = 0.97f; // stays thick -- a column, not a tapering branch
			p.baseRigidity = 1.0f;
			p.rigidityFalloffPerSegment = 0.0f; // cacti don't sway
			p.circularSegments = 8;
			p.maxJoints = 14;
			return p;
		}

		// Bushes: short trunk, more branches per node, a much wider pitch range (closer to
		// horizontal than a tree's own mostly-vertical spread), higher branchChance (splits
		// sooner/more often) -- the same generator as Tree above, just parametrized bushier
		// rather than a separate mechanism (same reasoning the old presets already used).
		//
		// branchChance/maxBranchesPerNode turned down from an earlier 0.5/3-4 (A) and 0.5/3-5
		// (B), and segmentLength bumped from 0.14 to 0.18: sgreenhusted reported the bifurcation
		// logic producing a visually crowded bush, and a direct 100-seed sweep per preset showed
		// the old values compounding with back-to-back branch points (see
		// PlantGenParams::minSegmentsBetweenBranches) into real self-intersection -- 6-8
		// BuildCurvedSegment straight-fallback warnings per 100 generations. The
		// minSegmentsBetweenBranches cooldown alone got that down to ~1; this tuning (together
		// with the cooldown) gets it to 0 straight-fallbacks across 100 seeds each, and thins out
		// the branch density the "crowded" report was about in the first place.
		PlantGenParams BushVariantA() {
			PlantGenParams p;
			p.seed = 201;
			p.maxSegments = 20;
			p.segmentLength = 0.05f;
			p.directionJitterDegrees = 16.0f;
			p.branchChance = 0.35f;
			p.minBranchesPerNode = 2;
			p.maxBranchesPerNode = 3;
			p.branchPitchMinDegrees = 45.0f;
			p.branchPitchMaxDegrees = 85.0f;
			p.trunkRadiusLow = 0.2f;
			p.trunkRadiusHigh = 0.1f;
			p.radiusFalloffPerSegment = 0.85f;
			p.baseRigidity = 0.99f; // bushes sway more overall than a tree's own rigid trunk
			p.rigidityFalloffPerSegment = 0.05f;
			p.circularSegments = 8;
			p.maxJoints = 64;
			return p;
		}

		PlantGenParams BushVariantB() {
			PlantGenParams p = BushVariantA();
			p.seed = 202;
			p.maxSegments = 24;
			p.maxBranchesPerNode = 4;
			p.maxJoints = 64;
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

	ProceduralArchetypeIndices RegisterProceduralArchetypes(ModelRegistry& registry, MaterialManager& materialManager) {
		ProceduralArchetypeIndices indices;

		// Real materials instead of every procedural mesh defaulting to materialIndex 0 (flat
		// white, fully rough -- MaterialData's own default) -- one per archetype *kind* (all
		// three tree sizes share bark, all three rock sizes share rock color, etc.), since
		// BuildSkinnedManifoldAsset/BuildMeshlets only take one materialIndex for a whole asset
		// right now. Flower's color covers the whole stem+head, not just petals -- a real
		// simplification, flagged here rather than hidden.
		MaterialData barkMaterial;
		barkMaterial.baseColorFactor = glm::vec4(0.36f, 0.24f, 0.14f, 1.0f);
		barkMaterial.colorTintOffset.y = 0.9f; // roughness strength -- bark reads as matte
		// Opts bark into gltf.frag's vertical ridge/fissure noise (see MaterialRole). normal
		// blend stays partial (0.6) so the trunk's real macro curvature still reads through the
		// ridge detail instead of being fully replaced by it.
		barkMaterial.materialRole = static_cast<std::uint32_t>(MaterialRole::ProceduralBark);
		barkMaterial.noiseScale = 3.0f;
		barkMaterial.noiseWarp = 0.4f;
		barkMaterial.noiseSharpness = 0.25f;
		barkMaterial.proceduralShading.x = 0.6f;
		// c=0.5 (half a cosine cycle over t in [0,1]) so t=0 and t=1 land on opposite ends --
		// gltf.frag feeds this 1-ridgeMask, so grooves (dark end) and ridge crests (light end)
		// read as genuinely different shades, not the same color wrapping around.
		barkMaterial.palette[0] = glm::vec4(0.22f, 0.14f, 0.08f, 0.0f);
		barkMaterial.palette[1] = glm::vec4(0.12f, 0.08f, 0.05f, 0.0f);
		barkMaterial.palette[2] = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
		barkMaterial.palette[3] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
		std::uint32_t barkMaterialIndex = materialManager.RegisterMaterial(barkMaterial);

		MaterialData bushMaterial;
		bushMaterial.baseColorFactor = glm::vec4(0.22f, 0.35f, 0.14f, 1.0f); // mostly leaf mass, not bark
		bushMaterial.colorTintOffset.y = 0.8f;
		std::uint32_t bushMaterialIndex = materialManager.RegisterMaterial(bushMaterial);

		MaterialData rockMaterial;
		rockMaterial.baseColorFactor = glm::vec4(0.45f, 0.43f, 0.40f, 1.0f);
		rockMaterial.colorTintOffset.y = 0.95f;
		// Opts rock into gltf.frag's faceted Worley noise (see MaterialRole). normal blend is
		// close to fully procedural (0.8) -- unlike bark's subtler ridges, facets are meant to
		// be the dominant visual read on a rock's surface.
		rockMaterial.materialRole = static_cast<std::uint32_t>(MaterialRole::ProceduralRock);
		rockMaterial.noiseScale = 4.0f;
		rockMaterial.noiseWarp = 0.3f;
		rockMaterial.noiseSharpness = 0.12f;
		rockMaterial.proceduralShading.x = 0.8f;
		// Same c=0.5 half-cycle convention as bark's palette, so the seam (dark) and facet
		// interior (light) read as distinct greys instead of wrapping to the same shade.
		rockMaterial.palette[0] = glm::vec4(0.45f, 0.43f, 0.40f, 0.0f);
		rockMaterial.palette[1] = glm::vec4(0.12f, 0.12f, 0.12f, 0.0f);
		rockMaterial.palette[2] = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
		rockMaterial.palette[3] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
		std::uint32_t rockMaterialIndex = materialManager.RegisterMaterial(rockMaterial);

		MaterialData flowerMaterial;
		flowerMaterial.baseColorFactor = glm::vec4(0.85f, 0.35f, 0.55f, 1.0f);
		flowerMaterial.colorTintOffset.y = 0.5f;
		std::uint32_t flowerMaterialIndex = materialManager.RegisterMaterial(flowerMaterial);

		// A real leaf tuft (ManifoldLeafAsset.hpp) on Tree/Bush terminal tips, via
		// ManifoldJointSpec::materialIndex's per-joint override -- brighter/more saturated than
		// either bark or the bush's own muted stem green, so leaves read as distinct from the
		// wood they're attached to.
		MaterialData leafMaterial;
		leafMaterial.baseColorFactor = glm::vec4(0.25f, 0.55f, 0.16f, 1.0f);
		leafMaterial.colorTintOffset.y = 0.75f;
		// Opts this material into gltf.frag's procedural leaf-canopy noise (see MaterialRole).
		// scale/sharpness match the noise's own former hardcoded constants (LEAF_SCALE=7.5,
		// smoothstep width 0.14) and normal blend stays fully procedural (1.0) -- a straight
		// data-driven swap of gltf.frag's old unconditional-for-every-fragment behavior, not a
		// visual change. warp stays 0 (no domain warp applied at this scale yet).
		leafMaterial.materialRole = static_cast<std::uint32_t>(MaterialRole::ProceduralLeaf);
		leafMaterial.noiseScale = 7.5f;
		leafMaterial.noiseSharpness = 0.14f;
		leafMaterial.proceduralShading.x = 1.0f;
		// Green cosine palette (a + b*cos(2*PI*(c*t+d)), see evaluateCosinePalette) driving
		// per-leaf hue variance in gltf.frag instead of baseColorFactor's flat tint above, which
		// the ProceduralLeaf role now ignores in favor of this. t=0 reads as a fresh green,
		// t=0.5 as a deep green close to the old deepGreen/freshGreen split gltf.frag's
		// commented-out shadeLeaf() sketch used, with a small per-channel phase offset for a
		// little natural hue drift across the cycle instead of a flat brightness-only ramp.
		leafMaterial.palette[0] = glm::vec4(0.16f, 0.32f, 0.10f, 0.0f);
		leafMaterial.palette[1] = glm::vec4(0.09f, 0.14f, 0.06f, 0.0f);
		leafMaterial.palette[2] = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
		leafMaterial.palette[3] = glm::vec4(0.0f, 0.05f, 0.10f, 0.0f);
		std::uint32_t leafMaterialIndex = materialManager.RegisterMaterial(leafMaterial);

		// Darker, bluer than leafMaterial -- real conifer needles read noticeably less bright
		// than deciduous leaves -- and a grey-green for the cactus body, distinct from both.
		MaterialData needleMaterial;
		needleMaterial.baseColorFactor = glm::vec4(0.11f, 0.28f, 0.18f, 1.0f);
		needleMaterial.colorTintOffset.y = 0.8f;
		std::uint32_t needleMaterialIndex = materialManager.RegisterMaterial(needleMaterial);

		MaterialData cactusMaterial;
		cactusMaterial.baseColorFactor = glm::vec4(0.32f, 0.45f, 0.30f, 1.0f);
		cactusMaterial.colorTintOffset.y = 0.7f;
		std::uint32_t cactusMaterialIndex = materialManager.RegisterMaterial(cactusMaterial);

		PlantGenParams smallTree = SmallTree();
		smallTree.materialIndex = barkMaterialIndex;
		smallTree.enableLeaves = true;
		smallTree.leafChance = 1.0f;
		smallTree.leafSize = 0.625f;
		smallTree.leafCount = 5;
		smallTree.baseRigidity = 0.99f;
		smallTree.rigidityFalloffPerSegment = 0.00625f;
		smallTree.leafMaterialIndex = leafMaterialIndex;
		PlantGenParams mediumTree = MediumTree();
		mediumTree.materialIndex = barkMaterialIndex;
		mediumTree.enableLeaves = true;
		mediumTree.leafChance = 1.0f;
		mediumTree.leafSize = 1.25f;
		mediumTree.leafCount = 5;
		mediumTree.baseRigidity = 0.99f;
		mediumTree.rigidityFalloffPerSegment = 0.00625f;
		mediumTree.leafMaterialIndex = leafMaterialIndex;
		PlantGenParams largeTree = LargeTree();
		largeTree.materialIndex = barkMaterialIndex;
		largeTree.enableLeaves = true;
		largeTree.leafChance = 1.0f;
		largeTree.leafSize = 1.5f;
		largeTree.leafCount = 5;
		largeTree.leafMaterialIndex = leafMaterialIndex;
		largeTree.baseRigidity = 0.99f;
		largeTree.rigidityFalloffPerSegment = 0.00625f;

		PineGenParams pine = Pine();
		pine.materialIndex = barkMaterialIndex;
		pine.needleMaterialIndex = needleMaterialIndex;

		PalmGenParams palm = Palm();
		palm.materialIndex = barkMaterialIndex;
		palm.frondMaterialIndex = leafMaterialIndex;

		PlantGenParams cactus = Cactus();
		cactus.enableFlowers = true;
		cactus.materialIndex = cactusMaterialIndex;

		indices.treeBase = registry.RegisterAsset(BuildProceduralPlant(smallTree), "Small Tree");
		registry.RegisterAsset(BuildProceduralPlant(mediumTree), "Medium Tree");
		registry.RegisterAsset(BuildProceduralPlant(largeTree), "Large Tree");
		registry.RegisterAsset(BuildPineTree(pine), "Pine");
		registry.RegisterAsset(BuildPalmTree(palm), "Palm");
		registry.RegisterAsset(BuildProceduralPlant(cactus), "Cactus");

		PlantGenParams bushA = BushVariantA();
		bushA.materialIndex = bushMaterialIndex;
		bushA.enableLeaves = true;
		bushA.enableFlowers = true;
		bushA.leafChance = 0.85f;
		bushA.leafSize = 1.25f;
		bushA.leafCount = 4;
		bushA.baseRigidity = 0.99f;
		bushA.leafMaterialIndex = leafMaterialIndex;
		PlantGenParams bushB = BushVariantB();
		bushB.materialIndex = bushMaterialIndex;
		bushB.enableLeaves = true;
		bushB.enableFlowers = true;
		bushB.leafChance = 0.85f;
		bushB.leafSize = 1.75f;
		bushB.leafCount = 4;
		bushB.baseRigidity = 0.99f;
		bushB.leafMaterialIndex = leafMaterialIndex;
		indices.bushBase = registry.RegisterAsset(BuildProceduralPlant(bushA), "Bush A");
		registry.RegisterAsset(BuildProceduralPlant(bushB), "Bush B");

		RockGenParams smallRock = SmallRock();
		smallRock.materialIndex = rockMaterialIndex;
		RockGenParams mediumRock = MediumRock();
		mediumRock.materialIndex = rockMaterialIndex;
		RockGenParams largeRock = LargeRock();
		largeRock.materialIndex = rockMaterialIndex;
		indices.rockBase = registry.RegisterAsset(BuildProceduralRock(smallRock), "Small Rock");
		registry.RegisterAsset(BuildProceduralRock(mediumRock), "Medium Rock");
		registry.RegisterAsset(BuildProceduralRock(largeRock), "Large Rock");

		PlantGenParams smallFlower = SmallFlower();
		smallFlower.materialIndex = flowerMaterialIndex;
		PlantGenParams mediumFlower = MediumFlower();
		mediumFlower.materialIndex = flowerMaterialIndex;
		PlantGenParams largeFlower = LargeFlower();
		largeFlower.materialIndex = flowerMaterialIndex;
		indices.flowerBase = registry.RegisterAsset(BuildProceduralPlant(smallFlower), "Small Flower");
		registry.RegisterAsset(BuildProceduralPlant(mediumFlower), "Medium Flower");
		registry.RegisterAsset(BuildProceduralPlant(largeFlower), "Large Flower");

		indices.barkMaterialIndex = barkMaterialIndex;
		indices.bushMaterialIndex = bushMaterialIndex;
		indices.rockMaterialIndex = rockMaterialIndex;
		indices.flowerMaterialIndex = flowerMaterialIndex;
		indices.leafMaterialIndex = leafMaterialIndex;
		indices.needleMaterialIndex = needleMaterialIndex;
		indices.cactusMaterialIndex = cactusMaterialIndex;

		return indices;
	}

} // namespace brassica
