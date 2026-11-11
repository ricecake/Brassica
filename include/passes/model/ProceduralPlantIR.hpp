#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace brassica {

	// Flat intermediate representation for procedurally-grown plants, adapted from
	// external/boidish/include/procedural_ir.h -- ported down to only what a plant actually
	// needs (no Leaf/Puffball/Box/Wedge/Pyramid/Structure/Critter fields, no color/material/
	// skinning-mode metadata, since those belong to boidish's own raw-mesh pipeline, not
	// Brassica's Manifold-CSG one). A generator builds this, ProceduralPlantOptimizer simplifies
	// it in place, then ProceduralPlantGenerator converts the result into ManifoldJointSpecs.
	// Leaf (procedural plant generation v2's leaf-cluster follow-up) reuses the same
	// radius/variant slots Flower already does, just for a different customShape.
	enum class PlantElementType { Tube, Hub, Flower, Leaf, ControlPoint };

	struct PlantElement {
		PlantElementType type{PlantElementType::Tube};

		glm::vec3 position{0.0f};    // Tube: start. Hub/Flower/Leaf/ControlPoint: center.
		glm::vec3 endPosition{0.0f}; // Tube only: end.
		float     radius{0.0f};     // Tube: start radius. Hub: flare radius. Flower/Leaf: cluster size.
		float     endRadius{0.0f};  // Tube: end radius. Leaf: attaching tube's own tip radius (CSG margin).
		// 0 = fully flexible, 1 = rigid -- carried straight through to ManifoldJointSpec::rigidity
		// once this element becomes a joint.
		float rigidity{0.5f};
		int   variant{0}; // Flower: petal profile (Diamond/Bell/Star/Heart). Leaf: blade count.

		int              parent{-1};
		std::vector<int> children;
	};

	struct PlantIR {
		std::vector<PlantElement> elements;

		int AddTube(glm::vec3 start, glm::vec3 end, float startRadius, float endRadius, float rigidity, int parentIdx = -1) {
			PlantElement e;
			e.type = PlantElementType::Tube;
			e.position = start;
			e.endPosition = end;
			e.radius = startRadius;
			e.endRadius = endRadius;
			e.rigidity = rigidity;
			e.parent = parentIdx;
			return Append(e, parentIdx);
		}

		int AddHub(glm::vec3 position, float radius, int parentIdx = -1) {
			PlantElement e;
			e.type = PlantElementType::Hub;
			e.position = position;
			e.radius = radius;
			e.parent = parentIdx;
			return Append(e, parentIdx);
		}

		int AddFlower(glm::vec3 position, float radius, int variant, float rigidity, int parentIdx = -1) {
			PlantElement e;
			e.type = PlantElementType::Flower;
			e.position = position;
			e.radius = radius;
			e.variant = variant;
			e.rigidity = rigidity;
			e.parent = parentIdx;
			return Append(e, parentIdx);
		}

		int AddLeaf(glm::vec3 position, float size, int leafCount, float attachRadius, float rigidity, int parentIdx = -1) {
			PlantElement e;
			e.type = PlantElementType::Leaf;
			e.position = position;
			e.radius = size;
			e.endRadius = attachRadius;
			e.variant = leafCount;
			e.rigidity = rigidity;
			e.parent = parentIdx;
			return Append(e, parentIdx);
		}

		int AddControlPoint(glm::vec3 position, float radius, int parentIdx = -1) {
			PlantElement e;
			e.type = PlantElementType::ControlPoint;
			e.position = position;
			e.radius = radius;
			e.parent = parentIdx;
			return Append(e, parentIdx);
		}

	private:
		int Append(PlantElement e, int parentIdx) {
			int idx = static_cast<int>(elements.size());
			elements.push_back(std::move(e));
			if (parentIdx != -1 && parentIdx < idx) {
				elements[parentIdx].children.push_back(idx);
			}
			return idx;
		}
	};

} // namespace brassica
