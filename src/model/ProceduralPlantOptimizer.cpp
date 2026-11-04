#include "model/ProceduralPlantOptimizer.hpp"

#include <map>

namespace brassica {

	void ProceduralPlantOptimizer::Optimize(PlantIR& ir) {
		ConsolidateTubes(ir);
		HandleJunctions(ir);
	}

	void ProceduralPlantOptimizer::ConsolidateTubes(PlantIR& ir) {
		if (ir.elements.empty()) {
			return;
		}

		bool changed = true;
		int  safetyCounter = 0;
		while (changed && safetyCounter < 10) {
			changed = false;
			++safetyCounter;
			std::vector<bool>          toRemove(ir.elements.size(), false);
			std::vector<PlantElement> controlPointsToAdd;

			int originalCount = static_cast<int>(ir.elements.size());
			for (int i = 0; i < originalCount; ++i) {
				if (toRemove[i]) {
					continue;
				}
				PlantElement& e1 = ir.elements[i];
				if (e1.type != PlantElementType::Tube || e1.children.size() != 1) {
					continue;
				}

				int           childIdx = e1.children[0];
				PlantElement& e2 = ir.elements[childIdx];
				if (e2.type != PlantElementType::Tube || toRemove[childIdx]) {
					continue;
				}

				glm::vec3 d1 = glm::normalize(e1.endPosition - e1.position);
				glm::vec3 d2 = glm::normalize(e2.endPosition - e2.position);
				float     alignment = glm::dot(d1, d2);

				if (alignment < 0.999f) {
					// Nearly but not perfectly collinear -- preserve the bend as a ControlPoint
					// instead of silently straightening it out.
					PlantElement cp;
					cp.type = PlantElementType::ControlPoint;
					cp.position = e1.endPosition;
					cp.radius = e1.endRadius;
					cp.parent = i;
					cp.children = e2.children;

					int cpIdx = static_cast<int>(ir.elements.size()) + static_cast<int>(controlPointsToAdd.size());
					for (int childOfChild : e2.children) {
						ir.elements[childOfChild].parent = cpIdx;
					}

					e1.endPosition = e2.endPosition;
					e1.endRadius = e2.endRadius;
					e1.children = {cpIdx};

					controlPointsToAdd.push_back(cp);
				} else {
					// Perfectly collinear -- simple merge, no ControlPoint needed.
					e1.endPosition = e2.endPosition;
					e1.endRadius = e2.endRadius;
					e1.children = e2.children;

					for (int childOfChild : e2.children) {
						ir.elements[childOfChild].parent = i;
					}

					toRemove[childIdx] = true;
					changed = true;
				}
			}

			for (const PlantElement& cp : controlPointsToAdd) {
				ir.elements.push_back(cp);
			}

			if (changed) {
				std::vector<PlantElement> newElements;
				std::map<int, int>        oldToNew;
				for (int i = 0; i < static_cast<int>(ir.elements.size()); ++i) {
					if (!toRemove[i]) {
						oldToNew[i] = static_cast<int>(newElements.size());
						newElements.push_back(ir.elements[i]);
					}
				}
				for (PlantElement& e : newElements) {
					if (e.parent != -1) {
						e.parent = oldToNew[e.parent];
					}
					for (int& child : e.children) {
						child = oldToNew[child];
					}
				}
				ir.elements = std::move(newElements);
			}
		}
	}

	void ProceduralPlantOptimizer::HandleJunctions(PlantIR& ir) {
		std::vector<PlantElement> hubsToAdd;

		for (int i = 0; i < static_cast<int>(ir.elements.size()); ++i) {
			PlantElement& e = ir.elements[i];
			if (e.children.size() <= 1) {
				continue;
			}
			if (e.type != PlantElementType::Tube && e.type != PlantElementType::ControlPoint) {
				continue;
			}

			glm::vec3 hubPos = (e.type == PlantElementType::Tube) ? e.endPosition : e.position;
			float     hubRadius = (e.type == PlantElementType::Tube) ? e.endRadius : e.radius;

			PlantElement hub;
			hub.type = PlantElementType::Hub;
			hub.position = hubPos;
			// 1.5x, not boidish's own 1.1x: that project's mesher draws every hub sphere
			// unconditionally (raw triangle-soup accumulation, no CSG dedup), so even a barely-
			// larger sphere was always fully visible. Brassica unions this via real CSG (Add),
			// which keeps only the outer envelope -- a 1.1x sphere sitting at a junction of
			// similarly-sized tubes can end up entirely inside their combined volume and
			// contribute zero visible surface (confirmed empirically: several hubs in
			// tests/test_procedural_plant_generator.cpp ended up with no surviving vertices at
			// 1.1x). 1.5x is a first, unverified-on-real-hardware guess at "generous enough to
			// actually poke out" -- flag for a visual check, not a proven-correct constant.
			hub.radius = hubRadius * 1.5f;
			hub.parent = i;
			hub.children = e.children;

			int hubIdx = static_cast<int>(ir.elements.size()) + static_cast<int>(hubsToAdd.size());
			for (int childIdx : e.children) {
				ir.elements[childIdx].parent = hubIdx;
			}
			e.children = {hubIdx};

			hubsToAdd.push_back(hub);
		}

		for (const PlantElement& hub : hubsToAdd) {
			ir.elements.push_back(hub);
		}
	}

} // namespace brassica
