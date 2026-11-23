#include "passes/model/ProceduralPlantOptimizer.hpp"

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

				// 0.999 (~2.6 degrees) -- an earlier attempt loosened this to 0.96 (~16 degrees)
				// on the theory that dense per-step jitter was both over-detailing the curve and
				// making CurveIsSafeToWarp trip more easily. Measured empirically and both parts
				// were wrong: real branching presets (SmallTree/BushVariantA, frequent branching
				// via branchChance) have few or no multi-segment non-branching chains at all, so
				// threshold value doesn't change their waypoint count either way; and across
				// every preset that DOES have real chains (MediumTree/LargeTree/BushVariantB/
				// Pine/Cactus), a direct instrumented run at 0.999 showed 100% of the resulting
				// curves pass CurveIsSafeToWarp's current (already-fixed, per-frame-local-radius)
				// safety check -- zero straight-fallbacks. The actual cause of the original
				// "runs into the limit, falls back straight" report was CurveIsSafeToWarp's own
				// now-fixed global-max-radius bug, not this threshold. 0.96 only erased real
				// curve data that would otherwise have rendered correctly.
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

					// e2's own role (its span, its children) is now fully represented by e1's
					// extended span plus the new CP -- e2 itself must be removed, exactly like
					// the collinear-merge branch below already does for its own absorbed child.
					// Leaving e2 alive (the original bug here) meant it could survive to be
					// independently reprocessed as its own "e1" later in this same pass if its
					// own index came up again, creating a second, conflicting claim on its
					// children (both the new CP *and* e2 itself pointing at them) -- confirmed
					// via direct instrumentation to produce a real cycle back toward the tree's
					// root once enough passes compounded, which made ConvertToJoints loop
					// through a handful of elements over and over until maxJoints capped it
					// (visually: wildly duplicated/overlapping geometry, not a tuning issue).
					toRemove[childIdx] = true;
					changed = true;

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
			// toRemove was sized to ir.elements.size() *before* the append above -- a freshly-
			// added ControlPoint (never eligible for removal, it didn't exist when toRemove was
			// built) must get a real `false` entry here, or the renumbering loop below reads
			// toRemove[i] out of bounds for every one of these new indices. std::vector<bool>::
			// operator[] has no bounds check, so this was silent UB rather than a crash -- real
			// memory layout usually (but not guaranteedly) read back as 0/false, so this mostly
			// "worked," except when it didn't: a CP that happened to read back as `true` got
			// wrongly excluded from newElements, leaving it out of oldToNew entirely, so any
			// element still pointing at its old index via parent/children silently remapped to
			// oldToNew's operator[]-on-missing-key default of 0 -- i.e. the tree's own root,
			// fabricating a cycle back to element 0. That's what was producing many more final
			// ManifoldJointSpecs than the IR actually had elements (ConvertToJoints looping
			// through the cycle until maxJoints capped it) -- confirmed via direct
			// instrumentation, not guessed.
			toRemove.resize(ir.elements.size(), false);

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
			// 1.1x, down from an earlier 1.5x: sgreenhusted reported the flare spheres reading
			// as oversized, visually-disconnected "balls" at every branch point rather than a
			// seamless joint ("the hub ball being the same diameter as the tube they're
			// capping"). A real absorption risk motivated the original 1.5x (boidish's own
			// raw-triangle-soup mesher drew every hub sphere unconditionally; Brassica's real
			// CSG union (Add) keeps only the outer envelope, so an undersized sphere at a
			// junction can end up entirely inside the combined tube volume and contribute zero
			// visible surface -- a real, previously-confirmed bug class, not a hypothetical
			// one). Re-verified directly rather than trusting that old 1.1x-is-unsafe note at
			// face value (see feedback_verify_cited_precedent): a direct sweep checking, for
			// every real Hub joint across 337 instances (50 seeds each of SmallTree and
			// BushVariantA, the two catalog presets with real multi-child branch points), 1.2x/
			// 1.15x/1.1x all produced zero absorbed hubs; only 1.05x started losing a couple.
			// 1.1x is the smallest value this sweep actually confirmed safe, not a guess.
			hub.radius = hubRadius * 1.1f;
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
