#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>

#include "graph/Graph.hpp"
#include "model/GltfModel.hpp"
#include "model/ModelInstanceManager.hpp"

using namespace brassica;

namespace {

	glm::mat4 SkinVertex(const GltfVertex& v, const glm::mat4* joints) {
		float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
		if (weightSum <= 0.0001f) {
			return glm::mat4(1.0f);
		}
		return v.weights.x * joints[v.joints.x] + v.weights.y * joints[v.joints.y] +
			v.weights.z * joints[v.joints.z] + v.weights.w * joints[v.joints.w];
	}

} // namespace

TEST_CASE("BakeClips lays out palette/clip-table offsets consistently") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));

	std::vector<glm::mat4>      palette;
	std::vector<ModelClipEntry> clipTable;
	model.BakeClips(120.0f, palette, clipTable);

	std::size_t jointCount = model.JointCount();
	REQUIRE(jointCount == 42);
	CHECK(clipTable.size() == model.Animations().size());

	// Rest pose occupies palette[0 .. jointCount).
	CHECK(palette.size() >= jointCount);

	std::size_t expectedBase = jointCount;
	for (const ModelClipEntry& entry : clipTable) {
		CHECK(entry.paletteBase == expectedBase);
		CHECK(entry.frameCount >= 1);
		expectedBase += static_cast<std::size_t>(entry.frameCount) * jointCount;
	}
	CHECK(palette.size() == expectedBase);
}

TEST_CASE("BakeClips' 120 Hz matrix-lerp blend stays close to the real continuous pose") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));

	std::vector<glm::mat4>      palette;
	std::vector<ModelClipEntry> clipTable;
	model.BakeClips(120.0f, palette, clipTable);

	std::size_t jointCount = model.JointCount();
	float       maxErrorAllClips = 0.0f;

	for (std::size_t clipIdx = 0; clipIdx < clipTable.size(); ++clipIdx) {
		const GltfAnimation&  anim = model.Animations()[clipIdx];
		const ModelClipEntry& entry = clipTable[clipIdx];
		if (anim.duration <= 0.0f || entry.frameCount < 2) {
			continue;
		}

		for (int sample = 0; sample < 11; ++sample) {
			float t = (static_cast<float>(sample) / 11.0f) * anim.duration;

			std::vector<glm::mat4> reference;
			model.EvaluatePose(static_cast<std::uint32_t>(clipIdx), t, reference);

			float frameF = (t * 1000.0f) / entry.frameDtMs;
			auto  f0 = static_cast<std::uint32_t>(frameF);
			f0 = std::min(f0, entry.frameCount - 2);
			float blend = frameF - static_cast<float>(f0);

			const glm::mat4* jointsA = &palette[entry.paletteBase + f0 * jointCount];
			const glm::mat4* jointsB = &palette[entry.paletteBase + (f0 + 1) * jointCount];

			for (const GltfVertex& v : model.Vertices()) {
				glm::mat4 skinRef = SkinVertex(v, reference.data());
				glm::mat4 skinA = SkinVertex(v, jointsA);
				glm::mat4 skinB = SkinVertex(v, jointsB);
				glm::mat4 skinBaked = (1.0f - blend) * skinA + blend * skinB;

				glm::vec3 posRef = glm::vec3(skinRef * glm::vec4(v.position, 1.0f));
				glm::vec3 posBaked = glm::vec3(skinBaked * glm::vec4(v.position, 1.0f));
				maxErrorAllClips = std::max(maxErrorAllClips, glm::length(posRef - posBaked));
			}
		}
	}

	// Stage 0's spike measured ~0.0041 (4.3% of the model's own 0.096-unit body diagonal) across
	// the 5 longest/fastest clips at 120 Hz -- this threshold is 0.01 (about 10%), comfortably
	// above the measured value but still tight enough to catch a real regression (e.g. a bake
	// rate change or a blend-math mistake).
	CHECK(maxErrorAllClips < 0.01f);
}

TEST_CASE("BakeClips' bounds sphere contains every baked frame's skinned vertex") {
	GltfModel model;
	REQUIRE(model.LoadFromFile("assets/cow.glb"));

	std::vector<glm::mat4>      palette;
	std::vector<ModelClipEntry> clipTable;
	model.BakeClips(120.0f, palette, clipTable);

	std::size_t jointCount = model.JointCount();
	for (const ModelClipEntry& entry : clipTable) {
		if (entry.boundsRadius <= 0.0f) {
			continue; // an unanimated/zero-duration clip never updated bounds from its init value
		}
		for (std::uint32_t f = 0; f < entry.frameCount; ++f) {
			const glm::mat4* joints = &palette[entry.paletteBase + f * jointCount];
			for (const GltfVertex& v : model.Vertices()) {
				if (v.weights.x + v.weights.y + v.weights.z + v.weights.w <= 0.0001f) {
					continue;
				}
				glm::mat4 skin = SkinVertex(v, joints);
				glm::vec3 pos = glm::vec3(skin * glm::vec4(v.position, 1.0f));
				CHECK(glm::length(pos - entry.boundsCenter) <= entry.boundsRadius + 0.0001f);
			}
		}
	}
}

TEST_CASE("ModelInstanceManager reuses despawned slots and coalesces per-slot writes") {
	ModelInstanceManager mgr(4);

	ModelInstanceHandle a = mgr.Spawn(0, glm::vec3(0.0f));
	ModelInstanceHandle b = mgr.Spawn(0, glm::vec3(1.0f));
	ModelInstanceHandle c = mgr.Spawn(0, glm::vec3(2.0f));
	ModelInstanceHandle d = mgr.Spawn(0, glm::vec3(3.0f));
	CHECK(a != kInvalidModelInstanceHandle);
	CHECK(b != kInvalidModelInstanceHandle);
	CHECK(c != kInvalidModelInstanceHandle);
	CHECK(d != kInvalidModelInstanceHandle);

	// Capacity exhausted.
	CHECK(mgr.Spawn(0, glm::vec3(4.0f)) == kInvalidModelInstanceHandle);

	mgr.Despawn(b);
	ModelInstanceHandle reused = mgr.Spawn(0, glm::vec3(5.0f));
	CHECK(reused == b); // LIFO free list: the most recently freed slot comes back first

	graph::Graph graph;
	mgr.RegisterBufferNodes(graph);
	// 4 spawns (a, c, d, reused) + 1 despawn (b) touched 5 distinct slots total across the whole
	// run, but b and reused are the *same* slot touched twice -- still 4 distinct dirty slots by
	// the time this first flush happens.
	CHECK(mgr.GetLastFlushCount() == 4);

	// Nothing pending now -- a second flush in the same "frame" shape sees zero.
	graph::Graph graph2;
	mgr.RegisterBufferNodes(graph2);
	CHECK(mgr.GetLastFlushCount() == 0);

	// Two SetClip calls on the same handle before a flush coalesce into one write, not two.
	mgr.SetClip(a, 1, 100, true);
	mgr.SetClip(a, 2, 200, false);
	graph::Graph graph3;
	mgr.RegisterBufferNodes(graph3);
	CHECK(mgr.GetLastFlushCount() == 1);
}
