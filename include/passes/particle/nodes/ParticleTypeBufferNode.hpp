#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "graph/Node.hpp"
#include "passes/RenderPhases.hpp"
#include "render/NodeLifecycle.hpp"
#include "resources.hpp"
#include "types/Particle.hpp"

namespace brassica {

	// The sole producer of ParticleTypeBuffer -- previously a bare PredefinedBufferNode alias
	// with no owner anywhere in production (the particle cluster's 4 readers all declared their
	// own Read<ParticleTypeBuffer>(), but nothing ever registered a node to Create<> it, so the
	// real engine's graph was rejected outright the one time the cluster was actually enabled:
	// "nothing created the particle type buffer"). A derived type instead of a bare alias
	// specifically so a default-constructed instance (BRASSICA_REGISTER_NODE's own
	// NodeLifecycle<T>, same as every other flat-registered node here) self-seeds real data --
	// there's no per-type tuning a manager would ever mutate at runtime (unlike MaterialManager's
	// m_materials), so a one-time self-seed is simpler than threading a RegisterTypeBufferNode-
	// style call through Engine.cpp for data that never changes. `using Base::Base;` preserves
	// PredefinedBufferNode's own (initialData, desc) constructor for callers (tests) that want to
	// seed different data explicitly.
	//
	// Every entry is deliberately all-zero, not ParticleType's own non-zero in-class defaults:
	// particle.mesh's own fallback logic (`pType.color.a > 0.0 ? pType.color : p.color`,
	// `pType.size > 0.0 ? pType.size : <per-type hardcoded size>`) only kicks in when this
	// buffer's entry is zero -- a non-zero default here would silently override the already-
	// correct per-type color/size the liveness shader assigns per particle on respawn.
	// gravityScale/drag are declared in every particle shader's ParticleType mirror but never
	// actually indexed anywhere today (confirmed: only particle.mesh ever reads
	// uParticleTypes[p.type] at all), so zero is a correct value for them, not just a safe
	// placeholder.
	struct ParticleTypeBufferNode: graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare> {
		using Base = graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare>;
		using Base::Base;

		ParticleTypeBufferNode():
			Base(std::vector<ParticleType>(16, ParticleType{glm::vec4(0.0f), 0.0f, 0.0f, 0.0f, 0.0f})) {}

		// No GPU objects of its own to manage -- this node's only job is the one-time host
		// upload Base (HostWriteNode) already does via Setup/Execute. NodeLifecycle<T>
		// (render/NodeLifecycle.hpp) calls both unconditionally for any flat-registered node.
		void Init(const render::NodeServices&) {}
		void Destroy(vk::Device) {}
	};

	BRASSICA_REGISTER_NODE(ParticleTypeBufferNode);

} // namespace brassica
