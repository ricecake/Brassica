#pragma once

#include "graph/Execution.hpp"

// Home for this engine's own named render phases -- see graph::Phase's own comment
// (graph/Execution.hpp) for why the generic graph layer only defines the generic
// Early/Default/Late bands (plus PreviousFrame/NextFrame) and deliberately leaves anything more
// specific (e.g. DeferredShading, ForwardTranslucent) to whoever's building on top of it.
//
// Empty for now: every current node (GradientNode/TerrainNode/DeferredNode, the two atmosphere
// LUT nodes) runs at Phase::Default and relies purely on resource-dependency edges for ordering,
// not phase separation -- nothing here needs a name yet. Add one the same way any future node
// would opt in: a `static constexpr graph::Phase kPhase = SomeName;` on the node (see
// graph::HasPhase's comment, Node.hpp), paired with an
// `inline constexpr graph::Phase SomeName = graph::Phase::Late;` (or a custom int32_t value
// spaced away from the existing bands) here.

namespace brassica {

	// S4's 6-phase keystone backbone (design plan's §D) -- the replacement for the SubPhase int
	// bands below. SetupBuilder::In<P>() orders a node after P::Begin and before P::End. Most
	// nodes need none of this at all, because their real resource edges (or an explicit
	// relation) already order them correctly; a phase is for the nodes that genuinely have
	// nothing else pinning them down. A phase's keystone (SetupBuilder::Keystone<P>(), at most
	// one per phase) is the one real node everything downstream can name directly
	// (After<DeferredNode>()) instead of naming the marker.
	//
	// The 6 phases chain into each other with no explicit edge of their own, and
	// RelationalCompiler.hpp needs zero built-in knowledge of "phase" as a concept to make that
	// true: each phase's End *is* (type alias, same TypeName(), same marker identity) the next
	// phase's Begin. CompileRelational already auto-creates a virtual marker node the first time
	// any relation names it and resolves every later reference to the same one -- so a GBuffer
	// node's Before<GBuffer::End>() and a Lighting node's After<Lighting::Begin>() land on the
	// exact same marker without this header, or the generic compiler, ever stating "GBuffer
	// precedes Lighting" anywhere. Only 7 distinct marker identities exist for all 6 phases.
	//
	// Absorbs the 17 SubPhase bands below as: Prepare -> Prepare; GBuffer/Default -> GBuffer;
	// LightPreparation/GIComput/DeferredShading -> Lighting (keystone DeferredNode);
	// Reflection/Atmosphere/Underwater*/Water/Translucent/Particle -> Composite;
	// Exposure/BloomAndOptics/ToneMapping -> ToneMap (keystone TonemapNode);
	// PostProcessing/HudOverlay/MenuOverlay -> Overlay.
	namespace phase {
		namespace boundary {
			struct Start {};

			struct PrepareGBuffer {};

			struct GBufferLighting {};

			struct LightingComposite {};

			struct CompositeToneMap {};

			struct ToneMapOverlay {};

			struct End {};
		} // namespace boundary

		struct Prepare {
			using Begin = boundary::Start;
			using End = boundary::PrepareGBuffer;
		};

		struct GBuffer {
			using Begin = boundary::PrepareGBuffer;
			using End = boundary::GBufferLighting;
		};

		struct Lighting {
			using Begin = boundary::GBufferLighting;
			using End = boundary::LightingComposite;
		};

		struct Composite {
			using Begin = boundary::LightingComposite;
			using End = boundary::CompositeToneMap;
		};

		struct ToneMap {
			using Begin = boundary::CompositeToneMap;
			using End = boundary::ToneMapOverlay;
		};

		struct Overlay {
			using Begin = boundary::ToneMapOverlay;
			using End = boundary::End;
		};
	} // namespace phase

	namespace SubPhase {
		inline constexpr graph::Phase Prepare = graph::Phase(-500);
		inline constexpr graph::Phase GBuffer = graph::Phase(0);
		inline constexpr graph::Phase LightPreparation = graph::Phase(
			500
		); // Light Clustering & Froxel Volumetric Injection
		inline constexpr graph::Phase GIComput = graph::Phase(600);
		inline constexpr graph::Phase DeferredShading = graph::Phase(
			700
		); // Deferred Direct Lighting (Reads G-Buffer, Lights, GI, and Froxels)
		inline constexpr graph::Phase Reflection = graph::Phase(
			800
		); // SSR Compute & Composite (Reads G-Buffer, HZB, and Lit HDR)
		inline constexpr graph::Phase Atmosphere = graph::Phase(900); // Distant Volumetrics & Atmosphere Composite
		inline constexpr graph::Phase UnderwaterStructuralTranslucentRender = graph::Phase(1000);
		inline constexpr graph::Phase UnderwaterParticleRender = graph::Phase(1100);
		inline constexpr graph::Phase WaterRender = graph::Phase(1200);
		inline constexpr graph::Phase StructuralTranslucent = graph::Phase(1300);
		inline constexpr graph::Phase ParticleRender = graph::Phase(1400);
		inline constexpr graph::Phase Exposure = graph::Phase(1500);
		inline constexpr graph::Phase BloomAndOptics = graph::Phase(1600);
		inline constexpr graph::Phase ToneMapping = graph::Phase(1700);
		inline constexpr graph::Phase PostProcessing = graph::Phase(1800);
		inline constexpr graph::Phase HudOverlay = graph::Phase(2000);
		inline constexpr graph::Phase MenuOverlay = graph::Phase(3000);

	} // namespace SubPhase
} // namespace brassica
