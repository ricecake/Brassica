#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/ResourceKey.hpp"

namespace brassica::graph {

	class SetupBuilder;

	// A relation recorded by After<X>()/Before<X>()/Requires<X>() below -- inert bookkeeping for
	// now, per the design plan's staged rollout (§C/§E land in S3). Nothing in Graph::Compile
	// reads these yet; phase (kPhase, same convention as today) is still what actually orders a
	// builder node. Target identity is TypeName<X>() rather than a ResourceId/NodeHandle pointer:
	// a node declaring a relation to another node TYPE has no handle to that other node to point
	// at, only its name. A string compare is not the final identity scheme -- S3's real scheduler
	// needs something more robust (a stable registration-order-independent lookup) -- but this
	// shim only needs to compile and record intent for now, not act on it.
	struct NodeRelation {
		// Keystone is distinct from After/Before: it doesn't just order this node relative to the
		// target, it designates this node as *the* keystone of a phase marker (design plan's §D)
		// -- CompileRelational retroactively routes every other node In<> that same phase through
		// the keystone, so anything downstream can say After<TheKeystoneNode>() instead of naming
		// the marker. At most one keystone per marker; see RelationalCompiler.hpp.
		enum class Kind : std::uint8_t { After, Before, Requires, Keystone };
		Kind             kind;
		std::string_view targetName;
	};

	// The runtime counterpart of a node's compile-time `Resources` declaration (Declaration.hpp)
	// -- see the design plan's §B. A builder node declares each access by calling
	// Create/Read/Modify/ReadPrevious exactly where it used to hand-build a Recipe's realizations
	// vector by hand. Every recorded access carries access.desc == ResourceDesc{} deliberately
	// empty: PhysicalResourceRegistry::Provision already prefers a converted key's own catalog
	// Desc() over whatever a realization supplies (see ResourceCatalog.hpp, built in S1) over
	// whatever a realization's desc carries, so a builder node is never responsible for building
	// one itself -- that's the whole point of pairing this with the type-owned-desc model, not an
	// S2-specific shortcut.
	class SetupBuilder {
	public:
		// Returned by Create<K>()/Read<K>()/Modify<K>()/ReadPrevious<K>() below, to let a caller
		// chain per-access options onto the realization just recorded (the same options
		// ResourceRealization already carries) -- `r.Create<GBufferAlbedo>().Clear({0,0,0,0})`.
		// Holds an index into the owning SetupBuilder's realization vector, not a reference to
		// the element itself: a later Create/Read/Modify call in the same Setup() can reallocate
		// that vector, which would dangle a direct reference but leaves an index valid.
		class AccessHandle {
		public:
			AccessHandle(SetupBuilder& builder, std::size_t index): m_builder(builder), m_index(index) {}

			AccessHandle& Clear(std::array<float, 4> color) {
				m_builder.m_realizations[m_index].clearColor = color;
				return *this;
			}

			AccessHandle& ShadingRateTexelSize(std::array<std::uint32_t, 2> size) {
				m_builder.m_realizations[m_index].shadingRateTexelSize = size;
				return *this;
			}

			// Escape hatch for a key that can't go through the catalog yet (not converted) or
			// ever (AccelerationStructure -- Provision() switches on this exact realization's
			// own desc.kind to skip it entirely, before any catalog lookup runs, so an
			// AccelerationStructure access must set this explicitly or it's silently treated as
			// an empty Image2D instead). Prefer converting the key to the catalog over reaching
			// for this, when the key's shape allows it.
			AccessHandle& Desc(ResourceDesc desc) {
				m_builder.m_realizations[m_index].desc = desc;
				return *this;
			}

			// Asserts this write commutes with any other AnyOrder() write of the same key --
			// see ResourceRealization::anyOrder's own comment for what this does and does not
			// guarantee.
			AccessHandle& AnyOrder() {
				m_builder.m_realizations[m_index].anyOrder = true;
				return *this;
			}

			// Marks this Read access as load-bearing for this node's own activity -- see
			// ResourceRealization::required's own comment. Only meaningful on a Read; calling it
			// on a Create/Modify access is a no-op flag nothing checks (there's no "writer of my
			// own write" to look for).
			AccessHandle& Required() {
				m_builder.m_realizations[m_index].required = true;
				return *this;
			}

		private:
			SetupBuilder& m_builder;
			std::size_t   m_index;
		};

		SetupBuilder& Domain(ExecutionDomain domain) {
			m_domain = domain;
			return *this;
		}

		SetupBuilder& Active(bool active) {
			m_active = active;
			return *this;
		}

		template <ResourceRef K>
		AccessHandle Create() {
			return Record<K>(AccessKind::Write);
		}

		template <ResourceRef K>
		AccessHandle Read() {
			return Record<K>(AccessKind::Read);
		}

		template <ResourceRef K>
		AccessHandle Modify() {
			return Record<K>(AccessKind::ReadWrite);
		}

		// Sugar for Read<History<K>>() -- History<K> stays a key wrapper for now (Lifetime::
		// History exists only as a type-level classification today, nothing reads it yet -- see
		// ResourceKey.hpp).
		template <ResourceRef K>
		AccessHandle ReadPrevious() {
			return Record<History<K>>(AccessKind::Read);
		}

		template <typename X>
		SetupBuilder& After() {
			m_relations.push_back(NodeRelation{NodeRelation::Kind::After, TypeName<X>()});
			return *this;
		}

		template <typename X>
		SetupBuilder& Before() {
			m_relations.push_back(NodeRelation{NodeRelation::Kind::Before, TypeName<X>()});
			return *this;
		}

		template <typename X>
		SetupBuilder& Requires() {
			m_relations.push_back(NodeRelation{NodeRelation::Kind::Requires, TypeName<X>()});
			return *this;
		}

		// Sugar for After<P::Begin>() + Before<P::End>() -- "this node runs somewhere inside
		// phase P, ordered against everything else in P only by whatever else it declares."
		// Doesn't order same-phase nodes against each other, same as a raw kPhase never did.
		template <typename P>
		SetupBuilder& In() {
			After<typename P::Begin>();
			Before<typename P::End>();
			return *this;
		}

		// Marks this node as phase P's keystone -- see NodeRelation::Kind::Keystone's comment.
		// Implies In<P>() (the keystone is itself a member of the phase it closes). Targets
		// P::End directly (not P itself): that's the exact marker identity every other In<P>()
		// node already points at via Before<P::End>(), which is what lets the compiler stay
		// generic -- it never needs to know "P" is a phase, only that this node claims to be a
		// marker's sink.
		template <typename P>
		SetupBuilder& Keystone() {
			In<P>();
			m_relations.push_back(NodeRelation{NodeRelation::Kind::Keystone, TypeName<typename P::End>()});
			return *this;
		}

		[[nodiscard]] ExecutionDomain Domain() const { return m_domain; }

		[[nodiscard]] bool Active() const { return m_active; }

		[[nodiscard]] std::span<const ResourceRealization> Realizations() const { return m_realizations; }

		[[nodiscard]] std::span<const NodeRelation> Relations() const { return m_relations; }

	private:
		friend class AccessHandle;

		template <ResourceRef K>
		AccessHandle Record(AccessKind access) {
			m_realizations.push_back(ResourceRealization{.key = IdOf<K>(), .access = access});
			return AccessHandle(*this, m_realizations.size() - 1);
		}

		ExecutionDomain                  m_domain = ExecutionDomain::Graphics;
		bool                              m_active = true;
		std::vector<ResourceRealization> m_realizations;
		std::vector<NodeRelation>        m_relations;
	};

} // namespace brassica::graph
