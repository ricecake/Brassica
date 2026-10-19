#pragma once
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/Node.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/SetupBuilder.hpp"

namespace brassica::graph {

	// Spike B (see the design plan's §C) -- a shadow scheduler that replaces phase-partitioned
	// leveling with one global constraint DAG: order edges (phase lowered to one compat marker
	// per distinct int, plus explicit relations) closed under reachability, then resource edges
	// derived from that closure (replacing version lifting and self-modify chaining), then one
	// global leveling pass. Lives entirely side-by-side with Graph::Compile() -- nothing here is
	// wired into Schedule/Provision/the backend. Its only job is to answer Spike B's exit
	// criteria empirically before S3 commits to this design: zero cycles with WAR edges in, an
	// empty-where-expected ambiguity set, matching order on the known conflicting pairs, and
	// comfortably under budget on a real node count.

	constexpr std::size_t kMaxRelationalNodes = 256;
	using ReachSet = std::bitset<kMaxRelationalNodes>;

	enum class RelEdgeKind : std::uint8_t { Order, RAW, WAR, WAW };

	struct RelEdge {
		std::size_t from;
		std::size_t to;
		RelEdgeKind kind;
		std::string reason; // human-readable provenance, for cycle diagnostics
		ResourceId  key = nullptr; // non-null only for RAW/WAR/WAW edges -- what SynthesizeBarrier
		                           // needs; an Order edge (phase marker, explicit relation) is
		                           // pure scheduling constraint and barriers nothing by itself.
	};

	struct Ambiguity {
		ResourceId  key;
		std::size_t a;
		std::size_t b;
	};

	struct RelationalResult {
		bool                                  ok = false;
		std::string                           error; // cycle or other hard failure, empty if ok
		std::vector<Ambiguity>                ambiguities;
		std::vector<std::vector<std::size_t>> stages; // real-node indices only, markers/inactive contracted out
		std::vector<RelEdge>                  edges;  // every edge actually used (G0 + resource), for inspection
		// Final per-real-node activity (size == node count), after the Requires<A>/.Required()
		// fixed point (§E) -- a node's own declared Recipe::isActive, further turned off by
		// cascading activity coupling. The caller (Graph::Compile()) writes this back into its
		// own m_recipes[i].isActive so Provision/the backend/ZeroInitializeUndefinedReads all
		// see the resolved value, not just what each node's own Setup() declared in isolation.
		std::vector<bool>                     activity;
		// Every (node index, key) pair where that node's Read access of that key has no active
		// writer preceding it this frame (§E's "read whatever state K has") -- the backend's
		// generalization of ZeroInitializeUndefinedReads should clear exactly these, for images
		// and buffers alike, instead of only ever checking History<K> images by name.
		std::vector<std::pair<std::size_t, ResourceId>> zeroInitReads;
	};

	namespace detail {

		// Transient unless the key is catalog-registered with an explicit Lifetime -- same
		// default LifetimeOfK<K>() uses, just read from the already-erased ResourceId instead of
		// a compile-time K.
		inline bool IsPersistentKey(ResourceId key) { return key->lifetime == Lifetime::Persistent; }

	} // namespace detail

	// Takes the same two parallel spans Graph already exposes (Nodes()/Recipes()) -- this is
	// deliberately not a Graph method, so it can be dropped entirely without touching Graph.hpp.
	// All of a graph's accesses must already be real NodeHandles that have had Setup() called this
	// frame (same precondition Graph::Compile() has).
	inline RelationalResult CompileRelational(std::span<const NodeHandle> nodes, std::span<const Recipe> recipes) {
		RelationalResult result;
		const std::size_t n = nodes.size();
		if (n == 0) {
			result.ok = true;
			return result;
		}
		if (n > kMaxRelationalNodes / 2) {
			// Leaves half the id space for phase markers -- generous for today's ~40-node real
			// graph; a real implementation would size this dynamically, this spike doesn't need to.
			result.error = "CompileRelational: node count exceeds this spike's fixed bitset capacity";
			return result;
		}

		std::vector<NodeDescriptor> descriptors;
		descriptors.reserve(n);
		for (const auto& node : nodes) {
			descriptors.push_back(node.Descriptor());
		}

		// Resolve a key to its version base (VersionedKey<K,N-1>/<K,N> group under K, matching
		// Graph::AccessOf -- History<K> stays distinct from K, never unwrapped) and classify a
		// node's access to a resolved base purely from NodeDescriptor::consumes/produces. Built
		// from descriptors, not Recipe::realizations: realizations are authoritative for a
		// builder node (LowerBuilder derives both from the same SetupBuilder call, so they can't
		// disagree), but a legacy node's Resources/Recipe split is explicitly allowed to leave
		// realizations sparser than its declared contract -- the same reason Graph::AccessOf
		// itself never reads realizations.
		auto resolveBase = [](ResourceId id) { return id->versionBase ? id->versionBase : id; };
		auto accessOf = [&](std::size_t node, ResourceId base) {
			const auto& d = descriptors[node];
			const bool  reads =
				std::any_of(d.consumes.begin(), d.consumes.end(), [&](ResourceId id) { return resolveBase(id) == base; });
			const bool writes =
				std::any_of(d.produces.begin(), d.produces.end(), [&](ResourceId id) { return resolveBase(id) == base; });
			if (reads && writes) {
				return AccessKind::ReadWrite;
			}
			return writes ? AccessKind::Write : AccessKind::Read;
		};

		// Unlike access-kind classification above, AnyOrder() is a Recipe-level annotation with
		// no descriptor equivalent -- it can only be read from a node's own realizations. Defaults
		// to false for a legacy node whose Recipe is sparser than its declared contract (same
		// split accessOf's own comment explains): an opt-out has to be explicit, never inferred.
		auto anyOrderFor = [&](std::size_t node, ResourceId base) {
			for (const ResourceRealization& r : recipes[node].realizations) {
				if (resolveBase(r.key) == base) {
					return r.anyOrder;
				}
			}
			return false;
		};

		auto findByName = [&](std::string_view name) -> std::optional<std::size_t> {
			for (std::size_t i = 0; i < n; ++i) {
				if (descriptors[i].name == name) {
					return i;
				}
			}
			return std::nullopt;
		};

		// --- Phase compat markers: one per distinct phase value present, chained in ascending
		// phase order. Node i at the j-th distinct phase gets i->marker(j) and (j>0) marker(j-1)->i
		// -- see RelationalCompiler.hpp's own top comment for why this reproduces today's
		// phase-partition order without serializing same-phase nodes against each other.
		std::vector<Phase> distinctPhases;
		for (const auto& d : descriptors) {
			if (std::find(distinctPhases.begin(), distinctPhases.end(), d.phase) == distinctPhases.end()) {
				distinctPhases.push_back(d.phase);
			}
		}
		std::sort(distinctPhases.begin(), distinctPhases.end());

		const std::size_t phaseMarkerBase = n;
		const std::size_t numPhaseMarkers = distinctPhases.size();

		// --- Virtual relation-target markers (design plan's §D): a relation naming something
		// that isn't a registered node becomes an auto-created virtual node instead of being
		// dropped -- generically, with zero built-in knowledge of what a "phase" or "marker" is.
		// This is what makes the 6-phase keystone backbone (phase::X::Begin/End,
		// RenderPhases.hpp) work with no special-casing here: two different nodes' relations
		// naming the exact same unresolved type name land on the exact same auto-created marker,
		// so e.g. a GBuffer-phase node's Before<GBuffer::End>() and a Lighting-phase node's
		// After<Lighting::Begin>() -- GBuffer::End and Lighting::Begin being the same type via
		// RenderPhases.hpp's own aliasing -- transitively chain through it without this file ever
		// stating "GBuffer precedes Lighting" anywhere. A relation to a genuinely unregistered
		// node name (a typo, a forward-declared-but-never-built type) harmlessly becomes a marker
		// nothing else ever references, which contracts away to nothing -- same observable
		// behavior §E's "dropped, with a one-time warning" describes, just implemented uniformly.
		std::vector<std::string_view> virtualMarkerNames;
		auto                          virtualMarkerFor = [&](std::string_view name) -> std::size_t {
			auto it = std::find(virtualMarkerNames.begin(), virtualMarkerNames.end(), name);
			if (it != virtualMarkerNames.end()) {
				return phaseMarkerBase + numPhaseMarkers + static_cast<std::size_t>(it - virtualMarkerNames.begin());
			}
			virtualMarkerNames.push_back(name);
			return phaseMarkerBase + numPhaseMarkers + virtualMarkerNames.size() - 1;
		};
		for (std::size_t i = 0; i < n; ++i) {
			for (const NodeRelation& rel : nodes[i].Relations()) {
				if (!findByName(rel.targetName)) {
					virtualMarkerFor(rel.targetName); // pre-register so `total` below is final
				}
			}
		}

		const std::size_t total = n + numPhaseMarkers + virtualMarkerNames.size();
		if (total > kMaxRelationalNodes) {
			result.error = "CompileRelational: node+marker count exceeds this spike's fixed bitset capacity";
			return result;
		}

		auto markerFor = [&](Phase p) -> std::size_t {
			auto it = std::find(distinctPhases.begin(), distinctPhases.end(), p);
			return phaseMarkerBase + static_cast<std::size_t>(it - distinctPhases.begin());
		};

		// Readable name for provenance messages -- a real node uses its own descriptor name; a
		// phase-int marker has none, so it's named by the phase it represents; a virtual marker
		// is named by whatever relation target string created it.
		auto nameOf = [&](std::size_t idx) -> std::string {
			if (idx < n) {
				return std::string(descriptors[idx].name);
			}
			if (idx < phaseMarkerBase + numPhaseMarkers) {
				return "<phase " + std::to_string(static_cast<std::int32_t>(distinctPhases[idx - phaseMarkerBase])) +
					" marker>";
			}
			return "<marker " + std::string(virtualMarkerNames[idx - phaseMarkerBase - numPhaseMarkers]) + ">";
		};

		std::vector<std::vector<std::size_t>> succ(total);
		auto                                  addOrderEdge = [&](std::size_t from, std::size_t to, std::string reason) {
			if (from == to) {
				return;
			}
			succ[from].push_back(to);
			result.edges.push_back(RelEdge{from, to, RelEdgeKind::Order, std::move(reason)});
		};

		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t phaseIdx =
				static_cast<std::size_t>(std::find(distinctPhases.begin(), distinctPhases.end(), descriptors[i].phase) -
				                          distinctPhases.begin());
			addOrderEdge(i, markerFor(descriptors[i].phase), "phase marker");
			if (phaseIdx > 0) {
				addOrderEdge(markerFor(distinctPhases[phaseIdx - 1]), i, "phase marker");
			}
		}

		// --- Explicit relations (After<X>/Before<X>/Requires<X>/Keystone<X>). Requires<X> is
		// lowered to an After<X> order edge here -- the "inactive unless X is active" half of
		// Requires is S5's job (activity fixed point), not this one. A relation's target resolves
		// against real nodes first, then falls back to the virtual marker created for it above.
		std::vector<std::pair<std::size_t, std::size_t>> keystoneOf; // {markerIndex, keystoneNodeIndex}
		for (std::size_t i = 0; i < n; ++i) {
			for (const NodeRelation& rel : nodes[i].Relations()) {
				std::size_t target = 0;
				if (auto found = findByName(rel.targetName)) {
					target = *found;
				} else {
					target = virtualMarkerFor(rel.targetName);
				}
				switch (rel.kind) {
				case NodeRelation::Kind::After:
				case NodeRelation::Kind::Requires:
					addOrderEdge(target, i, "explicit relation");
					break;
				case NodeRelation::Kind::Before:
					addOrderEdge(i, target, "explicit relation");
					break;
				case NodeRelation::Kind::Keystone:
					for (const auto& [existingMarker, existingKeystone] : keystoneOf) {
						if (existingMarker == target && existingKeystone != i) {
							result.error = "CompileRelational: " + nameOf(target) +
								" already has a keystone (" + nameOf(existingKeystone) + "), cannot also give it " +
								nameOf(i);
							return result;
						}
					}
					keystoneOf.push_back({target, i});
					break;
				}
			}
		}

		// --- Keystone retroactive routing: a marker's keystone must be a true sink for that
		// marker, not just another participant pointing at it alongside everyone else -- so
		// After<TheKeystone>() gives the same "after every other In<P>() node too" guarantee
		// After<marker> would. For every other node X with an edge straight to the marker,
		// reroute it through the keystone instead (X -> keystone, replacing the direct edge to
		// the marker; the keystone's own edge to the marker, from its own In<P>(), stays).
		for (const auto& [marker, keystone] : keystoneOf) {
			for (std::size_t x = 0; x < total; ++x) {
				if (x == keystone) {
					continue;
				}
				if (std::find(succ[x].begin(), succ[x].end(), marker) == succ[x].end()) {
					continue;
				}
				std::erase(succ[x], marker);
				if (x != keystone && std::find(succ[x].begin(), succ[x].end(), keystone) == succ[x].end()) {
					succ[x].push_back(keystone);
					result.edges.push_back(
						RelEdge{x, keystone, RelEdgeKind::Order, "rerouted through phase keystone"}
					);
				}
			}
		}

		// --- Create(K) before every other accessor of K, for transient keys only -- a reader
		// legitimately running before a persistent K's Create is last frame's content (§C.1), not
		// an error, so persistent keys skip this rule entirely here. "Create" = produces K's base
		// without also consuming it (accessOf == Write). Two independent Creates of the same
		// transient key is suspicious (§C.1), but *not yet* decidable here: the closure this rule
		// itself feeds into hasn't been built yet, so there's no way to tell "two creators with
		// nothing ordering them" (a real ambiguity) apart from "two creators already ordered by
		// an explicit relation or the phase chain" (completely fine -- e.g. Import<K>, whose
		// Phase::PreviousFrame already precedes everything, and some ordinary same-phase-or-later
		// Write-only producer of the same key). Add no Create-before-accessor edges at all when
		// there isn't exactly one creator, and defer the real decision to the writer-pairwise
		// check below, which runs after the closure exists and already does exactly this
		// precedes-or-ambiguous-or-AnyOrder test for every other multi-writer case.
		{
			std::vector<ResourceId> allBases;
			for (const auto& d : descriptors) {
				for (ResourceId id : d.consumes) {
					allBases.push_back(resolveBase(id));
				}
				for (ResourceId id : d.produces) {
					allBases.push_back(resolveBase(id));
				}
			}
			std::sort(allBases.begin(), allBases.end());
			allBases.erase(std::unique(allBases.begin(), allBases.end()), allBases.end());

			for (ResourceId base : allBases) {
				if (detail::IsPersistentKey(base)) {
					continue;
				}
				std::vector<std::size_t> creators;
				std::vector<std::size_t> accessors;
				for (std::size_t i = 0; i < n; ++i) {
					const bool touches =
						std::any_of(
							descriptors[i].consumes.begin(),
							descriptors[i].consumes.end(),
							[&](ResourceId id) { return resolveBase(id) == base; }
						) ||
						std::any_of(descriptors[i].produces.begin(), descriptors[i].produces.end(), [&](ResourceId id) {
							return resolveBase(id) == base;
						});
					if (!touches) {
						continue;
					}
					accessors.push_back(i);
					if (accessOf(i, base) == AccessKind::Write) {
						creators.push_back(i);
					}
				}
				if (creators.size() == 1) {
					for (std::size_t j : accessors) {
						addOrderEdge(creators[0], j, "Create-before-accessor");
					}
				}
			}
		}

		// --- Closure: process in reverse topological order (Kahn's algorithm forward, then walk
		// the resulting order backwards), reach[i] = union of reach[s] | bit(s) over direct
		// successors s. G0 must be acyclic for this to terminate correctly -- checked by
		// confirming Kahn's algorithm placed every node.
		std::vector<std::size_t> indegree(total, 0);
		for (std::size_t i = 0; i < total; ++i) {
			for (std::size_t s : succ[i]) {
				++indegree[s];
			}
		}
		std::vector<std::size_t> topoOrder;
		{
			std::vector<std::size_t> frontier;
			for (std::size_t i = 0; i < total; ++i) {
				if (indegree[i] == 0) {
					frontier.push_back(i);
				}
			}
			std::vector<std::size_t> indegreeMut = indegree;
			while (!frontier.empty()) {
				std::sort(frontier.begin(), frontier.end());
				std::vector<std::size_t> next;
				for (std::size_t node : frontier) {
					topoOrder.push_back(node);
					for (std::size_t s : succ[node]) {
						if (--indegreeMut[s] == 0) {
							next.push_back(s);
						}
					}
				}
				frontier = std::move(next);
			}
		}
		if (topoOrder.size() != total) {
			std::vector<std::size_t> placed(topoOrder.begin(), topoOrder.end());
			std::string              detail;
			for (std::size_t i = 0; i < total; ++i) {
				if (std::find(placed.begin(), placed.end(), i) != placed.end()) {
					continue;
				}
				detail += "\n  " + nameOf(i) + " blocked by:";
				for (const RelEdge& e : result.edges) {
					if (e.to != i || std::find(placed.begin(), placed.end(), e.from) != placed.end()) {
						continue;
					}
					detail += ' ' + nameOf(e.from) + '(' + e.reason;
					if (e.key) {
						detail += " on ";
						detail += e.key->name;
					}
					detail += ')';
				}
			}
			result.error =
				"CompileRelational: G0 (phase markers + explicit relations, before any resource edge) "
				"itself contains a cycle -- most often a Create-before-accessor edge (a later-phase "
				"producer) directly contradicting the phase chain (an earlier-phase reader), the same "
				"real bug the old phase-violation error caught, just surfacing structurally instead of "
				"via a dedicated check now:" +
				detail;
			return result;
		}

		std::vector<ReachSet> reach(total);
		for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
			ReachSet r;
			for (std::size_t s : succ[*it]) {
				r.set(s);
				r |= reach[s];
			}
			reach[*it] = r;
		}
		auto precedes = [&](std::size_t a, std::size_t b) { return reach[a].test(b); };

		// --- Resource edges from the closure (§C.3). Keyed by each key's resolved base (so
		// VersionedKey<K,N-1>/VersionedKey<K,N> group under K, matching Graph::AccessOf's own
		// resolution and reproducing what version lifting replaces -- History<K> stays distinct
		// from K, same as everywhere else in this codebase). resolveBase/accessOf are the ones
		// defined up near `descriptors` above.
		struct Toucher {
			std::size_t node;
			AccessKind  access;
		};
		std::vector<std::pair<ResourceId, std::vector<Toucher>>> byKey;
		auto                                                      touchersFor = [&](ResourceId key) -> std::vector<Toucher>& {
			for (auto& [k, v] : byKey) {
				if (k == key) {
					return v;
				}
			}
			byKey.push_back({key, {}});
			return byKey.back().second;
		};
		for (std::size_t i = 0; i < n; ++i) {
			for (ResourceId id : descriptors[i].consumes) {
				touchersFor(resolveBase(id));
			}
			for (ResourceId id : descriptors[i].produces) {
				touchersFor(resolveBase(id));
			}
		}
		for (auto& [key, touchers] : byKey) {
			for (std::size_t i = 0; i < n; ++i) {
				const bool touches =
					std::any_of(
						descriptors[i].consumes.begin(),
						descriptors[i].consumes.end(),
						[&](ResourceId id) { return resolveBase(id) == key; }
					) ||
					std::any_of(descriptors[i].produces.begin(), descriptors[i].produces.end(), [&](ResourceId id) {
						return resolveBase(id) == key;
					});
				if (touches) {
					touchers.push_back(Toucher{i, accessOf(i, key)});
				}
			}
		}
		// Stable iteration order for default-edge insertion below: key name, then node name.
		std::sort(byKey.begin(), byKey.end(), [](const auto& a, const auto& b) {
			return std::string_view(a.first->name) < std::string_view(b.first->name);
		});

		// Records a RAW/WAR/WAW classification of an edge the closure already implies -- per
		// §C.3, "WAR R->W' to each writer it already precedes" means WAR edges are a
		// *classification* of an existing order, never a new constraint, so they can never
		// introduce a cycle by construction. This is also where every "already-ordered via G0"
		// writer pair's WAW edge comes from (SynthesizeBarrier needs the key regardless of
		// whether the order came from an explicit relation, Create-before-accessor, or anything
		// else) -- recorded unconditionally, never silently dropped into a no-op.
		auto recordImpliedEdge =
			[&](std::size_t from, std::size_t to, RelEdgeKind kind, ResourceId key, std::string reason) {
				result.edges.push_back(RelEdge{from, to, kind, std::move(reason), key});
			};

		// Adds a genuinely new constraint (only the "default: reader sees final state" case and
		// the AnyOrder tie-break need this -- RAW/WAR/already-ordered-WAW above are always
		// already implied). Mutates succ/reach and propagates the new reachability to every node
		// that already reached `from`; skipped outright if it would close a cycle, per §C.3.
		auto addResourceEdge =
			[&](std::size_t from, std::size_t to, RelEdgeKind kind, ResourceId key, std::string reason) {
				if (from == to || precedes(from, to)) {
					return; // already implied; nothing new to add
				}
				if (precedes(to, from)) {
					return; // would close a cycle -- §C.3's "skipped if it would close a cycle"
				}
				succ[from].push_back(to);
				reach[from].set(to);
				reach[from] |= reach[to];
				for (std::size_t p : topoOrder) {
					if (reach[p].test(from)) {
						reach[p] |= reach[from];
					}
				}
				result.edges.push_back(RelEdge{from, to, kind, std::move(reason), key});
			};

		for (auto& [key, touchers] : byKey) {
			std::vector<std::size_t> writers;
			std::vector<std::size_t> readers;
			for (const Toucher& t : touchers) {
				if (t.access == AccessKind::Write || t.access == AccessKind::ReadWrite) {
					writers.push_back(t.node);
				}
				if (t.access == AccessKind::Read || t.access == AccessKind::ReadWrite) {
					readers.push_back(t.node);
				}
			}
			// Writers must be pairwise ordered already (via G0's closure) -- an unordered pair is
			// a real ambiguity UNLESS both sides marked AnyOrder() for this key, in which case a
			// real edge still gets added (direction picked deterministically, by node name) --
			// AnyOrder() opts out of the *error*, never out of having a real edge/barrier. Two
			// same-stage Graphics nodes writing the same attachment-backed resource with no edge
			// at all between them is a genuine WAW hazard (ScheduleStage's own contract treats
			// stage-mates as "free to dispatch concurrently" -- confirmed empirically against
			// PhysicalExecutionBackend::RunSchedule, which issues one acquire barrier per stage,
			// not one per node-pair within it): "depth-tested opaque writes commute" means the
			// *visual result* doesn't depend on which one draws first, not that no dependency is
			// needed between two separate begin/endRendering instances touching the same images.
			for (std::size_t a = 0; a < writers.size(); ++a) {
				for (std::size_t b = a + 1; b < writers.size(); ++b) {
					if (precedes(writers[a], writers[b])) {
						recordImpliedEdge(writers[a], writers[b], RelEdgeKind::WAW, key, "writers already ordered");
						continue;
					}
					if (precedes(writers[b], writers[a])) {
						recordImpliedEdge(writers[b], writers[a], RelEdgeKind::WAW, key, "writers already ordered");
						continue;
					}
					if (anyOrderFor(writers[a], key) && anyOrderFor(writers[b], key)) {
						const bool aFirst = descriptors[writers[a]].name < descriptors[writers[b]].name;
						addResourceEdge(
							aFirst ? writers[a] : writers[b],
							aFirst ? writers[b] : writers[a],
							RelEdgeKind::WAW,
							key,
							"AnyOrder: deterministic tie-break"
						);
						continue;
					}
					result.ambiguities.push_back(Ambiguity{key, writers[a], writers[b]});
				}
			}
			for (std::size_t reader : readers) {
				// Among writers that already precede this reader, "the last writer" is the one not
				// itself preceded (within this subset) by another such writer -- well-defined once
				// pairwise order holds (ambiguous pairs already reported above; still deterministic
				// here, just picks the first ambiguity-survivor found).
				std::vector<std::size_t> precedingWriters;
				for (std::size_t w : writers) {
					if (w != reader && precedes(w, reader)) {
						precedingWriters.push_back(w);
					}
				}
				std::optional<std::size_t> lastWriter;
				for (std::size_t w : precedingWriters) {
					bool isLast = true;
					for (std::size_t w2 : precedingWriters) {
						if (w2 != w && precedes(w2, w)) {
							continue;
						}
						if (w2 != w && precedes(w, w2)) {
							isLast = false;
							break;
						}
					}
					if (isLast) {
						lastWriter = w;
						break;
					}
				}
				if (lastWriter) {
					recordImpliedEdge(*lastWriter, reader, RelEdgeKind::RAW, key, "last writer before reader");
				}
				for (std::size_t w : writers) {
					if (w != reader && precedes(reader, w)) {
						recordImpliedEdge(reader, w, RelEdgeKind::WAR, key, "reader before later writer");
					}
				}
				// Default edge: reader unordered against some writer at all (neither precedes the
				// other) -- "reads the final state" per §C.3, reproducing today's lifting. Inserted
				// in the already-established stable (key, node) order.
				for (std::size_t w : writers) {
					if (w == reader || precedes(w, reader) || precedes(reader, w)) {
						continue;
					}
					addResourceEdge(w, reader, RelEdgeKind::RAW, key, "default: reader sees final state");
				}
			}
		}

		// --- Activity + marker contraction (§C.4/§D): replace every inactive real node and every
		// marker with pred(X)*succ(X), so leveling only ever sees active real nodes. Markers are
		// never active-checked (they have no Recipe); a real node's isActive starts from its own
		// Recipe, then the fixed point below can additionally turn it off.
		std::vector<bool> isActive(total, true);
		for (std::size_t i = 0; i < n; ++i) {
			isActive[i] = recipes[i].isActive;
		}
		for (std::size_t i = n; i < total; ++i) {
			isActive[i] = false; // markers always contract
		}

		// Shared by the Requires<A>/.Required() fixed point below and the zero-init list after
		// it: does some node w, active per the `active` vector given, write `base` and satisfy
		// w <= reader in the schedule order established so far? "Write" here means Write or
		// ReadWrite (accessOf != Read) -- a plain reader of the same key is never a source of
		// defined content for another reader.
		//
		// A Persistent-lifetime key is exempt: "this frame's producer is inactive" is its normal,
		// expected state (a throttled LUT not regenerating because nothing changed, say), not
		// "there's no valid data" -- unlike a genuine feature toggle, where inactive really does
		// mean no contribution wanted. Create-before-accessor already carves out this same
		// exemption for persistent keys (§C.1); this is that same carve-out reaching the
		// activity/zero-init side too. Found the hard way: a real hardware test showed a stable
		// sun angle (so TransmittanceLUTNode/MultiScatteringLUTNode's own throttle correctly
		// stopped regenerating them) producing a persistently black sky, because
		// SkyViewLUTNode's plain Read<>() of both got flagged and zero-cleared every frame
		// regardless of the perfectly good cached content already sitting there.
		auto hasActiveWriterPreceding =
			[&](ResourceId base, std::size_t reader, const std::vector<bool>& active) {
				if (detail::IsPersistentKey(base)) {
					return true;
				}
				for (std::size_t w = 0; w < n; ++w) {
					if (w == reader || !active[w]) {
						continue;
					}
					const bool touchesBase =
						std::any_of(
							descriptors[w].consumes.begin(),
							descriptors[w].consumes.end(),
							[&](ResourceId id) { return resolveBase(id) == base; }
						) ||
						std::any_of(descriptors[w].produces.begin(), descriptors[w].produces.end(), [&](ResourceId id) {
							return resolveBase(id) == base;
						});
					if (!touchesBase || accessOf(w, base) == AccessKind::Read) {
						continue;
					}
					if (precedes(w, reader)) {
						return true;
					}
				}
				return false;
			};

		// --- Requires<A>/.Required() activity fixed point (§E, S5). Requires<A> couples this
		// node's activity to a specific *node* (A); .Required() on a Read<K> couples it to
		// whether *key* K actually has an active writer flowing into this read (a writer that
		// only runs after this node doesn't count -- W <= R, never the reverse, since activity
		// can't depend on something that hasn't happened yet in schedule order). Both can
		// cascade (a node deactivated this way might itself be depended on by another), so
		// iterate to a fixed point -- each pass only ever turns a node OFF, so this converges in
		// at most n passes. Runs after the full closure (G0 + every resource edge) is built,
		// since .Required()'s "W <= R" check needs the final precedes() relation, not just G0's.
		bool activityChanged = true;
		while (activityChanged) {
			activityChanged = false;
			for (std::size_t i = 0; i < n; ++i) {
				if (!isActive[i]) {
					continue;
				}
				bool deactivate = false;
				for (const NodeRelation& rel : nodes[i].Relations()) {
					if (rel.kind != NodeRelation::Kind::Requires) {
						continue;
					}
					if (auto target = findByName(rel.targetName); target && *target < n && !isActive[*target]) {
						deactivate = true;
						break;
					}
				}
				for (std::size_t ri = 0; !deactivate && ri < recipes[i].realizations.size(); ++ri) {
					const ResourceRealization& r = recipes[i].realizations[ri];
					if (r.required && r.access == AccessKind::Read &&
					    !hasActiveWriterPreceding(resolveBase(r.key), i, isActive)) {
						deactivate = true;
					}
				}
				if (deactivate) {
					isActive[i] = false;
					activityChanged = true;
				}
			}
		}
		result.activity.assign(isActive.begin(), isActive.begin() + n);

		// --- Zero-init list (§E, S5): "read whatever state K has". Unlike .Required() (opt-in,
		// deactivates the whole node), this applies to every plain Read, unconditionally -- a
		// node that reads a key with no active writer preceding it keeps running, it just needs
		// its content defined as zero instead of whatever garbage is physically there (or
		// whatever a since-deactivated producer left behind on a previous frame). ValidateRuntime
		// already guarantees every consumed key has at least one *registered* producer somewhere
		// (Import or node) -- this only flags the "registered but not active this frame" gap, so
		// a genuine typo/missing producer still fails there, not here.
		for (std::size_t i = 0; i < n; ++i) {
			if (!isActive[i]) {
				continue;
			}
			for (const ResourceRealization& r : recipes[i].realizations) {
				if (r.access != AccessKind::Read) {
					continue;
				}
				const ResourceId base = resolveBase(r.key);
				if (!hasActiveWriterPreceding(base, i, isActive)) {
					result.zeroInitReads.push_back({i, r.key});
				}
			}
		}

		std::vector<std::vector<std::size_t>> contractedSucc(total);
		for (std::size_t i = 0; i < total; ++i) {
			contractedSucc[i] = succ[i];
		}
		// Repeated single pass suffices here: process in topo order so a node's own successor
		// list is already free of earlier-contracted nodes by the time it's its turn.
		for (std::size_t node : topoOrder) {
			if (isActive[node]) {
				continue;
			}
			std::vector<std::size_t> expanded;
			for (std::size_t s : contractedSucc[node]) {
				expanded.push_back(s);
			}
			for (std::size_t other = 0; other < total; ++other) {
				auto& list = contractedSucc[other];
				bool  hadEdge = std::find(list.begin(), list.end(), node) != list.end();
				if (!hadEdge) {
					continue;
				}
				std::erase(list, node);
				for (std::size_t s : expanded) {
					if (s != other && std::find(list.begin(), list.end(), s) == list.end()) {
						list.push_back(s);
					}
				}
			}
			contractedSucc[node].clear();
		}

		// --- Global leveling over active real nodes only, ties broken by node name (never
		// registration index) per §C.5.
		std::vector<std::size_t> activeReal;
		for (std::size_t i = 0; i < n; ++i) {
			if (isActive[i]) {
				activeReal.push_back(i);
			}
		}

		std::vector<std::size_t> indeg(total, 0);
		std::vector<std::vector<std::size_t>> dependents(total);
		for (std::size_t i : activeReal) {
			std::vector<std::size_t> producers;
			for (std::size_t j : activeReal) {
				if (j != i && std::find(contractedSucc[j].begin(), contractedSucc[j].end(), i) != contractedSucc[j].end()) {
					producers.push_back(j);
				}
			}
			indeg[i] = producers.size();
			for (std::size_t p : producers) {
				dependents[p].push_back(i);
			}
		}

		std::vector<std::size_t> frontier;
		for (std::size_t i : activeReal) {
			if (indeg[i] == 0) {
				frontier.push_back(i);
			}
		}
		std::size_t placedCount = 0;
		while (!frontier.empty()) {
			std::sort(frontier.begin(), frontier.end(), [&](std::size_t a, std::size_t b) {
				return descriptors[a].name < descriptors[b].name;
			});
			result.stages.push_back(frontier);
			placedCount += frontier.size();
			std::vector<std::size_t> next;
			for (std::size_t node : frontier) {
				for (std::size_t dep : dependents[node]) {
					if (--indeg[dep] == 0) {
						next.push_back(dep);
					}
				}
			}
			frontier = std::move(next);
		}

		if (placedCount < activeReal.size()) {
			std::vector<std::size_t> stuck;
			for (std::size_t i : activeReal) {
				if (indeg[i] != 0) {
					stuck.push_back(i);
				}
			}
			std::string detail;
			for (std::size_t i : stuck) {
				detail += "\n  ";
				detail += descriptors[i].name;
				detail += " blocked by:";
				// Provenance (§C.8): every edge among the un-placed set pointing at this node,
				// with its reason -- G0 edges carry a reason string already; resource edges
				// (RAW/WAR/WAW) carry the key name instead, since "last writer before reader" on
				// its own doesn't say which resource.
				for (const RelEdge& e : result.edges) {
					if (e.to != i || std::find(stuck.begin(), stuck.end(), e.from) == stuck.end()) {
						continue;
					}
					detail += ' ';
					detail += descriptors[e.from].name;
					detail += '(';
					detail += e.reason;
					if (e.key) {
						detail += " on ";
						detail += e.key->name;
					}
					detail += ')';
				}
			}
			result.error =
				"CompileRelational: cycle in the combined order+resource graph; never reach zero "
				"remaining dependencies:" +
				detail;
			return result;
		}

		// --- Post-level assertion (§C.6): per stage, per key, either one writer and nothing else,
		// or every accessor is a plain read. Same descriptor-based access source as the resource
		// edges above, for the same reason (realizations can be sparser than the declared
		// contract for a still-legacy node).
		for (const auto& stage : result.stages) {
			std::vector<std::pair<ResourceId, AccessKind>> stageTouches;
			for (std::size_t node : stage) {
				std::vector<ResourceId> nodeBases;
				for (ResourceId id : descriptors[node].consumes) {
					nodeBases.push_back(resolveBase(id));
				}
				for (ResourceId id : descriptors[node].produces) {
					nodeBases.push_back(resolveBase(id));
				}
				std::sort(nodeBases.begin(), nodeBases.end());
				nodeBases.erase(std::unique(nodeBases.begin(), nodeBases.end()), nodeBases.end());
				for (ResourceId base : nodeBases) {
					stageTouches.push_back({base, accessOf(node, base)});
				}
			}
			std::sort(stageTouches.begin(), stageTouches.end(), [](const auto& a, const auto& b) {
				return std::string_view(a.first->name) < std::string_view(b.first->name);
			});
			for (std::size_t i = 0; i < stageTouches.size();) {
				std::size_t j = i;
				bool        anyWrite = false;
				std::size_t count = 0;
				while (j < stageTouches.size() && stageTouches[j].first == stageTouches[i].first) {
					if (stageTouches[j].second != AccessKind::Read) {
						anyWrite = true;
					}
					++count;
					++j;
				}
				if (anyWrite && count > 1) {
					result.error = "CompileRelational: same-stage conflicting access on '" +
						std::string(stageTouches[i].first->name) + "'";
					return result;
				}
				i = j;
			}
		}

		result.ok = true;
		return result;
	}

} // namespace brassica::graph
