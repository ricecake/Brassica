#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#include "graph/Execution.hpp"
#include "graph/ResourceKey.hpp"

namespace brassica::graph {

	// A type that has opted into owning its own ResourceDesc, the first piece of the resource-
	// type-ownership redesign: `ResourceDesc Desc(const FrameContext&) const;`. Most keys haven't
	// been converted yet (most of ResourceKeys.hpp is still a plain empty tag with no Desc at
	// all) -- that's fine, this concept and everything built on it below is additive. A converted
	// key with config fields (e.g. TerrainMinMaxTexture::gridDim) reads them directly off `this`
	// inside Desc(); there is no separate config parameter, since the catalog's whole point is
	// that config lives ON the one persistent instance it owns, not beside it.
	template <typename T>
	concept HasTypeOwnedDesc = requires(const T& t, const FrameContext& ctx) {
		{ t.Desc(ctx) } -> std::same_as<ResourceDesc>;
	};

	// Get<K> below is for a BASE key only -- History<K> borrows the base's one instance and
	// Desc() (DescFor's historyTarget redirect below), it never gets its own.
	template <typename K>
	concept CatalogKey = ResourceRef<K> && !IsHistory<K> && HasTypeOwnedDesc<K>;

	// One persistent instance per converted resource TYPE, holding whatever config an owning
	// manager publishes (e.g. TerrainManager setting TerrainMinMaxTexture::gridDim) plus the
	// type's own Desc(FrameContext). Persistent because the graph itself is rebuilt every frame
	// (Engine.cpp) but config and physical resources are not. Owned by PhysicalResourceRegistry
	// (see its Catalog() accessor) -- "beside" it, per the design plan, not inside Execution.hpp's
	// Vulkan-free seam, since nothing here is Vulkan-aware either way, it's just colocated with
	// the other persistent-across-frames registry state.
	class ResourceCatalog {
	public:
		// Returns the one persistent instance of K, default-constructing it on first touch. The
		// owning system (a node's Init, or a manager like TerrainManager) calls this to publish
		// config: `catalog.Get<TerrainMinMaxTexture>().gridDim = gridParams.x;`. Also what lets
		// DescFor below answer for K's id at all -- see its comment.
		template <CatalogKey K>
		K& Get() {
			const ResourceId id = IdOf<K>();
			if (auto it = m_instances.find(id); it != m_instances.end()) {
				return *static_cast<K*>(it->second.get());
			}
			auto      inst = std::make_shared<K>();
			K* const  raw = inst.get();
			m_descFns.emplace(id, [raw](const FrameContext& ctx) { return raw->Desc(ctx); });
			m_instances.emplace(id, std::move(inst));
			return *raw;
		}

		// Type-erased: non-nullopt iff some earlier Get<K>() call registered K's Desc(). A key no
		// production node has converted yet simply has no entry -- the caller (PhysicalRegistry)
		// falls back to whatever a legacy node realization supplied, same as before this catalog
		// existed. id may be a History<K> id: redirected to the base K's entry via
		// ResourceTypeInfo::historyTarget, since History<K> never gets its own Get<K>() instance
		// (CatalogKey excludes it) -- the ping-ponged pair is physically two instances of the SAME
		// desc, not two different types with two different descs.
		[[nodiscard]] std::optional<ResourceDesc> DescFor(ResourceId id, const FrameContext& ctx) const {
			const ResourceId lookupId = id->historyTarget ? id->historyTarget : id;
			auto              it = m_descFns.find(lookupId);
			if (it == m_descFns.end()) {
				return std::nullopt;
			}
			return it->second(ctx);
		}

	private:
		std::unordered_map<ResourceId, std::shared_ptr<void>>                     m_instances;
		std::unordered_map<ResourceId, std::function<ResourceDesc(const FrameContext&)>> m_descFns;
	};

} // namespace brassica::graph
