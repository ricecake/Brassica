#pragma once
#include <array>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>

#include "graph/TypeList.hpp"

namespace brassica::graph {

	// Resource identity is an empty, trivially-constructible tag struct: `struct GBufferAlbedo {};`.
	// The compiler enforces mismatches; there is no runtime representation to get wrong.
	template <typename T>
	concept ResourceKey = std::is_empty_v<T> && std::is_trivially_constructible_v<T>;

	// Wraps a key to mark it, at the type level, as inter-frame (last-frame) data.
	// History<T> is a distinct key from T throughout validation -- it is never unwrapped
	// during set algebra, so the PreviousFrame/NextFrame edge falls out of ordinary
	// reachability with no special-casing.
	template <ResourceKey T>
	struct History {
		using Target = T;
	};

	template <typename T>
	struct IsHistoryT: std::false_type {};

	template <typename T>
	struct IsHistoryT<History<T>>: std::true_type {};

	template <typename T>
	inline constexpr bool IsHistory = IsHistoryT<T>::value;

	template <typename T>
	struct HistoryTargetT {
		using type = T;
	};

	template <typename T>
	struct HistoryTargetT<History<T>> {
		using type = T;
	};

	// Unwraps History<K> to K; identity for a bare key. Needed only where the temporal
	// pairing must cross the History boundary (PreviousFrame/NextFrame invariant, and
	// eventually physical-resource lookup in last frame's pool).
	template <typename T>
	using HistoryTarget = typename HistoryTargetT<T>::type;

	// A resource reference is a bare key or a History-wrapped key. This is the constraint
	// used everywhere downstream, not ResourceKey directly.
	template <typename T>
	concept ResourceRef = ResourceKey<HistoryTarget<T>>;

	// Constexpr, compiler-specific-format type name for diagnostics. No RTTI.
	// WARNING: parses std::source_location::current().function_name(), whose format is
	// unspecified. The self-test below (kTypeNameParseIsSane) is not optional -- it is the
	// only thing standing between a compiler upgrade and every diagnostic silently
	// becoming "".
	template <typename T>
	constexpr std::string_view TypeName() {
		std::string_view full = std::source_location::current().function_name();
		auto             begin = full.find("T = ") + 4;
		auto             end = full.find_first_of(";]", begin);
		return full.substr(begin, end - begin);
	}

	namespace detail {
		struct TypeNameProbe {};
	} // namespace detail

	inline constexpr bool kTypeNameParseIsSane = TypeName<detail::TypeNameProbe>() ==
		"brassica::graph::detail::TypeNameProbe";

	static_assert(kTypeNameParseIsSane, "TypeName() parse is out of sync with this compiler's function_name() format");

	// Runtime identity for a resource key: the address of a unique inline-constexpr object
	// per key type (ODR-safe since C++17). Unique, stable, constexpr-comparable, and it
	// carries a human-readable name -- no RTTI, no counter of any kind.
	struct ResourceTypeInfo {
		std::string_view name;
		bool             isHistory;
	};

	template <ResourceRef K>
	inline constexpr ResourceTypeInfo kResourceTypeInfo{TypeName<K>(), IsHistory<K>};

	using ResourceId = const ResourceTypeInfo*;

	template <ResourceRef K>
	constexpr ResourceId IdOf() {
		return &kResourceTypeInfo<K>;
	}

	template <typename L>
	struct IdArrayT;

	template <typename... Ks>
	struct IdArrayT<TypeList<Ks...>> {
		static constexpr std::array<ResourceId, sizeof...(Ks)> value{IdOf<Ks>()...};
	};

	template <typename L>
	inline constexpr auto& kIdArray = IdArrayT<L>::value;

	// Lowers a compile-time TypeList<Ks...> to a runtime span of ResourceId, backed by
	// static storage. This is the mechanism by which type-erased nodes retain their
	// declared resource contract (see NodeHandle::Make in Node.hpp).
	template <typename L>
	constexpr std::span<const ResourceId> IdsOf() {
		return kIdArray<L>;
	}

} // namespace brassica::graph
