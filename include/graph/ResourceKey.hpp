#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>

#include "graph/TypeList.hpp"
#include <entt/core/type_info.hpp>

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

	// Wraps a key to mark it, at the type level, as a specific version in a write chain --
	// e.g. two nodes both writing Swapchain in sequence produce VersionedKey<Swapchain, 1> and
	// VersionedKey<Swapchain, 2>. Like History<T>, never unwrapped during set algebra (Concat/
	// Dedup/Difference), so a write chain's ordering falls out of ordinary reachability with no
	// special-casing in TypeList.hpp or Validation.hpp.
	template <ResourceKey K, std::size_t N>
	struct VersionedKey {};

	template <typename T>
	struct IsVersionedT: std::false_type {};

	template <typename K, std::size_t N>
	struct IsVersionedT<VersionedKey<K, N>>: std::true_type {};

	template <typename T>
	inline constexpr bool IsVersioned = IsVersionedT<T>::value;

	// VersionOf<K, 0> is K itself -- version 0 is whatever Create<K>/an import already produces,
	// not a distinct type. This is what lets Modify<K> (no version) and Import<K> keep working
	// untouched: they are indistinguishable from Modify<K, 0>.
	template <typename K, std::size_t N>
	struct VersionOfT {
		using type = VersionedKey<K, N>;
	};

	template <typename K>
	struct VersionOfT<K, 0> {
		using type = K;
	};

	template <typename K, std::size_t N>
	using VersionOf = typename VersionOfT<K, N>::type;

	// Unwraps VersionedKey<K, N> to K; identity for anything else. Named to mirror HistoryTarget:
	// VersionBase<T> is "what resource is this a version of", the same shape as "what resource is
	// this history of".
	template <typename T>
	struct VersionBaseT {
		using type = T;
	};

	template <typename K, std::size_t N>
	struct VersionBaseT<VersionedKey<K, N>> {
		using type = K;
	};

	template <typename T>
	using VersionBase = typename VersionBaseT<T>::type;

	template <typename T>
	struct VersionIndexT: std::integral_constant<std::size_t, 0> {};

	template <typename K, std::size_t N>
	struct VersionIndexT<VersionedKey<K, N>>: std::integral_constant<std::size_t, N> {};

	// The N in VersionedKey<K, N>; 0 for anything that isn't a VersionedKey (including K itself,
	// which is correct -- VersionOf<K, 0> collapses to K).
	template <typename T>
	inline constexpr std::size_t VersionIndexOf = VersionIndexT<T>::value;

	// A resource reference is a bare key, a History-wrapped key, or a versioned key. This is the
	// constraint used everywhere downstream, not ResourceKey directly.
	template <typename T>
	concept ResourceRef = ResourceKey<HistoryTarget<VersionBase<T>>>;

	// Constexpr, compiler-specific-format type name for diagnostics. No RTTI.
	// WARNING: parses std::source_location::current().function_name(), whose format is
	// unspecified. The self-test below (kTypeNameParseIsSane) is not optional -- it is the
	// only thing standing between a compiler upgrade and every diagnostic silently
	// becoming "".
	// template <typename T>
	// constexpr std::string_view TypeName() {
	// #if defined(__clang__)
	// 	constexpr std::string_view prefix = "std::string_view TypeName() [T = ";
	// 	constexpr std::string_view suffix = "]";
	// 	constexpr std::string_view sig = __PRETTY_FUNCTION__;
	// #elif defined(__GNUC__)
	// 	constexpr std::string_view prefix = "constexpr std::string_view TypeName() [with T = ";
	// 	constexpr std::string_view suffix = "]";
	// 	constexpr std::string_view sig = __PRETTY_FUNCTION__;
	// #elif defined(_MSC_VER)
	// 	constexpr std::string_view prefix = "class std::basic_string_view<char,struct std::char_traits<char> > __cdecl
	// TypeName<"; 	constexpr std::string_view suffix = ">(void)"; 	constexpr std::string_view sig = __FUNCSIG__; #else
	// 	#error "Unsupported compiler"
	// #endif

	// // 1. Calculate bounds using prefix/suffix lengths
	// // 2. Extract substring
	// // 3. (MSVC only) strip "struct " / "class " from the start of the substring
	// 	auto             begin = full.find("T = ");
	// 	auto             end = full.find_first_of(";]", begin);
	// 	return full.substr(begin, end - begin);
	// }

	constexpr std::string_view StripTags(std::string_view name) {
		if (name.starts_with("struct "))
			return name.substr(7);
		if (name.starts_with("class "))
			return name.substr(6);
		if (name.starts_with("enum "))
			return name.substr(5);
		return name;
	}

	template <typename T>
	constexpr std::string_view TypeName() {
		return StripTags(entt::type_name<T>::value());
	}

	namespace detail {
		struct TypeNameProbe {};
	} // namespace detail

	inline constexpr bool kTypeNameParseIsSane = TypeName<detail::TypeNameProbe>() ==
			"brassica::graph::detail::TypeNameProbe" ||
		TypeName<detail::TypeNameProbe>() == "detail::TypeNameProbe";

	static_assert(kTypeNameParseIsSane, "TypeName() parse is out of sync with this compiler's function_name() format");

	// Runtime identity for a resource key: the address of a unique inline-constexpr object
	// per key type (ODR-safe since C++17). Unique, stable, constexpr-comparable, and it
	// carries a human-readable name -- no RTTI, no counter of any kind.
	struct ResourceTypeInfo;

	using ResourceId = const ResourceTypeInfo*;

	struct ResourceTypeInfo {
		std::string_view name;
		bool             isHistory;
		ResourceId       historyTarget = nullptr; // non-null iff isHistory
		ResourceId       versionBase = nullptr;   // non-null iff this is a VersionedKey<K, N>
		std::uint32_t    version = 0;             // the N in VersionedKey<K, N>; 0 otherwise
	};

	template <ResourceRef K>
	constexpr ResourceId IdOf();

	// Resolves History<K>'s target id without ever instantiating IdOf<K>() for a plain
	// (non-History) K: the `else` branch is a discarded statement of a dependent
	// if-constexpr, so it's never substituted for K = a bare key, which is what keeps
	// kResourceTypeInfo<K>'s own initializer below from recursing into itself.
	template <typename K>
	constexpr ResourceId HistoryTargetIdOf() {
		if constexpr (IsHistory<K>) {
			return IdOf<HistoryTarget<K>>();
		} else {
			return nullptr;
		}
	}

	// Same discarded-if-constexpr-branch guard as HistoryTargetIdOf, for the same reason: a bare
	// (non-versioned) K must never instantiate IdOf<VersionBase<K>>() from within its own
	// kResourceTypeInfo<K> initializer.
	template <typename K>
	constexpr ResourceId VersionBaseIdOf() {
		if constexpr (IsVersioned<K>) {
			return IdOf<VersionBase<K>>();
		} else {
			return nullptr;
		}
	}

	template <ResourceRef K>
	inline constexpr ResourceTypeInfo kResourceTypeInfo{
		TypeName<K>(),
		IsHistory<K>,
		HistoryTargetIdOf<K>(),
		VersionBaseIdOf<K>(),
		static_cast<std::uint32_t>(VersionIndexOf<K>),
	};

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
