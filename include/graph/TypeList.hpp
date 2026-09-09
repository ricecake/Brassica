#pragma once
#include <cstddef>
#include <type_traits>

namespace brassica::graph {

	// A type-only tag list. Deliberately not std::tuple: elements need not be complete,
	// destructible, or constructible, and the operations below (dedup, difference) have
	// no std equivalent.
	template <typename... Ts>
	struct TypeList {};

	// -- contains --------------------------------------------------------------

	template <typename T, typename L>
	struct ContainsT;

	template <typename T, typename... Ts>
	struct ContainsT<T, TypeList<Ts...>>: std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

	template <typename T, typename L>
	inline constexpr bool Contains = ContainsT<T, L>::value;

	// -- concat (n-ary) ----------------------------------------------------------

	template <typename... Ls>
	struct ConcatT;

	template <>
	struct ConcatT<> {
		using type = TypeList<>;
	};

	template <typename... Ts>
	struct ConcatT<TypeList<Ts...>> {
		using type = TypeList<Ts...>;
	};

	template <typename... As, typename... Bs, typename... Rest>
	struct ConcatT<TypeList<As...>, TypeList<Bs...>, Rest...> {
		using type = typename ConcatT<TypeList<As..., Bs...>, Rest...>::type;
	};

	template <typename... Ls>
	using Concat = typename ConcatT<Ls...>::type;

	// -- dedup (order-preserving, first occurrence wins) --------------------------

	template <typename Acc, typename... Ts>
	struct DedupT {
		using type = Acc;
	};

	template <typename... As, typename T, typename... Ts>
	struct DedupT<TypeList<As...>, T, Ts...> {
		using type = typename DedupT<
			std::conditional_t<Contains<T, TypeList<As...>>, TypeList<As...>, TypeList<As..., T>>,
			Ts...>::type;
	};

	template <typename L>
	struct DedupListT;

	template <typename... Ts>
	struct DedupListT<TypeList<Ts...>> {
		using type = typename DedupT<TypeList<>, Ts...>::type;
	};

	template <typename L>
	using Dedup = typename DedupListT<L>::type;

	// -- difference: A \ B ---------------------------------------------------------
	// Non-recursive: one pack expansion feeding the n-ary Concat above, so instantiation
	// depth doesn't grow with list length. This is the hot operation for validation.

	template <typename A, typename B>
	struct DifferenceT;

	template <typename... As, typename B>
	struct DifferenceT<TypeList<As...>, B> {
		using type = Concat<std::conditional_t<Contains<As, B>, TypeList<>, TypeList<As>>...>;
	};

	template <typename A, typename B>
	using Difference = typename DifferenceT<A, B>::type;

	// -- size / subset --------------------------------------------------------------

	template <typename L>
	struct SizeT;

	template <typename... Ts>
	struct SizeT<TypeList<Ts...>>: std::integral_constant<std::size_t, sizeof...(Ts)> {};

	template <typename L>
	inline constexpr std::size_t Size = SizeT<L>::value;

	template <typename A, typename B>
	inline constexpr bool IsSubsetOf = Size<Difference<A, B>> == 0;

} // namespace brassica::graph
