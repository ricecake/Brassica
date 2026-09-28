// #include <fmt/core.h>

// #include <algorithm>
// #include <cmath>
// #include <concepts>
// #include <type_traits>
// #include <utility>
// #include <vector>

// #include <glm/glm.hpp>

// #include <glm/gtc/random.hpp>

// template <class A, typename V, typename R>
// 	requires requires(A a) { A()->V; }
// R process(V value) {
// 	return R();
// }

// struct Upstream {
// 	float operator()() { return 2.0f; }
// };

// template <class TypeA, typename ValueA, typename Return>
// 	requires requires(TypeA a) {
// 		{ a() } -> std::same_as<ValueA>;
// 	}
// class Nodey {
// 	Return operator()(ValueA paramA);
// };

// class MyNode: Nodey<Upstream, float, float> {
// 	float operator()(float p) { return 1.0f; }
// };
// ///////////////////

// template <class TypeA, typename ValueA, typename Return>
// 	requires requires(TypeA a) {
// 		{ a() } -> std::same_as<ValueA>;
// 		{ a } -> std::convertible_to<ValueA>;
// 	}
// class OtherNode {
// 	public:
// 	Return operator()(TypeA paramA);
// 	operator Return() const {

// 	}
// };


// ////////////

// // Forward declare the trait that deduces a node's output type
// template <typename T>
// struct NodeOutput {
// 	// Fallback for root/upstream nodes that don't inherit from Node.
// 	using type = decltype(std::declval<T>()());
// };

// template <typename T>
// using NodeOutput_t = typename NodeOutput<T>::type;

// // The Variadic CRTP Base Class
// template <class Derived, class... Upstreams>
// class Node {
// public:
// 	// Defer deduction until `Derived` is fully defined using a template parameter `D`.
// 	// This prevents "incomplete type" compiler errors during base class instantiation.
// 	template <class D = Derived>
// 	static auto DeduceReturn() -> decltype(std::declval<D>()(std::declval<NodeOutput_t<Upstreams>>()...));
// };

// // Concept to detect if a class is part of the CRTP Node hierarchy
// template <typename T>
// concept IsDerivedNode = requires { T::template DeduceReturn<T>(); };

// // Specialization of the trait for derived nodes
// template <IsDerivedNode T>
// struct NodeOutput<T> {
// 	using type = decltype(T::template DeduceReturn<T>());
// };

// /////

// template<typename Type>
// class BaseNode {
// 	ManagerClass getManager();
// 	operator Type(){
// 		auto manager = getManager();
// 		auto args = manager->GetArgs();
// 		return process(args...);
// 	}

// 	Type process(ClassA a, ClassB b) = 0;
// };

// class MyNode : BaseNode<float> {
// 	float process(ClassA a, ClassB b) {
// 		return a + b;
// 	}
// };


// int main() {}