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

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>

// --- 1. The Argument Wrapper ---
// Acts as a strong type for SignatureTraits, but implicitly converts to float for the math.
template <typename Node>
struct Arg {
	float v;

	operator float() const { return v; }
};

// Extractor to pull 'Node' out of 'Arg<Node>'
template <typename T>
struct ExtractNode;

template <typename T>
struct ExtractNode<Arg<T>> {
	using Type = T;
};

// --- 2. Signature Traits ---
template <typename T>
struct SignatureTraits;

template <typename ClassType, typename Return, typename... Args>
struct SignatureTraits<Return (ClassType::*)(Args...) const> {
	// Unpack the Arg<T> wrappers back into the raw Node types
	using Dependencies = std::tuple<typename ExtractNode<std::remove_cvref_t<Args>>::Type...>;
};

template <typename T>
using NodeTraits = SignatureTraits<decltype(&T::operator())>;

// --- 3. The Manager's State Container ---
template <typename Node>
struct NodeState {
	float value = Node::minVal;
	float invMass = Node::baseInvMass;

	void Clamp() {
		value = std::max(Node::minVal, std::min(value, Node::maxVal));
		// Soft limit dynamic resistance logic goes here
	}
};

struct Precipitation {
	static constexpr float minVal = 0.0f;
	static constexpr float maxVal = 1.0f;
	static constexpr float baseInvMass = 1.0f;

	// Root nodes have no dependencies.
	float operator()() const { return 0.0f; }
};

struct Humidity {
	static constexpr float minVal = 0.0f;
	static constexpr float maxVal = 1.0f;
	static constexpr float baseInvMass = 1.0f;

	// The signature tells the Manager this depends on Precipitation.
	// The implicit float conversion makes the math natural.
	float operator()(Arg<Precipitation> rain) const { return rain * 0.8f + 0.2f; }
};

template <typename... Nodes>
class MoodManager {
public:
	std::tuple<NodeState<Nodes>...> states;

	template <typename T>
	void SetOverride(float val) {
		auto& state = std::get<NodeState<T>>(states);
		state.value = val;
		state.invMass = 0.0f; // Infinite mass, solver cannot move it
	}

	void Update() {
		// Iterate constraints multiple times for PBD relaxation
		for (int i = 0; i < 3; ++i) {
			(ResolveConstraint<Nodes>(), ...);
		}

		// Post-solve bounds clamping
		(std::get<NodeState<Nodes>>(states).Clamp(), ...);
	}

	template <typename T>
	float Get() const {
		return std::get<NodeState<T>>(states).value;
	}

private:
	template <typename Node>
	void ResolveConstraint() {
		using DepsTuple = typename NodeTraits<Node>::Dependencies;
		constexpr size_t NumDeps = std::tuple_size_v<DepsTuple>;

		if constexpr (NumDeps > 0) {
			ResolveImpl<Node>(std::make_index_sequence<NumDeps>{});
		}
	}

	template <typename Node, size_t... Is>
	void ResolveImpl(std::index_sequence<Is...>) {
		using DepsTuple = typename NodeTraits<Node>::Dependencies;
		auto& myState = std::get<NodeState<Node>>(states);

		// Current values of all dependencies
		std::array<float, sizeof...(Is)> depVals = {
			std::get<NodeState<std::tuple_element_t<Is, DepsTuple>>>(states).value...
		};

		// Helper to evaluate the user's operator() from an array of floats
		auto evaluate = [&](const std::array<float, sizeof...(Is)>& vals) {
			return Node{}(Arg<std::tuple_element_t<Is, DepsTuple>>{vals[Is]}...);
		};

		// 1. Calculate Error
		float target = evaluate(depVals);
		float error = myState.value - target; // C = current - target

		if (std::abs(error) < 1e-5f)
			return;

		// 2. Numerical Gradients (Automatic Differentiation)
		float                            wSum = myState.invMass; // Gradient w.r.t self is 1.0 (1.0^2 * invMass)
		std::array<float, sizeof...(Is)> grads;
		constexpr float                  epsilon = 0.001f;

		// Fold expression to evaluate the derivative for each dependency
		(..., [&]() {
			std::array<float, sizeof...(Is)> nudgedVals = depVals;
			nudgedVals[Is] += epsilon;
			float nudgedTarget = evaluate(nudgedVals);

			// gradient = - (dTarget / dDependency)
			float grad = -(nudgedTarget - target) / epsilon;
			grads[Is] = grad;

			auto& depState = std::get<NodeState<std::tuple_element_t<Is, DepsTuple>>>(states);
			wSum += depState.invMass * (grad * grad);
		}());

		if (wSum < 1e-6f)
			return; // Everything is pinned

		// 3. Apply Corrections (Bumping)
		float deltaLambda = -error / wSum;

		myState.value += myState.invMass * deltaLambda * 1.0f; // Move self

		(..., [&]() {
			auto& depState = std::get<NodeState<std::tuple_element_t<Is, DepsTuple>>>(states);
			depState.value += depState.invMass * deltaLambda * grads[Is]; // Move dependencies
		}());
	}
};

int main() {
	MoodManager<Precipitation, Humidity> manager;
	// User sets rain to 1.0 (heavy rain) and locks it
	manager.SetOverride<Precipitation>(1.0f);
	manager.Update();
}