#pragma once

namespace Brassica {
/*
	The basic notion is to have phases and graphs both be able to act like nodes, so that
	the graph can hold a group of them, and sort the phases in order, and nodes can specify
	that they come before or after different phases, and graphs can be treated as nodes so
	that complex nodes can themselves be comprised of nodes.
*/
	template <typename T> concept NodeLike = requires(){ true; };
	class Phase {};
	class Graph{};
	class Node{};
	class Queue{};
	class Resource{
		void GetReadBarrier() {};
		void GetReleaseBarrier(Queue destQueue) {};
		void GetAcquireBarrier(Queue sourceQueue) {};
	};
	class ConcreteResource {
		Resource t;
		std::tuple initArgs;
	}; // Recipie should have a list of concrete resource requirements
	class Recipe{};
	class PreviousFrame : public Node {};
	class NextFrame : public Node {};
	class FrameStart : public Phase {};
	class FrameEnd : public Phase {};
	class Operation {};
	template <typename T> class Create : public operation {};
	template <typename T> class Read : public operation {};
	template <typename T, typename X = T> class Modify : public operation {};
}; // namespace Brassica


#include <iostream>
#include <tuple>
#include <utility>
#include <string>

// A template that holds a type T and the arguments needed to construct it
template <typename T, typename... Args>
class LazyInitializer {
private:
    // Store the arguments as a tuple
    std::tuple<Args...> stored_args;

public:
    // Constructor uses perfect forwarding to capture arguments efficiently
    explicit LazyInitializer(Args&&... args)
        : stored_args(std::forward<Args>(args)...) {}

    // Method to create and return the object T using the stored arguments
    T initialize() {
        // std::apply unpacks the tuple into the constructor of T
        return std::apply([](auto&&... args) {
            return T(std::forward<decltype(args)>(args)...);
        }, std::move(stored_args));
    }
};





class Graph {
		bool m_hasActiveInternalNodes = true;
	public:
		// Fulfills NodeLike Concept: Setup Phase
		Recipe Setup(const RenderContext& ctx) {
			// 1. Iterate internal nodes and call Setup()
			// 2. Perform internal topological sort and culling
			// 3. Determine if the subgraph as a whole has work to do

			return Recipe{
				.domain = ExecutionDomain::Graphics, // See domain mapping note below
				.isActive = m_hasActiveInternalNodes
			};
		}

		// Fulfills NodeLike Concept: Execution Phase
		void Execute(CommandBuffer& cmd) {
			if (!m_hasActiveInternalNodes) return;

			// Iterate surviving internal nodes and pass the parent's
			// command buffer down the chain.
			// for (auto& node : m_sortedActiveNodes) { node.Execute(cmd); }
		}

		template<NodeLike T, typename... Ops>
		void RegisterNode() {}
	};

	// A compile-time assertion to guarantee Graph always satisfies NodeLike
	static_assert(NodeLike<Graph>, "Graph must fulfill NodeLike to act as a subgraph");





#include <iostream>
#include <tuple>
#include <variant>
#include <vector>
#include <string>

// 1. Define the possible inputs using a std::variant
using InputVar = std::variant<int, double, std::string>;

// 2. The Worker Class (advertises what it wants, receives them in an array/vector)
class MyWorker {
public:
    // This defines the "type tags" the class expects to receive
    using ExpectedTypes = std::tuple<int, std::string>;

    // The method that accepts the variable set of inputs matching the tags
    void execute(int id, const std::string& name) {
        std::cout << "Worker executed with ID: " << id << " and Name: " << name << "\n";
    }
};

// 3. The Coordinator Class (extracts the right types from a pool of inputs)
class Coordinator {
public:
    template <typename Worker>
    static void dispatch(Worker& worker, const std::vector<InputVar>& input_pool) {
        // Extract the tuple of type tags from the worker
        typename Worker::ExpectedTypes tags;

        // Unpack the tuple tags at compile-time and extract them from the input pool
        std::apply([&worker, &input_pool](auto... dummy_tags) {

            // Lambda to extract a specific type 'T' from the generic input pool
            auto extract = [&input_pool](auto tag_type) {
                using T = decltype(tag_type);
                for (const auto& var : input_pool) {
                    if (std::holds_alternative<T>(var)) {
                        return std::get<T>(var);
                    }
                }
                throw std::runtime_error("Required type not found in input pool!");
            };

            // Call the execute method by passing the extracted arguments
            worker.execute(extract(dummy_tags)...);

        }, tags);
    }
};



// frameGraph.hpp
#pragma once
#include <concepts>
#include <type_traits>

namespace Brassica {

	// 1. Strong Typing for Keys
	// Users define empty structs like `struct GBufferAlbedo {};` to act as keys.
	// This ensures the compiler catches resource mismatches.
	template <typename T> concept ResourceKey = std::is_empty_v<T> && std::is_trivially_constructible_v<T>;

	// 2. Execution Domains
	// Replaces the empty Queue class with a strongly typed enum for Vulkan queue mapping.
	enum class ExecutionDomain {
		Graphics,
		Compute,
		Host,
		Transfer
	};

	// 3. Synchronization & Execution Stubs
	// Analogues for Vulkan 1.3 VkImageMemoryBarrier2 and VkCommandBuffer
	struct MemoryBarrier {};
	struct CommandBuffer {};
	struct RenderContext {}; // Carries frame state (e.g., UI toggles, camera data)

	// 4. Recipe Definition
	// Returned dynamically by node factories to indicate current frame requirements.
	struct Recipe {
		ExecutionDomain domain = ExecutionDomain::Graphics;
		bool isActive = true;
	};

	// 5. Node Factory Concept
	// Enforces that anything treated as a node can evaluate context to return a recipe,
	// and can execute commands when scheduled.
	template <typename T>
	concept NodeLike = requires(T t, const RenderContext& ctx, CommandBuffer& cmd) {
		{ t.Setup(ctx) } -> std::same_as<Recipe>;
		{ t.Execute(cmd) };
	};

	// 6. Phases & Ordering
	class Phase {};
	class FrameStart : public Phase {};
	class FrameEnd : public Phase {};
	class PreviousFrame {};
	class NextFrame {};

	// 7. Graph Container
	class Graph {
	public:
		// Registers a node factory, inferring input/output dependencies at compile time via Operations
		template<NodeLike T, typename... Ops>
		void RegisterNode() {}

		// Injects external resources (like the swapchain) directly via type keys
		template<ResourceKey K>
		void InjectExternalResource(class Resource* resource) {}

		// The three-phase frame loop
		void Setup(const RenderContext& ctx) {}
		void Compile() {}
		void Execute() {}
	};

	// 8. Operations (Constrained to Resource Keys)
	class Operation {};
	template <ResourceKey T> class Create : public Operation { using Key = T; };
	template <ResourceKey T> class Read : public Operation { using Key = T; };
	template <ResourceKey T, ResourceKey X = T> class Modify : public Operation { using ReadKey = T; using WriteKey = X; };

	// 9. Resource Interface
	// Returns barrier metadata structs for Graph-level batching instead of executing commands directly.
	class Resource {
	public:
		virtual ~Resource() = default;
		virtual MemoryBarrier GetReadBarrier() const = 0;
		virtual MemoryBarrier GetWriteBarrier() const = 0;
		virtual MemoryBarrier GetReleaseBarrier(ExecutionDomain destQueue) const = 0;
		virtual MemoryBarrier GetAcquireBarrier(ExecutionDomain sourceQueue) const = 0;
	};

}; // namespace Brassica