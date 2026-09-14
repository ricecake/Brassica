#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Graph.hpp"
#include "VulkanCompat.hpp"

namespace brassica {
	class ShaderWatcher;
	class TerrainAccelerationStructure;
} // namespace brassica

namespace brassica::render {

	class PipelineLibrary;

	// Every engine-level node's Init takes exactly this, pulling whichever fields it actually
	// needs -- collapses today's several distinct per-node Init signatures (device+library+
	// watcher; +TerrainAccelerationStructure*; +vk::Format; +DispatchLoaderDynamic*+vk::Format)
	// into one, which is what lets EngineNodeRegistry drive every registered node through the
	// same call instead of Engine hand-writing one call per node type.
	struct NodeServices {
		vk::Device                    device;
		PipelineLibrary*              pipelineLibrary = nullptr;
		ShaderWatcher*                shaderWatcher = nullptr;
		TerrainAccelerationStructure* terrainAS = nullptr;
		const DispatchLoaderDynamic*  dispatchLoader = nullptr;
		vk::Format                    swapchainFormat = vk::Format::eUndefined;
	};

	// This frame's data for whichever nodes need it -- unlike NodeServices (Init-time, mostly
	// pointers to Engine-owned services), these are plain values recomputed every frame. Only a
	// few nodes have per-frame data at all (today: TerrainNode/DeferredNode share the toroidal
	// clipmap offsets, WaterNode has its own two fields), so a node opts in by declaring its own
	// `void SetFrameParams(const NodeFrameParams&)` and pulling out whichever fields it needs --
	// see HasSetFrameParams below. Most nodes (GradientNode, the atmosphere LUT nodes,
	// ParticleSystemNode) don't declare one at all and are silently skipped.
	struct NodeFrameParams {
		glm::uvec4 terrainGridParams{8, 16, 2048, 1088};
		glm::uvec4 terrainLodOffsets0_3{0u};
		glm::uvec4 terrainLodOffsets4_7{0u};
		glm::vec3  waterColor{0.05f, 0.45f, 0.85f};
		float      waterLevel{0.0f};
	};

	template <typename T>
	concept HasSetFrameParams = requires(T t, const NodeFrameParams& p) { t.SetFrameParams(p); };

	// Type-erased per-node-type lifecycle handle the registry stores one of per registered type.
	// Deliberately separate from graph::NodeHandle/IErased (include/graph/Node.hpp) -- that
	// erasure is part of the Vulkan-free graph layer (verified by `make graph-check`); this one is
	// Vulkan-aware by design (NodeServices carries vk::Device/PipelineLibrary*/etc.), so merging
	// them would leak Vulkan into the graph-free seam.
	struct INodeLifecycle {
		virtual ~INodeLifecycle() = default;
		virtual void Init(const NodeServices&) = 0;
		virtual void Destroy(vk::Device) = 0;
		virtual void RegisterInto(graph::Graph&) = 0;
		virtual void SetFrameParams(const NodeFrameParams&) = 0;
	};

	template <graph::NodeLike T>
	struct NodeLifecycle final: INodeLifecycle {
		T value{};

		void Init(const NodeServices& s) override { value.Init(s); }

		void Destroy(vk::Device d) override { value.Destroy(d); }

		void RegisterInto(graph::Graph& g) override { g.RegisterRef(value); }

		void SetFrameParams(const NodeFrameParams& p) override {
			if constexpr (HasSetFrameParams<T>) {
				value.SetFrameParams(p);
			}
		}
	};

	// Owns exactly one instance of each registered node type, for the lifetime of the program --
	// a Meyer's singleton (construct on first use) rather than a namespace-scope object, so there
	// is no static-initialization-order question between this and the NodeRegistrar<T> instances
	// that call into it from other translation units.
	class EngineNodeRegistry {
	public:
		static EngineNodeRegistry& Instance() {
			static EngineNodeRegistry instance;
			return instance;
		}

		template <typename T>
		void RegisterType() {
			m_factories.push_back([] { return std::make_unique<NodeLifecycle<T>>(); });
		}

		void CreateAll() {
			m_instances.clear();
			m_instances.reserve(m_factories.size());
			for (auto& factory : m_factories) {
				m_instances.push_back(factory());
			}
		}

		void InitAll(const NodeServices& services) {
			for (auto& node : m_instances) {
				node->Init(services);
			}
		}

		void DestroyAll(vk::Device device) {
			for (auto& node : m_instances) {
				node->Destroy(device);
			}
		}

		void RegisterAllInto(graph::Graph& graph) {
			for (auto& node : m_instances) {
				node->RegisterInto(graph);
			}
		}

		void SetFrameParamsAll(const NodeFrameParams& params) {
			for (auto& node : m_instances) {
				node->SetFrameParams(params);
			}
		}

		[[nodiscard]] std::size_t RegisteredTypeCount() const { return m_factories.size(); }

	private:
		std::vector<std::function<std::unique_ptr<INodeLifecycle>()>> m_factories;
		std::vector<std::unique_ptr<INodeLifecycle>>                  m_instances;
	};

	// A node opts in to auto-registration by inheriting this instead of nothing:
	// struct GradientNode : render::NodeRegistrar<GradientNode> { ... };
	template <typename Derived>
	struct NodeRegistrar {
		static bool _registered;
	};

	template <typename Derived>
	bool NodeRegistrar<Derived>::_registered = [] {
		EngineNodeRegistry::Instance().RegisterType<Derived>();
		return true;
	}();

} // namespace brassica::render

// Inheriting from NodeRegistrar<Derived> alone does NOT force _registered's initializer to run:
// per [temp.inst], implicitly instantiating a class template specialization does not implicitly
// instantiate its (non-virtual) static data members -- only an actual ODR-use does. The obvious
// candidate for that ODR-use, Derived's own constructor, only helps if something somewhere
// actually constructs a bare Derived -- which was true back when Engine held a named
// `GradientNode gradientNode;` member, but is exactly what this mechanism removes. Confirmed
// empirically (tests/test_node_registry.cpp): without an explicit trigger, RegisteredTypeCount()
// silently reads back 0.
//
// The fix is an explicit-instantiation directive naming the static member directly. Every node
// header must invoke this once, right after its struct definition -- e.g.
// `BRASSICA_REGISTER_NODE(GradientNode);` inside namespace brassica, right after the struct. This
// is deliberately NOT an anonymous-namespace variable (the first thing that comes to mind): an
// unnamed-namespace entity has internal linkage, so a header included from two different
// translation units linked into the same binary (e.g. Engine.cpp and apps/*/main.cpp both
// include this via Engine.hpp) would get two independent copies, each running RegisterType<T>()
// once -- double-registering every node in any real executable. Explicit template instantiation
// doesn't have that problem: it emits a weak/COMDAT symbol like any other template instantiation,
// so the linker folds duplicates from multiple TUs into the one copy whose initializer actually
// runs, exactly like _registered's own defining declaration above already relies on.
#define BRASSICA_REGISTER_NODE(Type) template bool ::brassica::render::NodeRegistrar<Type>::_registered
