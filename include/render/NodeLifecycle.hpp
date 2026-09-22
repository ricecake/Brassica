#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Graph.hpp"
#include "types/AtmospherePushConstants.hpp"
#include "VulkanCompat.hpp"

namespace brassica {
	class ShaderWatcher;
	class TerrainAccelerationStructure;
	class TerrainClipmap;

	namespace graph {
		class PhysicalResourceRegistry;
	}
} // namespace brassica

namespace brassica::render {

	class PipelineLibrary;

	struct NodeServices {
		vk::Device                       device;
		PipelineLibrary*                 pipelineLibrary = nullptr;
		ShaderWatcher*                   shaderWatcher = nullptr;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainClipmap*                  terrainClipmap = nullptr;
		const DispatchLoaderDynamic*     dispatchLoader = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		vk::Format                       swapchainFormat = vk::Format::eUndefined;
	};

	struct NodeFrameParams {
		glm::vec3  cameraPosition{0.0f};
		glm::uvec4 terrainGridParams{10, 16, 2560, 1088};
		glm::uvec4 terrainLodOffsets0_3{0u};
		glm::uvec4 terrainLodOffsets4_7{0u};
		glm::uvec4 terrainLodOffsets8_11{0u};
		glm::uvec4 terrainLodDeltas0_3{0u};
		glm::uvec4 terrainLodDeltas4_7{0u};
		glm::uvec4 terrainLodDeltas8_11{0u};
		bool       terrainHasUpdate{false};
		glm::vec3  waterColor{0.05f, 0.45f, 0.85f};
		float      waterLevel{0.0f};
		glm::vec3  sunDir{0.0f, 1.0f, 0.0f};
		glm::vec3  sunRadiance{3.0f, 2.94f, 2.76f};
		glm::vec3  moonDir{0.0f, -1.0f, 0.0f};
		glm::vec3  moonRadiance{0.1f, 0.12f, 0.16f};
		float      time{0.0f};
		float      worldScale{1.0f};
		float      multiScatScale{1.0f};
		float      cloudShadowIntensity{0.5f};
		float      skyExposure{1.0f};
		// Not for shader delivery -- shaders read the real values from the global AtmosphereUBO
		// (atmosphere/common.glsl, set 0 binding 4). This is here purely so
		// AtmosphereRegenerationState::ShouldRegenerate (the 3 LUT nodes) has a current value to
		// byte-compare against each frame.
		AtmospherePushConstants atmosphere{};
	};

	template <typename T>
	concept HasSetFrameParams = requires(T t, const NodeFrameParams& p) { t.SetFrameParams(p); };

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

#define BRASSICA_REGISTER_NODE(Type) template bool ::brassica::render::NodeRegistrar<Type>::_registered
