#pragma once
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "IManager.hpp"
#include "Shader.hpp"

namespace brassica::render {

	struct PipelineLibraryState {
		bool cacheEnabled{true};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("cacheEnabled", "Pipeline Cache Enabled", &PipelineLibraryState::cacheEnabled)
			);
		}
	};

	struct GraphicsPipelineState {
		vk::CullModeFlags cullMode = vk::CullModeFlagBits::eBack;
		bool              depthTest = false;
		bool              depthWrite = false;
		vk::CompareOp     depthCompareOp = vk::CompareOp::eLess;
		bool              enableBlend = false;

		bool enableShadingRate = false;
	};

	struct GraphicsPipelineRequest {
		std::span<GraphicsShader* const>         stages;
		GraphicsPipelineState                    state{};
		std::span<const vk::Format>              colorFormats;
		vk::Format                               depthFormat = vk::Format::eUndefined;
		std::span<const vk::DescriptorSetLayout> setLayouts;
		std::span<const vk::PushConstantRange>   pushConstantRanges;
	};

	struct ComputePipelineRequest {
		ComputeShader*                           shader = nullptr;
		std::span<const vk::DescriptorSetLayout> setLayouts;
		std::span<const vk::PushConstantRange>   pushConstantRanges;
	};

	struct ResolvedPipeline {
		vk::Pipeline       pipeline{};
		vk::PipelineLayout layout{};
	};

	class PipelineLibrary: public ManagerBase<PipelineLibrary, PipelineLibraryState> {
	public:
		using State = PipelineLibraryState;

		PipelineLibrary(vk::Device device = {}, vk::PipelineCache cache = {}): m_device(device), m_cache(cache) {}

		~PipelineLibrary() override {
			if (m_initialized) {
				Shutdown();
			} else {
				Reset();
			}
		}

		void Initialize() override { m_initialized = true; }

		void Shutdown() override {
			Reset();
			m_initialized = false;
		}

		std::string GetManagerName() const override { return "PipelineLibrary"; }

		State GetState() const override { return State{m_cacheEnabled}; }

		void SetState(const State& state) override { m_cacheEnabled = state.cacheEnabled; }

		PipelineLibrary(const PipelineLibrary&) = delete;
		PipelineLibrary& operator=(const PipelineLibrary&) = delete;

		void SetDeviceAndCache(vk::Device device, vk::PipelineCache cache) {
			m_device = device;
			m_cache = cache;
		}

		[[nodiscard]] ResolvedPipeline Resolve(const GraphicsPipelineRequest& request) const;
		[[nodiscard]] ResolvedPipeline Resolve(const ComputePipelineRequest& request) const;

		[[nodiscard]] ResolvedPipeline ResolveCached(const GraphicsPipelineRequest& request);
		[[nodiscard]] ResolvedPipeline ResolveCached(const ComputePipelineRequest& request);

		void Reset();

	private:
		struct CacheKey {
			std::vector<std::uint64_t> words;

			bool operator==(const CacheKey&) const = default;
		};

		struct CacheKeyHash {
			std::size_t operator()(const CacheKey& key) const noexcept {
				std::size_t h = key.words.size();
				for (std::uint64_t w : key.words) {
					h ^= std::hash<std::uint64_t>{}(w) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
				}
				return h;
			}
		};

		static CacheKey BuildKey(const GraphicsPipelineRequest& request);
		static CacheKey BuildKey(const ComputePipelineRequest& request);

		[[nodiscard]] ResolvedPipeline Build(const GraphicsPipelineRequest& request) const;
		[[nodiscard]] ResolvedPipeline Build(const ComputePipelineRequest& request) const;

		vk::Device        m_device;
		vk::PipelineCache m_cache;
		bool              m_cacheEnabled{true};

		std::unordered_map<CacheKey, ResolvedPipeline, CacheKeyHash> m_graphicsCache;
		std::unordered_map<CacheKey, ResolvedPipeline, CacheKeyHash> m_computeCache;
	};

} // namespace brassica::render
