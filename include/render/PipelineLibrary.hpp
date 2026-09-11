#pragma once
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "Shader.hpp"

// One place builds every graphics/compute pipeline in the engine. Replaces
// RenderPass::InitRenderPipeline's fixed-2-stage implementation *and*
// TerrainPass::InitPipelineCustom's ~90-line hand-rolled duplicate of it -- the only reason the
// duplicate existed was RenderPass hardcoding a 2-slot vk::PipelineShaderStageCreateInfo array,
// which made task+mesh+frag impossible to express through it.

namespace brassica::render {

	// The only 5 scalars that vary across every graphics pipeline this engine builds today
	// (confirmed by reading every pass's pipeline-creation code before writing this) -- everything
	// else (vertex input, input assembly, viewport, multisample, the color-blend-state wrapper,
	// dynamic state) is identical in every pipeline and lives once, in PipelineLibrary::Resolve,
	// not duplicated per call site.
	struct GraphicsPipelineState {
		vk::CullModeFlags cullMode = vk::CullModeFlagBits::eBack;
		bool              depthTest = false;
		bool              depthWrite = false;
		vk::CompareOp     depthCompareOp = vk::CompareOp::eLess;
		bool              enableBlend = false;

		// On by default, matching every RenderPass-built pipeline's existing behavior (the
		// fragment-shading-rate pNext was previously chained unconditionally inside
		// RenderPass.cpp). TerrainPass's old hand-rolled InitPipelineCustom never chained it at
		// all, silently ignoring VRS while Gradient/Deferred honored it -- ported as `false` for
		// Terrain specifically (TerrainPass::InitPipeline) to stay pixel-identical for now.
		// Flipping it to fix that divergence is a deliberate, separately-validated follow-up.
		bool enableShadingRate = true;
	};

	struct GraphicsPipelineRequest {
		// In any order -- each entry carries its own vk::ShaderStageFlagBits (via
		// GetStageCreateInfo()), so pipeline creation doesn't care which position is "the vertex
		// stage" vs "the fragment stage" the way this comment's phrasing might suggest.
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

	// Resolve() is the original, uncached contract from this class's first version: every call
	// builds a fresh pipeline+layout, exactly as RenderPass/ComputePass already did -- the caller
	// still owns and destroys what comes back. Correct for today's RenderPass/ComputePass model
	// (one Pass object, persisting across frames, calls Resolve exactly once at construction/
	// hot-reload time), so those callers are untouched.
	//
	// ResolveCached() is for a node with no persistent object of its own to hold a pipeline in --
	// GradientNode and everything ported after it are reconstructed fresh every frame, so they
	// re-resolve their (identical) request every frame too. Without a cache that would mean
	// creating/destroying a real vk::Pipeline 60+ times a second; with it, resolving the same
	// request is a hash lookup. This PipelineLibrary instance owns whatever it hands back --
	// callers must not destroy it -- and it stays valid until superseded by a newer request (a
	// hot-reloaded shader bumps its generation, see Shader::GetGeneration, which changes the
	// cache key and produces a fresh entry rather than mutating the old one in place) or until
	// this PipelineLibrary itself is destroyed.
	class PipelineLibrary {
	public:
		PipelineLibrary(vk::Device device = {}, vk::PipelineCache cache = {}): m_device(device), m_cache(cache) {}

		~PipelineLibrary() { Reset(); }

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

		// Destroys every cached pipeline/layout and clears the cache. Must run while m_device is
		// still a live handle -- called explicitly from Engine::Cleanup() (mirroring
		// PhysicalResourceRegistry::Reset()'s own placement there), not left for the destructor
		// alone to discover a device that's already gone.
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

		std::unordered_map<CacheKey, ResolvedPipeline, CacheKeyHash> m_graphicsCache;
		std::unordered_map<CacheKey, ResolvedPipeline, CacheKeyHash> m_computeCache;
	};

} // namespace brassica::render
