#pragma once

#include "IManager.hpp"
#include "types/CloudData.hpp"

namespace brassica {

	struct CloudStateReflected : public CloudState {
		auto GetReflection() const {
			return std::make_tuple(
				MakeField("renderScale", "Render Resolution Scale", &CloudStateReflected::renderScale, 0.1f, 1.0f, UIHint::Slider),
				MakeField("worldScale", "World Scale Factor", &CloudStateReflected::worldScale, 0.01f, 10.0f, UIHint::Drag),

				MakeField("density", "Cloud Base Density", &CloudStateReflected::density, 0.001f, 2.0f, UIHint::Slider),
				MakeField("altitude", "Cloud Layer Altitude (m)", &CloudStateReflected::altitude, 100.0f, 20000.0f, UIHint::Drag),
				MakeField("thickness", "Cloud Layer Thickness (m)", &CloudStateReflected::thickness, 100.0f, 10000.0f, UIHint::Drag),
				MakeField("coverage", "Cloud Global Coverage", &CloudStateReflected::coverage, 0.0f, 1.0f, UIHint::Slider),
				MakeField("maxRayDistance", "Max Ray Distance (m)", &CloudStateReflected::maxRayDistance, 10000.0f, 500000.0f, UIHint::Drag),
				MakeField("minSamples", "Min Raymarch Samples", &CloudStateReflected::minSamples, 4, 128, UIHint::Slider),
				MakeField("maxSamples", "Max Raymarch Samples", &CloudStateReflected::maxSamples, 16, 256, UIHint::Slider),
				MakeField("extinction", "Cloud Extinction Coefficient", &CloudStateReflected::extinction, 0.01f, 2.0f, UIHint::Slider),
				MakeColorField("extinctionColor", "Extinction Tint Color", &CloudStateReflected::extinctionColor),
				MakeColorField("albedo", "Single Scattering Albedo", &CloudStateReflected::albedo),

				MakeField("phaseG1", "Forward Scattering Peak (g1)", &CloudStateReflected::phaseG1, 0.0f, 0.99f, UIHint::Slider),
				MakeField("phaseG2", "Back Scattering Peak (g2)", &CloudStateReflected::phaseG2, -0.99f, 0.0f, UIHint::Slider),
				MakeField("phaseAlpha", "Phase Lobe Weight", &CloudStateReflected::phaseAlpha, 0.0f, 1.0f, UIHint::Slider),
				MakeField("phaseIsotropic", "Isotropic Blend Factor", &CloudStateReflected::phaseIsotropic, 0.0f, 1.0f, UIHint::Slider),
				MakeField("powderScale", "Beer-Powder Optical Scale", &CloudStateReflected::powderScale, 0.0001f, 0.01f, UIHint::Drag),
				MakeField("powderMultiplier", "Beer-Powder Multiplier", &CloudStateReflected::powderMultiplier, 1.0f, 200.0f, UIHint::Drag),
				MakeField("beerPowderMix", "Beer vs Powder Blend Ratio", &CloudStateReflected::beerPowderMix, 0.0f, 1.0f, UIHint::Slider),

				MakeField("shadowOpticalDepthMultiplier", "Cloud Self Shadow Depth Scale", &CloudStateReflected::shadowOpticalDepthMultiplier, 0.1f, 20.0f, UIHint::Slider),
				MakeField("shadowStepMultiplier", "Cloud Shadow Step Multiplier", &CloudStateReflected::shadowStepMultiplier, 0.1f, 10.0f, UIHint::Slider),
				MakeField("shadowIntensity", "Direct Cloud Shadow Intensity", &CloudStateReflected::shadowIntensity, 0.0f, 1.0f, UIHint::Slider),
				MakeField("sunLightScale", "Sun Directional Light Scale", &CloudStateReflected::sunLightScale, 0.0f, 10.0f, UIHint::Slider),
				MakeField("moonLightScale", "Moon Directional Light Scale", &CloudStateReflected::moonLightScale, 0.0f, 10.0f, UIHint::Slider),

				MakeField("enableTileScheduler", "Enable Staggered Tile Scheduler", &CloudStateReflected::enableTileScheduler),
				MakeField("spatialUpdateFrames", "Tile Spatial Refresh Interval", &CloudStateReflected::spatialUpdateFrames, 1, 64, UIHint::Slider),
				MakeField("maxRefreshRate", "Tile Rendering Budget Ratio", &CloudStateReflected::maxRefreshRate, 0.01f, 1.0f, UIHint::Slider),
				MakeField("priorityErrorWeight", "Priority Error Score Weight", &CloudStateReflected::priorityErrorWeight, 0.0f, 10.0f, UIHint::Slider),
				MakeField("priorityThreshold", "Priority Scheduling Threshold", &CloudStateReflected::priorityThreshold, 0.001f, 1.0f, UIHint::Drag),

				MakeField("enableTemporal", "Enable Temporal Reprojection", &CloudStateReflected::enableTemporal),
				MakeField("enableSpatialFilter", "Enable SVGF Spatial Filter", &CloudStateReflected::enableSpatialFilter),
				MakeField("svgfPasses", "SVGF Filter Pass Count", &CloudStateReflected::svgfPasses, 1, 6, UIHint::Slider),
				MakeField("temporalGamma", "Temporal Clamp Extents (Gamma)", &CloudStateReflected::temporalGamma, 0.1f, 5.0f, UIHint::Slider),
				MakeField("maxHistoryLength", "Max History Length (Frames)", &CloudStateReflected::maxHistoryLength, 1.0f, 128.0f, UIHint::Slider),
				MakeField("phiLuma", "SVGF Phi Luma Weight", &CloudStateReflected::phiLuma, 0.1f, 100.0f, UIHint::Slider),
				MakeField("phiDepth", "SVGF Phi Depth Weight", &CloudStateReflected::phiDepth, 0.01f, 10.0f, UIHint::Slider),
				MakeField("phiDensity", "SVGF Phi Density Weight", &CloudStateReflected::phiDensity, 0.001f, 1.0f, UIHint::Slider),

				MakeField("flowSpeed", "Cloud Wind Flow Speed", &CloudStateReflected::flowSpeed, 0.0f, 10.0f, UIHint::Slider),
				MakeField("flowDirection", "Cloud Flow Direction (Rad)", &CloudStateReflected::flowDirection, 0.0f, 6.2831853f, UIHint::Slider),
				MakeField("curlStrength", "Curl Distortion Strength", &CloudStateReflected::curlStrength, 0.0f, 50.0f, UIHint::Slider)
			);
		}
	};

	class ICloudManager : public ManagerBase<ICloudManager, CloudStateReflected> {
	public:
		using State = CloudStateReflected;

		~ICloudManager() override = default;

		std::string GetManagerName() const override { return "CloudManager"; }

		State GetState() const override { return m_state; }

		void SetState(const State& state) override { m_state = state; }

		virtual void FlushHistory() { m_hasHistory = false; }
		[[nodiscard]] bool HasHistory() const { return m_hasHistory; }

	protected:
		State m_state{};
		bool m_hasHistory{false};
	};

} // namespace brassica
