#pragma once

#include "BufferManager.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "Shaders/TechniquesCompat.h"
#include "RayTracingHelpers.h"
#include "RaytracingStateObject.h"
#include "RaytracingShadow.h"
#include "DX12Lib/Commons/CommonConstants.h"
#include "DX12Lib/DXWrapper/SamplerDesc.h"
#include "Technique.h"

namespace DX12Lib
{
	class RootSignature;
	class ComputePipelineState;
	class ShadowCamera;
}

namespace CVGI
{
	enum class RayTraceShadowRootSignature
	{
		RayTracinShadowCommonCBV = 0,
		ShadowCameraCBV = 1,
		AccelerationStructureSRV = 2,
		ShadowMapSRV,
		CompactTableSRV,
		AABBTableSRV,
		ASBufferMapSRV,
		RayTraceShadowTableUAV,
		Count
	};

	class LightVoxel : public Technique
	{
	public:
		LightVoxel(std::shared_ptr<TechniqueData> data);
		~LightVoxel() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(RayTracingContext& context) override;
		virtual void TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize) override;


		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

		void SetShadowCamera(DX12Lib::ShadowCamera* shadowCamera) { m_shadowCamera = shadowCamera; }
		
	protected:
		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	public:
		static const std::wstring Name;

	private:
		ConstantBufferRTShadows m_cbShadowRaytrace;
		DX12Lib::ShadowCamera* m_shadowCamera;
	};
}



