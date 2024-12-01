#pragma once

#include "../Helpers/BufferManager.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "../Shaders/TechniquesCompat.h"
#include "../Helpers/RayTracingHelpers.h"
#include "../Helpers/RaytracingStateObject.h"
#include "../Data/Shaders/Include/RaytracingShadow.h"
#include "../Data/Shaders/Include/ClearBufferShader_CS.h"
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
		VoxelCommonCBV = 0,
		ShadowCommonCBV = 1,
		AccelerationStructureSRV = 2,
		PrefixSumBufferSRV,
		ClusterVoxelBufferSRV,
		RayTraceShadowTableUAV,
		Count
	};

	enum class ShadowRootSignature
	{
		VoxelCommonCBV = 0,
		LightCommonCBV = 1,
		ShadowTextureSRV,
		VoxelSRV,
		CompactSRV,
		ClusterSRV,
		LightVoxelUAV,
		Count
	};

	class LightVoxel : public Technique
	{
	public:
		LightVoxel(std::shared_ptr<TechniqueData> data);
		~LightVoxel() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

		//virtual void PerformTechnique(RayTracingContext& context) override;
		//virtual void TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize) override;
		void ClearBufferPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize);

		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
		void BuildClearBufferPso();

		void SetLightComponent(DX12Lib::LightComponent* lightComponent) { m_lightComponent = lightComponent; }
		
	protected:
		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	public:
		static const std::wstring Name;
		static const std::wstring ClearBufferName;

	private:
		ConstantBufferRTShadows m_cbShadowRaytrace;
		ConstantBufferClearBuffers m_cbClearBuffers;
		DX12Lib::LightComponent* m_lightComponent;
	};
}



