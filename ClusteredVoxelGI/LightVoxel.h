#pragma once

#include "BufferManager.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "Shaders/TechniquesCompat.h"
#include "RayTracingHelpers.h"
#include "RaytracingStateObject.h"
#include "RaytracingShadow.h"
#include "DX12Lib/Commons/CommonConstants.h"
#include "DX12Lib/DXWrapper/SamplerDesc.h"

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

	class LightVoxel
	{
	public:
		LightVoxel(DirectX::XMUINT3 voxelTexSize) : m_voxelTexDimensions(voxelTexSize) {}
		~LightVoxel() {}

		void InitializeBuffers(UINT voxelCount);
		void StartLightVoxel(DX12Lib::ShadowCamera& camera, BufferManager& compactBufferManager, BufferManager& aabbBufferManager,  BufferManager& rtBufferManager, TopLevelAccelerationStructure& tlas);
		void LightVoxelPass(RayTracingContext& context, DX12Lib::ShadowCamera& camera, BufferManager& compactBufferManager, BufferManager& aabbBufferManager, BufferManager& rtBufferManager, TopLevelAccelerationStructure& tlas);

		BufferManager* GetShadowBufferManager() { return &m_bufferManager; }

		std::shared_ptr<DX12Lib::RootSignature> BuildLightVoxelRootSignature();
		std::shared_ptr<RaytracingStateObject> BuildLightVoxelPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);
	
	private:
		ConstantBufferRTShadows m_cbShadowRaytrace;
		BufferManager m_bufferManager;
		DirectX::XMUINT3 m_voxelTexDimensions;
		UINT32 m_voxelCount;
	};
}



