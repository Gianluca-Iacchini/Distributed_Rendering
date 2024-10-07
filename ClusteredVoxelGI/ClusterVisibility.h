#pragma once

#include "BufferManager.h"
#include "RayTracingHelpers.h"
#include "RaytracingStateObject.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "Shaders/TechniquesCompat.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
	class SceneCamera;
}

namespace CVGI
{

	class ClusterVisibility
	{
	public:
		ClusterVisibility(DirectX::XMUINT3 voxelTexDimensions) : m_voxelTexDimensions(voxelTexDimensions) {}
		~ClusterVisibility() {}

		void InitializeBuffers(UINT voxelCount, UINT clusterCount);
		void StartVisibility(DX12Lib::ComputeContext& context, BufferManager& compactBufferManager);
		void VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager);

		void StartAABBGeneration(DX12Lib::ComputeContext& context, BufferManager& compactBufferManager, BufferManager& clusterBufferManager);
		void AABBGenerationPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager);
		
		void ClusterRayTrace(BufferManager& compactBufferManager, BufferManager& clusterBufferManager);
		void ClusterRayTracePass(RayTracingContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager);

		BufferManager* GetFaceBufferManager() { return &m_faceBufferManager; }
		BufferManager* GetAABBBufferManager() { return &m_aabbBufferManager; }
		BufferManager* GetRayTracingBufferManager() { return &m_raytracingBufferManager; }
		TopLevelAccelerationStructure& GetTLAS() { return m_TLAS; }

		UINT32 GetFaceCount() { return m_numberOfFaces; }

		std::shared_ptr<DX12Lib::RootSignature> BuildFaceCountRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

		std::shared_ptr<DX12Lib::RootSignature> BuildAABBGenerationRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildAABBGenerationPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

		std::shared_ptr<DX12Lib::RootSignature> BuildRaytracingGlobalRootSignature();
		std::shared_ptr<CVGI::RaytracingStateObject> BuildRayTracingPipelineState(std::shared_ptr<DX12Lib::RootSignature> globalRootSig);

		void BuildAccelerationStructures(DX12Lib::ComputeContext& context);

	private:
		DirectX::XMUINT3 m_voxelTexDimensions;
		BufferManager m_faceBufferManager;
		BufferManager m_aabbBufferManager;
		BufferManager m_raytracingBufferManager;

		ConstantBufferFaceCount m_cbFaceCount;
		ConstantBufferAABBGeneration m_cbAABBGeneration;
		RTSceneVisibility m_cbRayTracing;

		const std::wstring FaceCountPsoName = L"FACE_COUNT_PSO";
		const std::wstring AABBGenerationPsoName = L"AABB_GENERATION_PSO";

		UINT32 m_numberOfVoxels = 0;
		UINT32 m_numberOfClusters = 0;
		UINT32 m_numberOfFaces = 0;
		UINT32 m_gridOccupiedCount = 0;

		TopLevelAccelerationStructure m_TLAS;

	private:
		enum class FaceCountRootSignature
		{
			FaceCountCBV = 0,
			CompactSRVTable = 1,
			FaceCountUAVTable,
			Count
		};
		enum class AABBGenerationRootSignature
		{
			AABBGenerationCBV = 0,
			CompactSRVTable = 1,
			ClusterSRVTable,
			AABBGenerationUAVTable,
			Count
		};
		enum class RayTraceRootSignature
		{
			RayTraceCBV = 0,
			AccelerationStructureSRV = 1,
			CompactSRVTable = 2,
			ClusterSRVTable = 3,
			FaceSRVTable,
			AABBSRVTable,
			RayTraceUAVTable,
			Count
		};
	};

}

