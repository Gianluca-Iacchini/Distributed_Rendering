#pragma once

#include "BufferManager.h"
#include "RayTracingHelpers.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
}

namespace CVGI
{

	class ClusterVisibility
	{
	private:
		__declspec(align(16)) struct ConstantBufferFaceCount
		{
			DirectX::XMUINT3 GridDimension;
			UINT32 CurrentPhase = 0;

			UINT32 VoxelCount = 0;
			DirectX::XMUINT3 pad1;
		};

		__declspec(align(16)) struct ConstantBufferAABBGeneration
		{
			DirectX::XMUINT3 GridDimension;
			UINT32 ClusterCount = 0;
		};

		__declspec(align(16)) struct AABBInfo
		{
			DirectX::XMFLOAT3 ClusterMin;
			UINT32 ClusterStartIndex;
			DirectX::XMFLOAT3 ClusterMax;
			UINT32 ClusterElementCount;
		};

		struct VoxelAABB
		{
			DirectX::XMFLOAT3 Min;
			DirectX::XMFLOAT3 Max;
		};

	public:
		ClusterVisibility(DirectX::XMUINT3 voxelTexDimensions) : m_voxelTexDimensions(voxelTexDimensions) {}
		~ClusterVisibility() {}

		void InitializeBuffers(UINT voxelCount, UINT clusterCount);
		void StartVisibility(DX12Lib::ComputeContext& context, BufferManager& compactBufferManager);
		void VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager);

		void StartAABBGeneration(DX12Lib::ComputeContext& context, BufferManager& compactBufferManager, BufferManager& clusterBufferManager);
		void AABBGenerationPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager);

		BufferManager* GetFaceBufferManager() { return &m_faceBufferManager; }
		BufferManager* GetAABBBufferManager() { return &m_aabbBufferManager; }

		UINT32 GetFaceCount() { return m_numberOfFaces; }

		std::shared_ptr<DX12Lib::RootSignature> BuildFaceCountRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

		std::shared_ptr<DX12Lib::RootSignature> BuildAABBGenerationRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildAABBGenerationPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

		void BuildAccelerationStructures(DX12Lib::ComputeContext& context);

	private:
		DirectX::XMUINT3 m_voxelTexDimensions;
		BufferManager m_faceBufferManager;
		BufferManager m_aabbBufferManager;

		ConstantBufferFaceCount m_cbFaceCount;
		ConstantBufferAABBGeneration m_cbAABBGeneration;

		const std::wstring FaceCountPsoName = L"FACE_COUNT_PSO";
		const std::wstring AABBGenerationPsoName = L"AABB_GENERATION_PSO";

		UINT32 m_numberOfVoxels = 0;
		UINT32 m_numberOfClusters = 0;
		UINT32 m_numberOfFaces = 0;

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
	};

}

