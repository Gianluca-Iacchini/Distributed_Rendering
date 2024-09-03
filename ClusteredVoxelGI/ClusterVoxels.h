#pragma once
#include <DirectXMath.h>
#include "BufferManager.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
}

namespace CVGI
{
	class ClusterVoxels
	{
	private:
		__declspec(align(16)) struct ClusterData
		{
			DirectX::XMFLOAT3 Center;
			UINT32 VoxelCount;

			DirectX::XMFLOAT3 Normal;
			UINT32 FirstVoxelDataIndex;

		};

		__declspec(align(16)) struct ConstantBufferClusterizeBuffer
		{
			UINT32 CurrentPhase = 0;
			UINT32 VoxelCount = 0;
			UINT32 K = 10000;
			float m = 1.0f;

			DirectX::XMUINT3 VoxelTextureDimensions = DirectX::XMUINT3(128, 128, 128);
			UINT32 S = 1;

			DirectX::XMUINT3 TileGridDimension = DirectX::XMUINT3(6, 6, 6);
			UINT32 FirstClusterSet = 1;

			DirectX::XMUINT3 CurrentTileUpdate = DirectX::XMUINT3(0, 0, 0);
			float pad1 = 0.0f;
		};

		enum class ClusterizeRootSignature
		{
			ClusterizeCBV = 0,
			VoxelBuffersSRVTable = 1,
			StreamCompactionSRVTable = 2,
			ClusterizeUAVTable,
			Count
		};

	public:
		enum class ClusterBufferType
		{
			ClusterData,
			NextVoxel,
			AssignmentMap,
			DistanceMap,
			TileBuffer,
			NextCluster,
			ClusterCounterBuffer,
			VoxelNormalDirection,
			NextVoxelClusterData,
			SubClusterData,
		};

	public:
		ClusterVoxels(DirectX::XMUINT3 VoxelSceneSize) : m_voxelSceneDimensions(VoxelSceneSize) {}
		~ClusterVoxels() {}

		void InitializeBuffers(UINT VoxelCount);
		void StartClustering(BufferManager& voxelBufferManager, BufferManager& compactBufferManager);
		void ClusterPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& voxelBufferManager, BufferManager& compactBufferManager);

		std::shared_ptr<DX12Lib::RootSignature> BuildClusterizeRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildClusterizePipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

		BufferManager* GetBufferManager() { return &m_bufferManager; }

	private:
		DirectX::XMUINT3 m_voxelSceneDimensions;
		BufferManager m_bufferManager;

		ConstantBufferClusterizeBuffer m_cbClusterizeBuffer;

		float m_compactness = 10.0f;
		UINT32 m_numberOfClusters;
		UINT32 m_superPixelWidth;
		UINT32 m_voxelCount;

		DirectX::XMUINT3 m_tileGridDimension;

		const std::wstring ClusterPsoName = L"FAST_SLIC_PSO";
	};
}

