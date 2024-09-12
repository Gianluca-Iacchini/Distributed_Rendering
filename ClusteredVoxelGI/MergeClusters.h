#pragma once


#include "ClusterVoxels.h"


namespace CVGI
{
	class MergeClusters
	{
	private:
		enum class ClusterReduceRootSignature
		{
			ClusterReduceCBV = 0,
			ClusterizeSRVTable = 1,
			ClusterizeUAVTable = 2,
			ReduceUAVTable = 3,
			Count
		};

		__declspec(align(16)) struct ConstantBufferClusterReduce
		{
			UINT32 CurrentStep = 0;
			UINT32 NumberOfSubClusters = 0;
			UINT32 NumberOfSuperClusters = 0;
			float Compactness = 10.0f;

			DirectX::XMUINT3 VoxelDimension = DirectX::XMUINT3(512, 512, 512);
			UINT32 VoxelCount = 0;


			DirectX::XMUINT3 TileGridDimension = DirectX::XMUINT3(6, 6, 6);
			UINT32 S = 1;

			DirectX::XMUINT3 PreviousTileDimension = DirectX::XMUINT3(6, 6, 6);
			UINT32 PreviousS = 1;

			UINT32 CurrentIteration = 1;
			UINT32 FirstClusterSet = 0;
			float _pad1 = 0.0f;
			float _pad2 = 0.0f;
		};


	public:
		MergeClusters(DirectX::XMUINT3 voxelDimensions) : m_voxelTexDimension(voxelDimensions) {}
		~MergeClusters() {}

		BufferManager* GetBufferManager() { return &m_bufferManager; }

		void InitializeBuffers(DX12Lib::CommandContext& context, ClusterVoxels& clusterVoxels);
		void StartMerging(DX12Lib::ComputeContext& context, BufferManager& compactBufferManager);
		void MergeClusterPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager);

		std::shared_ptr<DX12Lib::RootSignature> BuildMergeClustersRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildMergeClustersPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSignature);

		UINT32 GetClusterCount() { return m_numberOfSubClusters; }

	private:
		ConstantBufferClusterReduce m_cbMergeClusters;
		BufferManager m_bufferManager;
		BufferManager* m_clusterBufferManager;

		DirectX::XMUINT3 m_voxelTexDimension;
		DirectX::XMUINT3 m_tileGridDimension;
		UINT32 m_numberOfSubClusters;
		UINT32 m_numberOfSuperClusters;
		UINT32 m_voxelCount;
		UINT32 m_superPixelWidth;



		const std::wstring MergeClustersPsoName = L"MergeClusters";
	};
}



