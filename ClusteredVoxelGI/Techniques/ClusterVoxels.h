#pragma once
#include <DirectXMath.h>
#include "Technique.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
}

namespace CVGI
{
	class ClusterVoxels : public VOX::Technique
	{
	private:
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
			ClusterData = 0,
			VoxelsInCluster = 1,
			AssignmentMap,
			VoxelColor,
			VoxelNormalDirection,
			TileBuffer,
			NextClusterInTileLinkedList,
			Counter,
			SubClusterData,
			NextVoxelLinkedList
		};

	public:
		ClusterVoxels(std::shared_ptr<VOX::TechniqueData> data)
		{
			m_bufferManager = std::make_shared<VOX::BufferManager>();
			data->SetBufferManager(Name, m_bufferManager);
			m_data = data;	
		}
		~ClusterVoxels() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual void BuildPipelineState() override;

		UINT GetNumberOfClusters();

		void SetClusterizationLevel(int clusterizationLevel) { m_clusterizationLevel = clusterizationLevel; }
		int GetClusterizationLevl() { return m_clusterizationLevel; }

	public:
		static const std::wstring Name;

	private:
		::ConstantBufferClusterizeBuffer m_cbClusterizeBuffer;

		float m_compactness = 10.0f;
		UINT32 m_numberOfClusters;
		float m_superPixelWidth;

		int m_clusterizationLevel = 1;

		UINT32 m_numberOfNonEmptyClusters = 0;

		DirectX::XMUINT3 m_tileGridDimension;


	};
}

