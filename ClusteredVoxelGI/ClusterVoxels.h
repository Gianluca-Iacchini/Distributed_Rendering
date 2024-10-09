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
	class ClusterVoxels : public Technique
	{
		struct ClusterData
		{
			XMUINT3 Center;
			UINT32 VoxelCount;

			XMFLOAT3 Normal;
			UINT32 FirstVoxelDataIndex;
		};

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
			NextVoxelLinkedList = 1,
			AssignmentMap,
			TileBuffer,
			NextClusterInTileLinkedList,
			Counter,
			VoxelNormalDirection,
			SubClusterData,
		};

	public:
		ClusterVoxels(std::shared_ptr<TechniqueData> data)
		{
			m_bufferManager = std::make_shared<BufferManager>();
			data->AddBufferManager(Name, m_bufferManager);
			m_data = data;	
		}
		~ClusterVoxels() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

	public:
		static const std::wstring Name;

	private:
		ConstantBufferClusterizeBuffer m_cbClusterizeBuffer;

		float m_compactness = 10.0f;
		UINT32 m_numberOfClusters;
		float m_superPixelWidth;

		UINT32 m_numberOfNonEmptyClusters = 0;

		DirectX::XMUINT3 m_tileGridDimension;


	};
}

