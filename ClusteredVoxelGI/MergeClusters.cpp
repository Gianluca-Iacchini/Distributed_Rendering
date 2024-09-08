#include "MergeClusters.h"
#include "DX12Lib/pch.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::MergeClusters::InitializeBuffers(CommandContext& context, ClusterVoxels& clusterVoxels)
{
	BufferManager* clusterBufferManager = clusterVoxels.GetBufferManager();

	for (UINT i = 0; i < 10; i++)
	{
		if (i == 4)
			continue;

		std::wstring name = L"ClusterBuffer" + std::to_wstring(i);
		GPUBuffer& buffer = clusterBufferManager->GetBuffer(i);
		buffer.Get()->SetName(name.c_str());
	}

	m_cbMergeClusters.VoxelDimension = DirectX::XMUINT3(UINT(m_voxelTexDimension.x), UINT(m_voxelTexDimension.y), UINT(m_voxelTexDimension.z));
	m_cbMergeClusters.Compactness = clusterVoxels.GetCompactness();
	m_cbMergeClusters.VoxelCount = clusterVoxels.GetVoxelCount();

	m_numberOfSubClusters = clusterVoxels.GetNonEmptyClusters();
	m_numberOfSuperClusters = clusterVoxels.GetNumberOfClusters();

	m_tileGridDimension = clusterVoxels.GetTileGridDimension();
	m_superPixelWidth = clusterVoxels.GetSuperPixelWidth();

	ConvertClusterBuffers(context, clusterVoxels);

	context.Flush(true);
}

void MergeClusters::StartMerging(ComputeContext& context, BufferManager& compactBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"ReduceClusters");

	bool isLessThan10kClusters = m_numberOfSuperClusters < 10000;
	

	for (UINT it = 0; it < 5 && !isLessThan10kClusters; it++)
	{

		float currentIterationMultiplier = pow(2, it + 1);

		// Avoiding converting to float, we are simply checking if 1 tile (size of 2 * S) is larger than 1/4 of the voxel grid dimension
		if ((currentIterationMultiplier * m_superPixelWidth) * 4 > m_voxelTexDimension.x)
		{
			break;
		}


		m_numberOfSuperClusters = ceil(m_numberOfSuperClusters / 8);


		m_cbMergeClusters.CurrentIteration = it;
		m_cbMergeClusters.NumberOfSubClusters = m_numberOfSubClusters;
		m_cbMergeClusters.NumberOfSuperClusters = m_numberOfSuperClusters;
		m_cbMergeClusters.TileGridDimension = m_tileGridDimension;
		m_cbMergeClusters.S = m_superPixelWidth;
		m_cbMergeClusters.CurrentStep = 0;
		m_cbMergeClusters.FirstClusterSet = 0;


		DirectX::XMUINT3 groupSize =
		{
			(UINT)ceil(m_tileGridDimension.x / (8.0f)),
			(UINT)ceil(m_tileGridDimension.y / (8.0f)),
			(UINT)ceil(m_tileGridDimension.z / (8.0f))
		};

		UINT32 numberOfClusters = *m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)ClusterVoxels::ClusterBufferType::ClusterCounterBuffer);
		DXLIB_INFO("ClusterCounter before: {0}", numberOfClusters);


		MergeClusterPass(context, groupSize, compactBufferManager);
		context.Flush(true);

		numberOfClusters = *m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)ClusterVoxels::ClusterBufferType::ClusterCounterBuffer);
		DXLIB_INFO("Super cluster Filled: {0}", numberOfClusters);

		isLessThan10kClusters = numberOfClusters < 10000;

		if (isLessThan10kClusters)
		{
			DXLIB_WARN("Less than 10K clusters, ending at next iteration");
		}

		for (int n = 0; n < 10; n++)
		{
			m_cbMergeClusters.FirstClusterSet = n == 0 ? 0 : 1;



			m_cbMergeClusters.CurrentStep = 1;

			MergeClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfSuperClusters / 512.0f), 1, 1), compactBufferManager);
			context.Flush();


			m_cbMergeClusters.CurrentStep = 2;
			MergeClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfSubClusters / 512.0f), 1, 1), compactBufferManager);
			context.Flush();


			m_cbMergeClusters.CurrentStep = 3;
			MergeClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), compactBufferManager);
			context.Flush(true);
		}

		m_cbMergeClusters.CurrentStep = 4;
		MergeClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfSuperClusters / 512.0f), 1, 1), compactBufferManager);
		context.Flush(true);

		m_cbMergeClusters.CurrentStep = 5;
		MergeClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfSuperClusters / 512.0f), 1, 1), compactBufferManager);
		context.Flush(true);

		m_cbMergeClusters.CurrentStep = 6;
		MergeClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), compactBufferManager);
		context.Flush(true);

		m_numberOfSubClusters = *m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)ClusterVoxels::ClusterBufferType::ClusterCounterBuffer);

		m_bufferManager.ZeroBuffer(context, (UINT)ClusterVoxels::ClusterBufferType::ClusterCounterBuffer);
	}

	m_cbMergeClusters.CurrentStep = 7;
	m_cbMergeClusters.NumberOfSubClusters = m_numberOfSubClusters;
	MergeClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfSubClusters / 512.0f), 1, 1), compactBufferManager);
	context.Flush(true);

	m_cbMergeClusters.CurrentStep = 8;
	MergeClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), compactBufferManager);

	PIXEndEvent(context.m_commandList->Get());
	context.Flush(true);
}

void CVGI::MergeClusters::MergeClusterPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager)
{
	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);;
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.FlushResourceBarriers();

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[MergeClustersPsoName].get());



	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ClusterReduceRootSignature::ClusterReduceCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbMergeClusters).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterReduceRootSignature::ClusterizeSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterReduceRootSignature::ClusterizeUAVTable, m_bufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::MergeClusters::BuildMergeClustersRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ClusterReduceRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterReduceCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterizeSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 10, D3D12_SHADER_VISIBILITY_ALL, 0);

	clusterRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return clusterRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::MergeClusters::BuildMergeClustersPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSignature)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\ClusterReduce_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelClusterizeComputePso = std::make_shared<ComputePipelineState>();
	voxelClusterizeComputePso->SetRootSignature(rootSignature);
	voxelClusterizeComputePso->SetComputeShader(computeShader);
	voxelClusterizeComputePso->Finalize();
	voxelClusterizeComputePso->Name = MergeClustersPsoName;

	return voxelClusterizeComputePso;
}

void CVGI::MergeClusters::ConvertClusterBuffers(CommandContext& context, ClusterVoxels& clusterVoxels)
{
	auto& otherBufferManager = *clusterVoxels.GetBufferManager();
	otherBufferManager.MoveDataTo(m_bufferManager);

	UINT numberOfSubClusters = clusterVoxels.GetNumberOfClusters();

	m_bufferManager.ReplaceBuffer((UINT)ClusterVoxels::ClusterBufferType::VoxelNormalDirection, numberOfSubClusters, sizeof(UINT32));

	GPUBuffer& nextVoxelDataBuffer = m_bufferManager.GetBuffer((UINT)ClusterVoxels::ClusterBufferType::NextVoxelClusterData);
	GPUBuffer& nextClusterList = m_bufferManager.GetBuffer((UINT)ClusterVoxels::ClusterBufferType::NextCluster);

	context.CopyBufferRegion(nextVoxelDataBuffer, 0, nextClusterList, 0, m_numberOfSubClusters * sizeof(UINT32));

	m_bufferManager.ZeroBuffer(context, (UINT)ClusterVoxels::ClusterBufferType::ClusterCounterBuffer);
}
