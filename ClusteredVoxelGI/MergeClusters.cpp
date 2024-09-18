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
	m_cbMergeClusters.VoxelDimension = DirectX::XMUINT3(UINT(m_voxelTexDimension.x), UINT(m_voxelTexDimension.y), UINT(m_voxelTexDimension.z));
	m_cbMergeClusters.Compactness = clusterVoxels.GetCompactness();
	m_cbMergeClusters.VoxelCount = clusterVoxels.GetVoxelCount();

	m_numberOfSubClusters = clusterVoxels.GetNonEmptyClusters();
	m_numberOfSuperClusters = clusterVoxels.GetNumberOfClusters();

	m_tileGridDimension = clusterVoxels.GetTileGridDimension();
	m_superPixelWidth = clusterVoxels.GetSuperPixelWidth();
	m_voxelCount = clusterVoxels.GetVoxelCount();

	m_clusterBufferManager = clusterVoxels.GetBufferManager();

	m_bufferManager.AddStructuredBuffer(m_numberOfSubClusters, sizeof(UINT32));
	m_bufferManager.AddStructuredBuffer(m_numberOfSubClusters, sizeof(UINT32));
	m_bufferManager.Add3DTextureBuffer(clusterVoxels.GetTileGridDimension(), DXGI_FORMAT_R32_UINT);
	m_bufferManager.AddStructuredBuffer(m_numberOfSubClusters, sizeof(UINT32));
	m_bufferManager.AddStructuredBuffer(m_numberOfSuperClusters, sizeof(ClusterVoxels::ClusterData));

	m_bufferManager.AllocateBuffers();

	context.Flush(true);
}

void MergeClusters::StartMerging(ComputeContext& context, BufferManager& compactBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"ReduceClusters");

	float currentSubd = 1;

	UINT32 voxelLinearSize = m_voxelTexDimension.x * m_voxelTexDimension.y * m_voxelTexDimension.z;

	m_cbMergeClusters.NumberOfSubClusters = m_numberOfSubClusters;
	m_cbMergeClusters.NumberOfSuperClusters = m_numberOfSuperClusters;
	m_cbMergeClusters.TileGridDimension = m_tileGridDimension;
	m_cbMergeClusters.PreviousTileDimension = m_tileGridDimension;
	m_cbMergeClusters.S = m_superPixelWidth;
	m_cbMergeClusters.PreviousS = m_superPixelWidth;
	m_cbMergeClusters.MaxClusterCount = m_numberOfSuperClusters;


	// We set m_numberOfSuperClusters to m_numberOfSubClusters, because in the first pass we are resetting all the buffers, so we passed
	// the total number of cluster is the SuperClusterCount field, now we are setting it to the SubClusterCount which is the actual
	// Number of subclusters we have.
	// This will then be subdivided by 2 each iteration.
	m_numberOfSuperClusters = m_numberOfSubClusters;

	bool isLessThan10k = false;

	for (UINT it = 0; it < 5 && !isLessThan10k; it++)
	{
		currentSubd = currentSubd * 2;

		m_cbMergeClusters.PreviousS = m_superPixelWidth;
		m_cbMergeClusters.PreviousTileDimension = m_tileGridDimension;

		m_numberOfSuperClusters = m_numberOfSubClusters / 2;
		m_superPixelWidth = ceilf(cbrtf((float)voxelLinearSize / m_numberOfSuperClusters));

		float denominator = 2.0f * m_superPixelWidth;
		m_tileGridDimension = { (UINT)ceilf(m_voxelTexDimension.x / denominator), (UINT)ceilf(m_voxelTexDimension.y / denominator), (UINT)ceilf(m_voxelTexDimension.z / denominator) };


		m_cbMergeClusters.CurrentIteration = currentSubd;
		m_cbMergeClusters.NumberOfSubClusters = m_numberOfSubClusters;
		m_cbMergeClusters.NumberOfSuperClusters = m_numberOfSuperClusters;
		m_cbMergeClusters.TileGridDimension = m_tileGridDimension;
		m_cbMergeClusters.S = m_superPixelWidth;
		m_cbMergeClusters.CurrentStep = 0;
		m_cbMergeClusters.FirstClusterSet = 0;
		m_cbMergeClusters.UnassignedPassOnly = 0;

		isLessThan10k = m_numberOfSuperClusters < 10000;

		DirectX::XMUINT3 groupSize =
		{
			(UINT)ceil(m_tileGridDimension.x / (8.0f)),
			(UINT)ceil(m_tileGridDimension.y / (8.0f)),
			(UINT)ceil(m_tileGridDimension.z / (8.0f))
		};

		m_cbMergeClusters.CurrentStep = 0;
		MergeClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), compactBufferManager);
		context.Flush();

		m_cbMergeClusters.CurrentStep = 1;
		MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSubClusters / 512), 1, 1), compactBufferManager);
		context.Flush();

		m_cbMergeClusters.CurrentStep = 2;
		MergeClusterPass(context, groupSize, compactBufferManager);
		context.Flush();

		for (int n = 0; n < 2; n++)
		{
			m_cbMergeClusters.FirstClusterSet = n == 0 ? 0 : 1;
			m_cbMergeClusters.CurrentStep = 3;
			MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSuperClusters / 512), 1, 1), compactBufferManager);
			context.Flush();

			m_cbMergeClusters.CurrentStep = 4;
			MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSubClusters / 512), 1, 1), compactBufferManager);
			context.Flush();

			m_cbMergeClusters.CurrentStep = 5;
			MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_voxelCount / 512), 1, 1), compactBufferManager);
			context.Flush();


		}
		m_cbMergeClusters.CurrentStep = 6;
		MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSuperClusters / 512), 1, 1), compactBufferManager);
		context.Flush();

		context.CopyBuffer(
			m_clusterBufferManager->GetBuffer((UINT)ClusterVoxels::ClusterBufferType::ClusterData),
			m_bufferManager.GetBuffer(4) // Voxel Temp Buffer
		);
		context.Flush();

		m_cbMergeClusters.CurrentStep = 4;
		m_cbMergeClusters.UnassignedPassOnly = 1;
		MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSubClusters / 512), 1, 1), compactBufferManager);
		context.Flush();

		m_cbMergeClusters.CurrentStep = 7;
		MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_voxelCount / 512), 1, 1), compactBufferManager);
		context.Flush();

		m_cbMergeClusters.CurrentStep = 8;
		MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_voxelCount / 512), 1, 1), compactBufferManager);
		context.Flush();

		context.CopyBuffer(
			m_clusterBufferManager->GetBuffer((UINT)ClusterVoxels::ClusterBufferType::SubClusterData),
			m_clusterBufferManager->GetBuffer((UINT)ClusterVoxels::ClusterBufferType::ClusterData)
		);

		m_numberOfSubClusters = *m_clusterBufferManager->ReadFromBuffer<UINT32*>(context, (UINT)ClusterVoxels::ClusterBufferType::Counter);
		

	}

	m_cbMergeClusters.CurrentStep = 9;
	MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_numberOfSubClusters / 512), 1, 1), compactBufferManager);
	context.Flush();

	m_cbMergeClusters.CurrentStep = 10;
	MergeClusterPass(context, DirectX::XMUINT3(ceilf(m_voxelCount / 512), 1, 1), compactBufferManager);
	context.Flush();

	PIXEndEvent(context.m_commandList->Get());
	context.Flush(true);
}

void CVGI::MergeClusters::MergeClusterPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager)
{
	m_clusterBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.FlushResourceBarriers();

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[MergeClustersPsoName].get());



	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ClusterReduceRootSignature::ClusterReduceCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbMergeClusters).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterReduceRootSignature::ClusterizeSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterReduceRootSignature::ClusterizeUAVTable, m_clusterBufferManager->GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterReduceRootSignature::ReduceUAVTable, m_bufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::MergeClusters::BuildMergeClustersRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ClusterReduceRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterReduceCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterizeSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ClusterizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 8, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterReduceRootSignature::ReduceUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 1);

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
