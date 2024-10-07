#include "ClusterVoxels.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::ClusterVoxels::InitializeBuffers(UINT voxelCount)
{
	m_voxelCount = voxelCount;

	UINT32 voxelLinearSize = m_voxelSceneDimensions.x * m_voxelSceneDimensions.y * m_voxelSceneDimensions.z;
	m_numberOfClusters = MathHelper::Min(250000u, (UINT32)(m_voxelCount / 10));
	m_superPixelWidth = cbrtf((float)voxelLinearSize / m_numberOfClusters);

	float denominator = 2.0f * m_superPixelWidth;
	m_tileGridDimension = { (UINT)ceilf(m_voxelSceneDimensions.x / denominator), (UINT)ceilf(m_voxelSceneDimensions.y / denominator), (UINT)ceilf(m_voxelSceneDimensions.z / denominator) };

	// ClusterData (u0)
	m_bufferManager.AddStructuredBuffer(m_numberOfClusters, sizeof(ClusterData));

	// To Remove
	// NextVoxelInCluster (u1)
	m_bufferManager.AddStructuredBuffer(m_voxelCount, sizeof(UINT32));

	// Assignment Map (u2)
	m_bufferManager.AddStructuredBuffer(m_voxelCount, sizeof(UINT32));

	// 3D Tile Map (u3)
	m_bufferManager.Add3DTextureBuffer(m_tileGridDimension, DXGI_FORMAT_R32_UINT);

	// Next Cluster in tile (u4)
	m_bufferManager.AddStructuredBuffer(m_numberOfClusters, sizeof(UINT32));
	
	// Cluster Counter Buffer (u5)
	m_bufferManager.AddByteAddressBuffer();

	// Voxel Normal Direction (u6)
	m_bufferManager.AddStructuredBuffer(m_voxelCount, sizeof(DirectX::XMFLOAT3));

	// Buffer used to store the data for cluster merging.
	// Sub Cluster Data (u7)
	m_bufferManager.AddStructuredBuffer(m_numberOfClusters, sizeof(ClusterData));

	m_bufferManager.AllocateBuffers();

	m_cbClusterizeBuffer.CurrentPhase = 0;
	m_cbClusterizeBuffer.VoxelCount = m_voxelCount;
	m_cbClusterizeBuffer.S = ceil(m_superPixelWidth);
	m_cbClusterizeBuffer.m = m_compactness;
	m_cbClusterizeBuffer.K = m_numberOfClusters;
	m_cbClusterizeBuffer.VoxelTextureDimensions = DirectX::XMUINT3(UINT(m_voxelSceneDimensions.x), UINT(m_voxelSceneDimensions.y), UINT(m_voxelSceneDimensions.z));
	m_cbClusterizeBuffer.TileGridDimension = m_tileGridDimension;
	m_cbClusterizeBuffer.UnassignedOnlyPass = 0;
}

void CVGI::ClusterVoxels::StartClustering(ComputeContext& context, BufferManager& voxelBufferManager, BufferManager& compactBufferManager)
{

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"ClusterizePass");
	
	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	context.FlushResourceBarriers();

	m_cbClusterizeBuffer.CurrentPhase = 0;
	ClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
	context.Flush();

	m_cbClusterizeBuffer.CurrentPhase = 1;
	ClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
	context.Flush();


	for (int i = 0; i < 10; i++)
	{
		m_cbClusterizeBuffer.CurrentPhase = 2;
		ClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
		context.Flush();

		m_cbClusterizeBuffer.CurrentPhase = 3;
		ClusterPass(context, DirectX::XMUINT3(ceil(m_voxelCount / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
		context.Flush();

		m_cbClusterizeBuffer.CurrentPhase = 4;
		ClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
		context.Flush();
	}

	m_cbClusterizeBuffer.UnassignedOnlyPass = 1;
	m_cbClusterizeBuffer.CurrentPhase = 2;
	ClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
	context.Flush();

	m_cbClusterizeBuffer.CurrentPhase = 5;
	ClusterPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1), voxelBufferManager, compactBufferManager);
	context.Flush();

	context.CopyBuffer(m_bufferManager.GetBuffer((UINT)ClusterVoxels::ClusterBufferType::ClusterData), 
		m_bufferManager.GetBuffer((UINT)ClusterVoxels::ClusterBufferType::SubClusterData));

	m_numberOfNonEmptyClusters = *m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)ClusterBufferType::Counter);
	DXLIB_CORE_INFO("Cluster count: {0}", m_numberOfNonEmptyClusters);

	PIXEndEvent(context.m_commandList->Get());



	context.Flush();
}

void CVGI::ClusterVoxels::ClusterPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& voxelBufferManager, BufferManager& compactBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[ClusterPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ClusterizeRootSignature::ClusterizeCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbClusterizeBuffer).GpuAddress());

	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::VoxelBuffersSRVTable, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::StreamCompactionSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::ClusterizeUAVTable, m_bufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVoxels::BuildClusterizeRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ClusterizeRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::VoxelBuffersSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::StreamCompactionSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 8, D3D12_SHADER_VISIBILITY_ALL, 0);

	clusterRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return clusterRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVoxels::BuildClusterizePipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FastSlic_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelClusterizeComputePso = std::make_shared<ComputePipelineState>();
	voxelClusterizeComputePso->SetRootSignature(rootSig);
	voxelClusterizeComputePso->SetComputeShader(computeShader);
	voxelClusterizeComputePso->Finalize();
	voxelClusterizeComputePso->Name = ClusterPsoName;

	return voxelClusterizeComputePso;
}
