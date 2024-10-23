#include "ClusterVoxels.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::ClusterVoxels::InitializeBuffers()
{

	DirectX::XMUINT3 voxelGridSize = m_data->GetVoxelGridSize();

	UINT32 voxelLinearSize = voxelGridSize.x * voxelGridSize.y * voxelGridSize.z;
	m_numberOfClusters = MathHelper::Min(250000u, (UINT32)(m_data->VoxelCount / 10));
	m_superPixelWidth = cbrtf((float)voxelLinearSize / m_numberOfClusters);

	float denominator = 2.0f * m_superPixelWidth;
	m_tileGridDimension = MathHelper::Ceil(voxelGridSize, denominator);

	// ClusterData (u0)
	m_bufferManager->AddStructuredBuffer(m_numberOfClusters, sizeof(ClusterData));


	// VoxelsInCluster (u1)
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(UINT32));

	// Assignment Map (u2)
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(UINT32));

	// Voxel color (u3)
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, 3 * sizeof(float));

	// Voxel Normal Direction (u4)
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(DirectX::XMFLOAT3));

	// 3D Tile Map (u5)
	m_bufferManager->Add3DTextureBuffer(m_tileGridDimension, DXGI_FORMAT_R32_UINT);

	// Next Cluster in tile (u6)
	m_bufferManager->AddStructuredBuffer(m_numberOfClusters, sizeof(UINT32));

	// Cluster Counter Buffer (u7)
	m_bufferManager->AddStructuredBuffer(2, sizeof(UINT32));

	// Buffer used to store the data for cluster merging.
	// Sub Cluster Data (u8)
	m_bufferManager->AddStructuredBuffer(m_numberOfClusters, sizeof(ClusterData));

	// Next voxel in cluster linked list (u9)
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(UINT32));

	m_bufferManager->AllocateBuffers();

	m_cbClusterizeBuffer.CurrentPhase = 0;
	m_cbClusterizeBuffer.VoxelCount = m_data->VoxelCount;
	m_cbClusterizeBuffer.S = ceil(m_superPixelWidth);
	m_cbClusterizeBuffer.m = m_compactness;
	m_cbClusterizeBuffer.K = m_numberOfClusters;
	m_cbClusterizeBuffer.VoxelTextureDimensions = voxelGridSize;
	m_cbClusterizeBuffer.TileGridDimension = m_tileGridDimension;
	m_cbClusterizeBuffer.UnassignedOnlyPass = 0;
}

void CVGI::ClusterVoxels::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), Name.c_str());


	m_cbClusterizeBuffer.CurrentPhase = 0;
	TechniquePass(context, DirectX::XMUINT3(ceil(m_data->VoxelCount / 512.0f), 1, 1));
	context.Flush();

	m_cbClusterizeBuffer.CurrentPhase = 1;
	TechniquePass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1));
	context.Flush();


	for (int i = 0; i < 10; i++)
	{
		m_cbClusterizeBuffer.CurrentPhase = 2;
		TechniquePass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1));
		context.Flush();

		m_cbClusterizeBuffer.CurrentPhase = 3;
		TechniquePass(context, DirectX::XMUINT3(ceil(m_data->VoxelCount / 512.0f), 1, 1));
		context.Flush();

		m_cbClusterizeBuffer.CurrentPhase = 4;
		TechniquePass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1));
		context.Flush();
	}

	m_cbClusterizeBuffer.UnassignedOnlyPass = 1;
	m_cbClusterizeBuffer.CurrentPhase = 2;
	TechniquePass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1));
	context.Flush();

	m_cbClusterizeBuffer.CurrentPhase = 5;
	TechniquePass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 512.0f), 1, 1));
	context.Flush();

	context.CopyBuffer(m_bufferManager->GetBuffer((UINT)ClusterVoxels::ClusterBufferType::ClusterData),
		m_bufferManager->GetBuffer((UINT)ClusterVoxels::ClusterBufferType::SubClusterData));

	m_numberOfNonEmptyClusters = *m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)ClusterBufferType::Counter);
	DXLIB_CORE_INFO("Cluster count: {0}", m_numberOfNonEmptyClusters);

	PIXEndEvent(context.m_commandList->Get());

	m_data->ClusterCount = m_numberOfNonEmptyClusters;

	context.Flush();
}

void CVGI::ClusterVoxels::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());


	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ClusterizeRootSignature::ClusterizeCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbClusterizeBuffer).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::VoxelBuffersSRVTable, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::StreamCompactionSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::ClusterizeUAVTable, m_bufferManager->GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVoxels::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ClusterizeRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::VoxelBuffersSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::StreamCompactionSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 10, D3D12_SHADER_VISIBILITY_ALL, 0);

	clusterRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return clusterRootSignature;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::ClusterVoxels::BuildPipelineState()
{
	std::shared_ptr<RootSignature> rootSig = BuildRootSignature();

	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FastSlic_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelClusterizeComputePso = std::make_shared<ComputePipelineState>();
	voxelClusterizeComputePso->SetRootSignature(rootSig);
	voxelClusterizeComputePso->SetComputeShader(computeShader);
	voxelClusterizeComputePso->Finalize();
	voxelClusterizeComputePso->Name = Name;

	return voxelClusterizeComputePso;
}

const std::wstring ClusterVoxels::Name = L"ClusterVoxels";


