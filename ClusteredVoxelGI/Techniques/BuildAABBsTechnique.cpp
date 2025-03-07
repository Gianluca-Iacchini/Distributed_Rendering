#include "BuildAABBsTechnique.h"
#include "DX12Lib/pch.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"
#include "../Data/Shaders/Include/ComputeAABB_CS.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::BuildAABBsTechnique::InitializeBuffers()
{
	UINT32 voxelCount = m_data->GetVoxelCount();

	// Buffers for acceleration structures
	// Voxel AABB data (u0);
	m_bufferManager->AddStructuredBuffer(voxelCount, sizeof(AABB));
	// Cluster Start Index and Count (u1);
	DirectX::XMUINT3 groupSize = MathHelper::Ceil(m_data->GetVoxelGridSize(), 8);
	m_bufferManager->AddStructuredBuffer(groupSize.x * groupSize.y * groupSize.z, sizeof(ClusterAABBInfo));
	// AABB voxel indices (u2);
	m_bufferManager->AddStructuredBuffer(voxelCount, sizeof(UINT32));
	// Cluster AABB Counter (u3);
	m_bufferManager->AddByteAddressBuffer();
	// Debug AABB Coutner (u4);
	m_bufferManager->AddByteAddressBuffer();

	m_bufferManager->AllocateBuffers();
}

void CVGI::BuildAABBsTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"AABBGeneration");

	m_cbAABBGeneration.GridDimension = m_data->GetVoxelGridSize();
	m_cbAABBGeneration.ClusterCount = m_data->GetClusterCount();

	TechniquePass(context, MathHelper::Ceil(m_data->GetVoxelGridSize(), 8));

	m_data->AABBGeometryGroupCount = *m_bufferManager->ReadFromBuffer<UINT32*>(context, 4);

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}

void CVGI::BuildAABBsTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(m_techniquePSO.get());

	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers();
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)AABBGenerationRootSignature::AABBGenerationCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbAABBGeneration).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::ClusterSRVTable, clusterBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::AABBGenerationUAVTable, m_bufferManager->GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::BuildAABBsTechnique::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> AABBComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)AABBGenerationRootSignature::Count, 1);
	AABBComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationCBV].InitAsConstantBuffer(0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::ClusterSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 0);

	AABBComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return AABBComputeRootSignature;
}

void CVGI::BuildAABBsTechnique::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

	auto shaderByteCode = CD3DX12_SHADER_BYTECODE((void*)g_pComputeAABB_CS, ARRAYSIZE(g_pComputeAABB_CS));

	std::unique_ptr<ComputePipelineState> AABBGenerationComputePso = std::make_unique<ComputePipelineState>();
	AABBGenerationComputePso->SetRootSignature(rootSig);
	AABBGenerationComputePso->SetComputeShader(shaderByteCode);
	AABBGenerationComputePso->Finalize();
	AABBGenerationComputePso->Name = Name;

	m_techniquePSO = std::move(AABBGenerationComputePso);
}



const std::wstring BuildAABBsTechnique::Name = L"BuildAABBsTechnique";
