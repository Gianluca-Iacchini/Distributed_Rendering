#include "BuildAABBsTechnique.h"
#include "DX12Lib/pch.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::BuildAABBsTechnique::InitializeBuffers()
{
	// Buffers for acceleration structures
	// Voxel AABB data (u0);
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(AABB));
	// Cluster Start Index and Count (u1);
	DirectX::XMUINT3 groupSize = MathHelper::Ceil(m_data->VoxelGridSize, 8);
	m_bufferManager->AddStructuredBuffer(groupSize.x * groupSize.y * groupSize.z, sizeof(AABBInfo));
	// AABB voxel indices (u2);
	m_bufferManager->AddStructuredBuffer(m_data->VoxelCount, sizeof(UINT32));
	// Cluster AABB Counter (u3);
	m_bufferManager->AddByteAddressBuffer();
	// Debug AABB Coutner (u4);
	m_bufferManager->AddByteAddressBuffer();

	m_bufferManager->AllocateBuffers();
}

void CVGI::BuildAABBsTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"AABBGeneration");

	m_cbAABBGeneration.GridDimension = m_data->VoxelGridSize;
	m_cbAABBGeneration.ClusterCount = m_data->ClusterCount;

	TechniquePass(context, MathHelper::Ceil(m_data->VoxelGridSize, 8));

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}

void CVGI::BuildAABBsTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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

std::shared_ptr<DX12Lib::PipelineState> CVGI::BuildAABBsTechnique::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\ComputeAABB_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> AABBGenerationComputePso = std::make_shared<ComputePipelineState>();
	AABBGenerationComputePso->SetRootSignature(rootSig);
	AABBGenerationComputePso->SetComputeShader(computeShader);
	AABBGenerationComputePso->Finalize();
	AABBGenerationComputePso->Name = Name;

	return AABBGenerationComputePso;
}



const std::wstring BuildAABBsTechnique::Name = L"BuildAABBsTechnique";
