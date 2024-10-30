#include "FacePenaltyTechnique.h"
#include "DX12Lib/pch.h"
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "FaceCountTechnique.h"
#include "ClusterVisibility.h"
#include "ClusterVoxels.h"
#include "FacePenalty_CS.h"
#include "DX12Lib/Scene/LightComponent.h"

using namespace CVGI;
using namespace DirectX;
using namespace DX12Lib;
using namespace Graphics;

CVGI::FacePenaltyTechnique::FacePenaltyTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->AddBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::FacePenaltyTechnique::InitializeBuffers()
{
	// Face penalty cluster buffer (u0)
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(float));
	// Face penalty close voxels buffer (u1)
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(float));

	m_bufferManager->AllocateBuffers();
}

void CVGI::FacePenaltyTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	m_cbFacePenalty.VoxelCount = m_data->VoxelCount * 6;
	m_cbFacePenalty.LightDirection = m_data->GetLightComponent()->Node->GetForward();
	m_cbFacePenalty.LightPosition = m_data->GetLightComponent()->Node->GetPosition();
	m_cbFacePenalty.LightIntensity = 15.0f;
	TechniquePass(context, DirectX::XMUINT3(ceil(m_data->FaceCount / 128.0f), 1, 1));
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::FacePenaltyTechnique::BuildPipelineState()
{
	std::shared_ptr<RootSignature> rootSig = BuildRootSignature();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pFacePenalty_CS, ARRAYSIZE(g_pFacePenalty_CS));


	std::shared_ptr<ComputePipelineState> facePenaltyPso = std::make_shared<ComputePipelineState>();
	facePenaltyPso->SetRootSignature(rootSig);
	facePenaltyPso->SetComputeShader(shaderBlob);
	facePenaltyPso->Finalize();
	facePenaltyPso->Name = Name;

	return facePenaltyPso;
}

void CVGI::FacePenaltyTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterVoxelBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);
	auto& clusterVisiblityBufferManager = m_data->GetBufferManager(ClusterVisibility::Name);

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterVoxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterVisiblityBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FacePenaltyRootSignature::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FacePenaltyRootSignature::FacePenaltyCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFacePenalty).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FacePenaltyRootSignature::VoxelSRVTable, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FacePenaltyRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FacePenaltyRootSignature::ClusterVoxelSRVTable, clusterVoxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FacePenaltyRootSignature::ClusterVisibilitySRVTable, clusterVisiblityBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FacePenaltyRootSignature::FacePenaltyUAVTable, m_bufferManager->GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::FacePenaltyTechnique::BuildRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> facePenaltyRootSig = std::make_shared<DX12Lib::RootSignature>((UINT)FacePenaltyRootSignature::Count, 0);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::VoxelCommonsCBV].InitAsConstantBuffer(0);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::FacePenaltyCBV].InitAsConstantBuffer(1);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::VoxelSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::ClusterVoxelSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::ClusterVisibilitySRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 3);
	(*facePenaltyRootSig)[(UINT)FacePenaltyRootSignature::FacePenaltyUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

	facePenaltyRootSig->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return facePenaltyRootSig;
}



const std::wstring CVGI::FacePenaltyTechnique::Name = L"FacePenaltyTechnique";
