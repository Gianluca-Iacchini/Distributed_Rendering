#include "LightVoxel.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

#include "BuildAABBsTechnique.h"
#include "ClusterVisibility.h"
///
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"

#include "DX12Lib/Scene/LightComponent.h"

#include "../Data/Shaders/Include/LitVoxels_CS.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;
using namespace VOX;

CVGI::LightVoxel::LightVoxel(std::shared_ptr<VOX::TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->SetBufferManager(Name, m_bufferManager);
	m_data = data;
	m_cbShadowRaytrace.FrameCount = 0;
}

void CVGI::LightVoxel::InitializeBuffers()
{
	UINT32 voxelBitSize = (m_data->GetVoxelCount() + 31) / 32;

	m_bufferManager->AddByteAddressBuffer(voxelBitSize);
	m_bufferManager->AddStructuredBuffer(m_data->GetClusterCount(), sizeof(XMUINT4));

	m_bufferManager->AllocateBuffers();
}

void CVGI::LightVoxel::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), Name.c_str());

	UINT voxelCount = m_data->GetVoxelCount();

	m_cbClearBuffers.ValueCount0 = voxelCount;
	m_cbClearBuffers.ValueCount1 = m_data->GetClusterCount();

	ClearBufferPass(context, DirectX::XMUINT3(ceilf(voxelCount / 128.0f), 1, 1));

	DirectX::XMUINT3 dispatchSize = DirectX::XMUINT3(ceilf(voxelCount / 128.0f), 1, 1);

	TechniquePass(context, dispatchSize);

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::LightVoxel::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(m_techniquePSO.get());

	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& prefixSumBuffer = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterVoxelBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	prefixSumBuffer.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterVoxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers();
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ShadowRootSignature::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ShadowRootSignature::LightCommonCBV, m_data->GetLightCameraResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ShadowRootSignature::ShadowTextureSRV, m_data->GetLightCameraSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ShadowRootSignature::VoxelSRV, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ShadowRootSignature::CompactSRV, prefixSumBuffer.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ShadowRootSignature::ClusterSRV, clusterVoxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ShadowRootSignature::LightVoxelUAV, m_bufferManager->GetUAVHandle());
	
	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}


void CVGI::LightVoxel::ClearBufferPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(m_clearBufferPso.get());

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers();
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView(0, Renderer::s_graphicsMemory->AllocateConstant(m_cbClearBuffers).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable(1, m_bufferManager->GetUAVHandle());
	
	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::LightVoxel::BuildRootSignature()
{
	SamplerDesc DefaultSamplerDesc;
	DefaultSamplerDesc.MaxAnisotropy = 8;

	SamplerDesc ShadowSamplerDesc;
	ShadowSamplerDesc.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	ShadowSamplerDesc.SetTextureAddressMode(D3D12_TEXTURE_ADDRESS_MODE_BORDER);
	ShadowSamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	ShadowSamplerDesc.MipLODBias = 0.0f;
	ShadowSamplerDesc.MaxAnisotropy = 1;
	ShadowSamplerDesc.SetBorderColor(Color::Black());

	std::shared_ptr<DX12Lib::RootSignature> shadowRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ShadowRootSignature::Count, 2);

	shadowRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
	shadowRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::LightCommonCBV].InitAsConstantBuffer(1);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::ShadowTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::VoxelSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::CompactSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::ClusterSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 3);
	(*shadowRootSignature)[(UINT)ShadowRootSignature::LightVoxelUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

	shadowRootSignature->Finalize();

	return shadowRootSignature;
}


void CVGI::LightVoxel::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

	std::unique_ptr<DX12Lib::ComputePipelineState> shadowPso = std::make_unique<DX12Lib::ComputePipelineState>();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pLitVoxels_CS, ARRAYSIZE(g_pLitVoxels_CS));

	shadowPso->SetRootSignature(rootSig);
	shadowPso->SetComputeShader(shaderBlob);
	shadowPso->Finalize();
	shadowPso->Name = Name;

	m_techniquePSO = std::move(shadowPso);

	////////////
	BuildClearBufferPso();
	/////////////
}

void CVGI::LightVoxel::BuildClearBufferPso()
{
	std::shared_ptr<DX12Lib::RootSignature> rootSignature = std::make_shared<DX12Lib::RootSignature>(2, 0);

	(*rootSignature)[0].InitAsConstantBuffer(0);
	(*rootSignature)[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);
	rootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pClearBufferShader_CS, ARRAYSIZE(g_pClearBufferShader_CS));

	m_clearBufferPso = std::make_unique<DX12Lib::ComputePipelineState>();
	m_clearBufferPso->SetComputeShader(shaderBlob);
	m_clearBufferPso->SetRootSignature(rootSignature);
	m_clearBufferPso->Finalize();
	m_clearBufferPso->Name = ClearBufferName;

}

const std::wstring CVGI::LightVoxel::Name = L"LightVoxel";
const std::wstring CVGI::LightVoxel::ClearBufferName = L"ClearLightBuffer";