#include "LightVoxel.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

#include "BuildAABBsTechnique.h"
#include "ClusterVisibility.h"
#include "FaceCountTechnique.h"
///
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"

#include "DX12Lib/Scene/LightComponent.h"


using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

CVGI::LightVoxel::LightVoxel(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->AddBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::LightVoxel::InitializeBuffers()
{
	UINT32 voxelBitCount = (m_data->VoxelCount + 31) / 32;

	m_bufferManager->AddByteAddressBuffer(voxelBitCount);
	m_bufferManager->AddByteAddressBuffer(voxelBitCount);

	m_bufferManager->AllocateBuffers();
}

void CVGI::LightVoxel::PerformTechnique(RayTracingContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), Name.c_str());

	m_cbShadowRaytrace.LightDirection = m_lightComponent->Node->GetForward();
	m_cbShadowRaytrace.FaceCount = m_data->FaceCount;

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);


	context.CopyBuffer(m_bufferManager->GetBuffer(1), m_bufferManager->GetBuffer(0));


	auto& shadowUavBuffer = m_bufferManager->GetBuffer(0);

	UINT clearValues[4] = { 0, 0, 0, 0 };

	UINT side = floor(std::cbrt(m_data->FaceCount));
	DirectX::XMUINT3 dispatchSize = DirectX::XMUINT3(side + 1, side + 1, side);

	assert(dispatchSize.x * dispatchSize.y * dispatchSize.z >= m_data->FaceCount);

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
	context.m_commandList->Get()->ClearUnorderedAccessViewUint(m_bufferManager->GetUAVHandle(), shadowUavBuffer.GetUAV(), shadowUavBuffer.Get(), clearValues, 0, nullptr);
	TechniquePass(context, dispatchSize);


	PIXEndEvent(context.m_commandList->Get());

}

void CVGI::LightVoxel::TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name.c_str()].get());

	auto& prefixSumBuffer = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& faceBufferManager = m_data->GetBufferManager(FaceCountTechnique::Name);

	faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::ShadowCommonCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbShadowRaytrace).GpuAddress());
	context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)RayTraceShadowRootSignature::AccelerationStructureSRV, m_data->GetTlas()->GetGpuVirtualAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::PrefixSumBufferSRV, prefixSumBuffer.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::FaceBufferSRV, faceBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::RayTraceShadowTableUAV, m_bufferManager->GetUAVHandle());



	context.DispatchRays3D(groupSize.x, groupSize.y, groupSize.z);
}



std::shared_ptr<DX12Lib::RootSignature> CVGI::LightVoxel::BuildRootSignature()
{
	DX12Lib::SamplerDesc ShadowSamplerDesc;
	ShadowSamplerDesc.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	ShadowSamplerDesc.SetTextureAddressMode(D3D12_TEXTURE_ADDRESS_MODE_BORDER);
	ShadowSamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	ShadowSamplerDesc.MipLODBias = 0.0f;
	ShadowSamplerDesc.MaxAnisotropy = 1;
	ShadowSamplerDesc.SetBorderColor(DX12Lib::Color::Black());

	std::shared_ptr<DX12Lib::RootSignature> rayTracingRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)RayTraceShadowRootSignature::Count, 1);

	rayTracingRootSignature->InitStaticSampler(0, ShadowSamplerDesc);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::ShadowCommonCBV].InitAsConstantBuffer(1);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::AccelerationStructureSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::PrefixSumBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::FaceBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::RayTraceShadowTableUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

	rayTracingRootSignature->Finalize();

	return rayTracingRootSignature;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::LightVoxel::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

	std::shared_ptr<RaytracingStateObject> rayTracingPso = std::make_shared<RaytracingStateObject>();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pRaytracingShadow, ARRAYSIZE(g_pRaytracingShadow));

	rayTracingPso->SetRootSignature(rootSig);
	rayTracingPso->SetShaderBytecode(shaderBlob);
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Raygen, L"ShadowRaygen");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Miss, L"ShadowMiss");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Intersection, L"ShadowIntersection");
	rayTracingPso->SetAttributeAndPayloadSize(sizeof(UINT32), sizeof(UINT32));
	rayTracingPso->SetRecursionDepth(1);

	auto& hitGroup = rayTracingPso->CreateHitGroup(L"ShadowHitGroup");
	hitGroup.AddIntersectionShader(L"ShadowIntersection");
	hitGroup.SetHitGroupType(D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE);

	rayTracingPso->Finalize();
	rayTracingPso->Name = Name;

	return rayTracingPso;
}

const std::wstring CVGI::LightVoxel::Name = L"LightVoxel";