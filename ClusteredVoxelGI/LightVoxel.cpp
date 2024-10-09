#include "LightVoxel.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

#include "BuildAABBsTechnique.h"
#include "PrefixSumVoxels.h"
#include "ClusterVisibility.h"
#include "DX12Lib/Scene/LightComponent.h"


using namespace CVGI;
using namespace DirectX;
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


	m_cbShadowRaytrace.GridDimension = m_data->VoxelGridSize;
	m_cbShadowRaytrace.ShadowTexDimensions = DirectX::XMUINT2(2048, 2048);

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);


	context.CopyBuffer(m_bufferManager->GetBuffer(1), m_bufferManager->GetBuffer(0));


	auto& shadowUavBuffer = m_bufferManager->GetBuffer(0);

	UINT clearValues[4] = { 0, 0, 0, 0 };



	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
	context.m_commandList->Get()->ClearUnorderedAccessViewUint(m_bufferManager->GetUAVHandle(), shadowUavBuffer.GetUAV(), shadowUavBuffer.Get(), clearValues, 0, nullptr);
	TechniquePass(context, DirectX::XMUINT3(1024, 1024, 1));


	PIXEndEvent(context.m_commandList->Get());

}

void CVGI::LightVoxel::TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name.c_str()].get());

	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& aabbBufferManager = m_data->GetBufferManager(BuildAABBsTechnique::Name);
	auto& rtBufferManger = m_data->GetBufferManager(ClusterVisibility::Name);

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	
	rtBufferManger.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	auto& shadowMapTableSrv = Renderer::GetShadowMapSrv(context);

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::RayTracinShadowCommonCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbShadowRaytrace).GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::ShadowCameraCBV, m_shadowCamera->GetShadowCB().GpuAddress());
	context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)RayTraceShadowRootSignature::AccelerationStructureSRV, m_data->GetTlas()->GetGpuVirtualAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::ShadowMapSRV, shadowMapTableSrv);
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::CompactTableSRV, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::AABBTableSRV, aabbBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::ASBufferMapSRV, rtBufferManger.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::RayTraceShadowTableUAV, m_bufferManager->GetUAVHandle());



	context.DispatchRays2D(1024, 1024);
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
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::RayTracinShadowCommonCBV].InitAsConstantBuffer(0);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::ShadowCameraCBV].InitAsConstantBuffer(1);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::AccelerationStructureSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::ShadowMapSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::CompactTableSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::AABBTableSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*rayTracingRootSignature)[(UINT)RayTraceShadowRootSignature::ASBufferMapSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 3);
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
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::ClosestHit, L"ShadowClosestHit");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Intersection, L"ShadowIntersection");
	rayTracingPso->SetAttributeAndPayloadSize(sizeof(UINT32), sizeof(UINT32));
	rayTracingPso->SetRecursionDepth(1);

	auto& hitGroup = rayTracingPso->CreateHitGroup(L"ShadowHitGroup");
	hitGroup.AddClosestHitShader(L"ShadowClosestHit");
	hitGroup.AddIntersectionShader(L"ShadowIntersection");
	hitGroup.SetHitGroupType(D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE);

	rayTracingPso->Finalize();
	rayTracingPso->Name = Name;

	return rayTracingPso;
}

const std::wstring CVGI::LightVoxel::Name = L"LightVoxel";