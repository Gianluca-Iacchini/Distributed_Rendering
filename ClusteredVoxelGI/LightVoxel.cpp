#include "LightVoxel.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"


using namespace CVGI;
using namespace DirectX;
using namespace Graphics;

void CVGI::LightVoxel::InitializeBuffers(UINT voxelCount)
{
	UINT32 voxelBitCount = (voxelCount + 31) / 32;
	m_bufferManager.AddByteAddressBuffer(voxelBitCount);
	m_bufferManager.AddByteAddressBuffer(voxelBitCount);

	m_bufferManager.AllocateBuffers();

	m_voxelCount = voxelCount;


}

void CVGI::LightVoxel::StartLightVoxel(DX12Lib::ShadowCamera& camera, BufferManager& compactBufferManager, BufferManager& aabbBufferManager, BufferManager& rtBufferManager, TopLevelAccelerationStructure& tlas)
{
	RayTracingContext& context = RayTracingContext::Begin();
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"RayTracing");


	m_cbShadowRaytrace.GridDimension = m_voxelTexDimensions;
	m_cbShadowRaytrace.ShadowTexDimensions = DirectX::XMUINT2(2048, 2048);

	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);


	context.CopyBuffer(m_bufferManager.GetBuffer(1), m_bufferManager.GetBuffer(0));


	auto& shadowUavBuffer = m_bufferManager.GetBuffer(0);

	UINT clearValues[4] = { 0, 0, 0, 0 };



	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
	context.m_commandList->Get()->ClearUnorderedAccessViewUint(m_bufferManager.GetUAVHandle(), shadowUavBuffer.GetUAV(), shadowUavBuffer.Get(), clearValues, 0, nullptr);
	LightVoxelPass(context, camera, compactBufferManager, aabbBufferManager, rtBufferManager, tlas);
	

	PIXEndEvent(context.m_commandList->Get());
	context.Finish();
}

void CVGI::LightVoxel::LightVoxelPass(RayTracingContext& context, DX12Lib::ShadowCamera& camera, BufferManager& compactBufferManager, BufferManager& aabbBufferManager,  BufferManager& rtBufferManager, TopLevelAccelerationStructure& tlas)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[L"ShadowRaytracingPso"].get());

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	rtBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	auto& shadowMapTableSrv = Renderer::GetShadowMapSrv(context);



	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::RayTracinShadowCommonCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbShadowRaytrace).GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceShadowRootSignature::ShadowCameraCBV, camera.GetShadowCB().GpuAddress());
	context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)RayTraceShadowRootSignature::AccelerationStructureSRV, tlas.GetGpuVirtualAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::ShadowMapSRV, shadowMapTableSrv);
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::CompactTableSRV, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::AABBTableSRV, aabbBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::ASBufferMapSRV, rtBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceShadowRootSignature::RayTraceShadowTableUAV, m_bufferManager.GetUAVHandle());
	


	context.DispatchRays2D(1024, 1024);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::LightVoxel::BuildLightVoxelRootSignature()
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

std::shared_ptr<RaytracingStateObject> CVGI::LightVoxel::BuildLightVoxelPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
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
	rayTracingPso->Name = L"ShadowRaytracingPso";

	return rayTracingPso;
}
