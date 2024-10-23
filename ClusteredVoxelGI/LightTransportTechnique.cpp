#include "LightTransportTechnique.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/pch.h"
#include "LightTransport_CS.h"
#include "LightTransportIndirect_CS.h"
#include "VoxelizeScene.h"
#include "ClusterVoxels.h"
#include "PrefixSumVoxels.h"
#include "FaceCountTechnique.h"
#include "BuildAABBsTechnique.h"
#include "ClusterVisibility.h"
#include "LightVoxel.h"
#include "RaytracingStateObject.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "FacePenaltyTechnique.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

CVGI::LightTransportTechnique::LightTransportTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	m_indirectBufferManager = std::make_shared<BufferManager>();
	data->AddBufferManager(Name, m_bufferManager);
	data->AddBufferManager(IndirectName, m_indirectBufferManager);
	m_data = data;
}

void CVGI::LightTransportTechnique::InitializeBuffers()
{
    this->CreateFrustumPlanes();

	UINT32 voxelBitCount = (m_data->VoxelCount + 31) / 32;

    m_bufferManager->AddByteAddressBuffer();
    m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(UINT32));
    m_bufferManager->AddStructuredBuffer(2, sizeof(D3D12_DISPATCH_ARGUMENTS));

	// packing 16bit floats into 32bit floats
    // Voxel face radiance packed
    m_indirectBufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2));
	// Voxel face FILTERED radiance packed (used by the gaussian filter)
    m_indirectBufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2));

    this->CreateExecuteIndirectCommandBuffer();

    m_bufferManager->AllocateBuffers();
	m_indirectBufferManager->AllocateBuffers();
}

void CVGI::LightTransportTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), Name.c_str());

    CreateFrustumPlanes();

	m_cbFrustumCulling.AABBGroupCount = m_data->AABBGeometryGroupCount;
    m_cbFrustumCulling.CameraPosition = m_data->GetCamera()->Node->GetPosition();
    m_cbFrustumCulling.CurrentStep = 0;
	m_cbFrustumCulling.VoxelCount = m_data->VoxelCount;
	m_cbFrustumCulling.FaceCount = m_data->FaceCount;


    // Reset buffers
	TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->FaceCount / 128.0f), 1, 1));

    m_cbFrustumCulling.CurrentStep = 1;

    TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->AABBGeometryGroupCount), 1, 1));

	auto* lightComponent = m_data->GetLightComponent();

    m_cbLightIndirect.LightDirection = lightComponent->Node->GetForward();
	m_cbLightIndirect.LightPosition = lightComponent->Node->GetPosition();
    m_cbLightIndirect.LightIntensity = 15.0f;


    TechniquePassIndirect(context);


	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::LightTransportTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
    context.SetDescriptorHeap(Renderer::s_textureHeap.get());
    context.SetPipelineState(Renderer::s_PSOs[Name].get());

	auto& prefixSumBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& faceBufferManager = m_data->GetBufferManager(FaceCountTechnique::Name);
	auto& aabbBufferManager = m_data->GetBufferManager(BuildAABBsTechnique::Name);

	prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_indirectBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(1), true);

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::LightTransportCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFrustumCulling).GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::CameraCBV, m_data->GetCamera()->GetCameraBuffer().GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::PrefixSumBuffersSRV, prefixSumBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::FaceBufferSRV, faceBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::AABBBuffersSRV, aabbBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)LightTransportTechniqueRootParameters::AccelerationStructureSRV, m_data->GetTlas()->GetGpuVirtualAddress());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::LightTransportBuffersUAV, m_bufferManager->GetUAVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::IndirectBuffersUAV, m_indirectBufferManager->GetUAVHandle());

    context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void CVGI::LightTransportTechnique::TechniquePassIndirect(DX12Lib::ComputeContext& context)
{
    context.SetDescriptorHeap(Renderer::s_textureHeap.get());
    context.SetPipelineState(Renderer::s_PSOs[IndirectName].get());

    auto& prefixSumBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterVoxelBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);
	auto& faceBufferManager = m_data->GetBufferManager(FaceCountTechnique::Name);
    auto& aabbBufferManager = m_data->GetBufferManager(BuildAABBsTechnique::Name);
	auto& visibilityBufferManager = m_data->GetBufferManager(ClusterVisibility::Name);
	auto& litVoxelsBufferManager = m_data->GetBufferManager(LightVoxel::Name);
	auto& facePenaltyBufferManager = m_data->GetBufferManager(FacePenaltyTechnique::Name);

    prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    clusterVoxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	visibilityBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	litVoxelsBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	facePenaltyBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_indirectBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_bufferManager->TransitionBuffer((UINT)LightTransportBufferType::VisibleFaceCounter,
        context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_bufferManager->TransitionBuffer((UINT)LightTransportBufferType::VisibleFaceIndices,
        context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_bufferManager->TransitionBuffer((UINT)LightTransportBufferType::IndirectDispatchBuffer,
        context, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

    context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0));
    context.FlushResourceBarriers();

    context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportIndirectRootParameters::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
    context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportIndirectRootParameters::IndirectCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbLightIndirect).GpuAddress());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::PrefixSumBuffersSRV, prefixSumBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::ClusterVoxelBufferSRV, clusterVoxelBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::FaceBufferSRV, faceBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::AABBBuffersSRV, aabbBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::ClusterVisibilitySRV, visibilityBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::FacePenaltyBufferSRV, facePenaltyBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::LitVoxelsSRV, litVoxelsBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::LightTransportBuffersSRV, m_bufferManager->GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::IndirectBuffersUAV, m_indirectBufferManager->GetUAVHandle());

	auto& indirectCommandBuffer = m_bufferManager->GetBuffer((UINT)LightTransportBufferType::IndirectDispatchBuffer);
	context.m_commandList->Get()->ExecuteIndirect(m_commandSignature.Get(), 1, indirectCommandBuffer.Get(), 0, nullptr, 0);
}

LightTransportTechnique::Plane& LightTransportTechnique::TransformAndNormalize(LightTransportTechnique::Plane& plane)
{
	// To transform the plane to voxel space we need the inverse transpose of the world to voxel matrix.
	// However we already have the inverse of the world to voxel transformation, which is the voxel to world matrix.
	// Matrices are transposed before being stored in the constant buffer due to hlsl being column major, so we don't need
    // to do anything.
	DirectX::XMMATRIX voxToWorld = DirectX::XMLoadFloat4x4(&m_data->GetVoxelToWorldMatrix());

	plane = DirectX::XMVector4Transform(plane, voxToWorld);

    DirectX::XMVECTOR normal = DirectX::XMVectorSetW(plane, 0.0f);
	float length = DirectX::XMVectorGetX(DirectX::XMVector3Length(normal));

	plane = plane / length;

	return plane;
}
void CVGI::LightTransportTechnique::CreateFrustumPlanes()
{
    auto view = m_data->GetCamera()->GetView();
	auto proj = m_data->GetCamera()->GetProjection();

	DirectX::XMMATRIX viewProjMatrix = view * proj;
	viewProjMatrix = DirectX::XMMatrixTranspose(viewProjMatrix);

    std::array<Plane, 6> frustumPlanes;

    frustumPlanes[0] = viewProjMatrix.r[3] + viewProjMatrix.r[0]; // Left plane
    frustumPlanes[1] = viewProjMatrix.r[3] - viewProjMatrix.r[0]; // Right plane
    frustumPlanes[2] = viewProjMatrix.r[3] + viewProjMatrix.r[1]; // Bottom plane
    frustumPlanes[3] = viewProjMatrix.r[3] - viewProjMatrix.r[1]; // Top plane
    frustumPlanes[4] = viewProjMatrix.r[3] + viewProjMatrix.r[2]; // Near plane
    frustumPlanes[5] = viewProjMatrix.r[3] - viewProjMatrix.r[2]; // Far plane


    // Normalize all planes
    for (int i = 0; i < 6; i++) 
    {
		auto& plane = frustumPlanes[i];
		DirectX::XMStoreFloat4(&m_cbFrustumCulling.FrustumPlanes[i], TransformAndNormalize(plane));
    }
}

void CVGI::LightTransportTechnique::CreateExecuteIndirectCommandBuffer()
{
	D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
	D3D12_INDIRECT_ARGUMENT_DESC argumentDesc[1] = {};
    argumentDesc[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    
	commandSignatureDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
	commandSignatureDesc.NumArgumentDescs = 1;
	commandSignatureDesc.pArgumentDescs = argumentDesc;

	Graphics::s_device->Get()->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&m_commandSignature));
}

std::shared_ptr<DX12Lib::RootSignature>  CVGI::LightTransportTechnique::BuildIndirectRootSignature()
{
    SamplerDesc defaultSamplerDesc;

    std::shared_ptr<DX12Lib::RootSignature> LightTransportRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)LightTransportIndirectRootParameters::Count, 1);
    LightTransportRootSignature->InitStaticSampler(0, defaultSamplerDesc);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::VoxelCommonsCBV].InitAsConstantBuffer(0);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::IndirectCBV].InitAsConstantBuffer(1);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::PrefixSumBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::ClusterVoxelBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::FaceBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 2);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::AABBBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 3);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::ClusterVisibilitySRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 4);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::FacePenaltyBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 5);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::LitVoxelsSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 6);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::LightTransportBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 7);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::IndirectBuffersUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

    LightTransportRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return LightTransportRootSignature;
}

void CVGI::LightTransportTechnique::BuildIndirectCommandPSO()
{
    std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildIndirectRootSignature();
    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pLightTransportIndirect_CS, ARRAYSIZE(g_pLightTransportIndirect_CS));

    std::shared_ptr<ComputePipelineState> lightTransportComputePso = std::make_shared<ComputePipelineState>();
    lightTransportComputePso->SetRootSignature(rootSig);
    lightTransportComputePso->SetComputeShader(shaderBlob);
    lightTransportComputePso->Finalize();
    lightTransportComputePso->Name = IndirectName;

	Renderer::s_PSOs[IndirectName] = lightTransportComputePso;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::LightTransportTechnique::BuildPipelineState()
{
    std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pLightTransport_CS, ARRAYSIZE(g_pLightTransport_CS));


    std::shared_ptr<ComputePipelineState> lightTransportComputePso = std::make_shared<ComputePipelineState>();
    lightTransportComputePso->SetRootSignature(rootSig);
    lightTransportComputePso->SetComputeShader(shaderBlob);
    lightTransportComputePso->Finalize();
    lightTransportComputePso->Name = Name;

    BuildIndirectCommandPSO();

    return lightTransportComputePso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::LightTransportTechnique::BuildRootSignature()
{
    SamplerDesc defaultSamplerDesc;

    std::shared_ptr<DX12Lib::RootSignature> LightTransportRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)LightTransportTechniqueRootParameters::Count, 1);
    LightTransportRootSignature->InitStaticSampler(0, defaultSamplerDesc);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::VoxelCommonsCBV].InitAsConstantBuffer(0);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::LightTransportCBV].InitAsConstantBuffer(1);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::CameraCBV].InitAsConstantBuffer(2);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::PrefixSumBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::FaceBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::AABBBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::AccelerationStructureSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 3);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::LightTransportBuffersUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::IndirectBuffersUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 1);

    LightTransportRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return LightTransportRootSignature;
}


const std::wstring CVGI::LightTransportTechnique::Name = L"LightTransportTechnique";
const std::wstring CVGI::LightTransportTechnique::IndirectName = L"LightTransportIndirectTechnique";
