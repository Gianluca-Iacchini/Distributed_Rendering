#include "LightTransportTechnique.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/pch.h"



#include "DX12Lib/Scene/LightComponent.h"
#include "GaussianFilterTechnique.h"

#include "../Data/Shaders/Include/LightTransportDepth_CS.h"
#include "../Data/Shaders/Include/LightTransportIndirect_CS.h"

using namespace VOX;
using namespace DX12Lib;
using namespace Graphics;

VOX::LightTransportTechnique::LightTransportTechnique(std::shared_ptr<TechniqueData> data, bool computeIndirect) : m_computeIndirect(computeIndirect)
{
	m_bufferManager = std::make_shared<BufferManager>();
    data->SetBufferManager(Name, m_bufferManager);


    m_indirectBufferManager = std::make_shared<BufferManager>();
    data->SetBufferManager(IndirectName, m_indirectBufferManager);
    


	m_data = data;
}

void VOX::LightTransportTechnique::InitializeBuffers()
{
    this->CreateFrustumPlanes();

	UINT32 voxelBitCount = (m_data->GetVoxelCount() + 31) / 32;

	// Visible face counter
	// 0: Visible faces to be updated for the indirect light
	// 1: Visible faces to be updated for the gaussian filter
    m_bufferManager->AddByteAddressBuffer(2);
    // Indirect light visible faces indices
    m_bufferManager->AddStructuredBuffer((UINT32)ceilf(m_data->FaceCount / 2.0f), sizeof(UINT32));
    // Gaussian visible faces indices
    m_bufferManager->AddStructuredBuffer((UINT32)ceilf(m_data->FaceCount / 2.0f), sizeof(UINT32));
	// Updated indirect light voxels bitmap
	m_bufferManager->AddByteAddressBuffer(voxelBitCount);
	// Updated gaussian filter voxels bitmap
	m_bufferManager->AddByteAddressBuffer(voxelBitCount);
    // Gaussian Filter dispatch buffer
    m_bufferManager->AddStructuredBuffer(1, sizeof(D3D12_DISPATCH_ARGUMENTS));
    // Indirect Light dispatch buffer    
    m_bufferManager->AddStructuredBuffer(1, sizeof(D3D12_DISPATCH_ARGUMENTS));

    m_bufferManager->AllocateBuffers();

    // packing 16bit floats into 32bit floats
    // Voxel face radiance packed
    m_indirectBufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2));
    // Radiance for visible face idx

    //if (m_computeIndirect)
    {

        m_indirectBufferManager->AddStructuredBuffer((UINT32)ceilf(m_data->FaceCount / 2.0f), sizeof(DirectX::XMUINT2));

		m_faceIndicesReadBack.Create((UINT32)ceilf(m_data->FaceCount / 2.0f), sizeof(UINT32));
		m_faceRadianceReadback.Create((UINT32)ceilf(m_data->FaceCount / 2.0f), sizeof(DirectX::XMUINT2));
		m_visibleFacesCountReadback.Create(1, sizeof(UINT32));
    }

    m_indirectBufferManager->AllocateBuffers();
    
    this->CreateExecuteIndirectCommandBuffer();

}

void VOX::LightTransportTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), Name.c_str());

    CreateFrustumPlanes();

	m_cbFrustumCulling.AABBGroupCount = m_data->AABBGeometryGroupCount;
    m_cbFrustumCulling.CameraPosition = m_data->GetCamera()->Node->GetPosition();
    m_cbFrustumCulling.CurrentStep = 0;
	m_cbFrustumCulling.VoxelCount = m_data->GetVoxelCount();
	m_cbFrustumCulling.FaceCount = m_data->FaceCount;


    // Reset buffers
	TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->FaceCount / ( 2.0f * 128.0f)), 1, 1));

    m_cbFrustumCulling.CurrentStep = 1;

    TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->GetVoxelCount() / 128.0f), 1, 1));

    m_cbFrustumCulling.ResetRadianceBuffers = 0;

	PIXEndEvent(context.m_commandList->Get());
}

void VOX::LightTransportTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
    context.SetDescriptorHeap(Renderer::s_textureHeap.get());
    context.SetPipelineState(m_techniquePSO.get());

	auto& prefixSumBufferManager = m_data->GetBufferManager(PREFIX_SUM);


	prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0), true);

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::LightTransportCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFrustumCulling).GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::CameraCBV, m_data->GetDepthCameraResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportTechniqueRootParameters::OffsetCameraCBV, m_data->GetOffsetDepthCameraResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::PrefixSumBuffersSRV, prefixSumBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::DepthMapSRV, m_data->GetDepthCameraSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportTechniqueRootParameters::LightTransportBuffersUAV, m_bufferManager->GetUAVHandle());


    context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void VOX::LightTransportTechnique::TechniquePassIndirect(DX12Lib::ComputeContext& context)
{
    //if (!m_computeIndirect)
    //    return;

    context.SetDescriptorHeap(Renderer::s_textureHeap.get());
    context.SetPipelineState(m_indirectLightPso.get());

    auto& prefixSumBufferManager = m_data->GetBufferManager(PREFIX_SUM);
	auto& clusterVoxelBufferManager = m_data->GetBufferManager(CLUSTERIZE_SCENE);
    auto& aabbBufferManager = m_data->GetBufferManager(BUILD_AABB);
	auto& visibilityBufferManager = m_data->GetBufferManager(CLUSTER_VISIBILITY);
	auto& litVoxelsBufferManager = m_data->GetBufferManager(LIT_VOXELS);

    prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    clusterVoxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	visibilityBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	litVoxelsBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_indirectBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    for (UINT i = 0; i < m_bufferManager->GetBufferCount(); i++)
    {
        if (i == (UINT)LightTransportBufferType::IndirectLightDispatchBuffer)
        {
            m_bufferManager->TransitionBuffer(i, context, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        else
        {
            m_bufferManager->TransitionBuffer(i, context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }


    context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0));
    context.FlushResourceBarriers();

    context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportIndirectRootParameters::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
    context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LightTransportIndirectRootParameters::IndirectCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbLightIndirect).GpuAddress());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::PrefixSumBuffersSRV, prefixSumBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::ClusterVoxelBufferSRV, clusterVoxelBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::AABBBuffersSRV, aabbBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::ClusterVisibilitySRV, visibilityBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::LitVoxelsSRV, litVoxelsBufferManager.GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::LightTransportBuffersSRV, m_bufferManager->GetSRVHandle());
    context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LightTransportIndirectRootParameters::RadianceBufferUAV, m_indirectBufferManager->GetUAVHandle());

	auto& indirectLightCommandBuffer = m_bufferManager->GetBuffer((UINT)LightTransportBufferType::IndirectLightDispatchBuffer);
	context.m_commandList->Get()->ExecuteIndirect(m_commandSignature.Get(), 1, indirectLightCommandBuffer.Get(), 0, nullptr, 0);
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
void VOX::LightTransportTechnique::CreateFrustumPlanes()
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

void VOX::LightTransportTechnique::CreateExecuteIndirectCommandBuffer()
{
	D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
	D3D12_INDIRECT_ARGUMENT_DESC argumentDesc[1] = {};
    argumentDesc[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    
	commandSignatureDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
	commandSignatureDesc.NumArgumentDescs = 1;
	commandSignatureDesc.pArgumentDescs = argumentDesc;

	Graphics::s_device->Get()->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&m_commandSignature));
}

std::shared_ptr<DX12Lib::RootSignature>  VOX::LightTransportTechnique::BuildIndirectRootSignature()
{
    SamplerDesc defaultSamplerDesc;

    std::shared_ptr<DX12Lib::RootSignature> LightTransportRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)LightTransportIndirectRootParameters::Count, 1);
    LightTransportRootSignature->InitStaticSampler(0, defaultSamplerDesc);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::VoxelCommonsCBV].InitAsConstantBuffer(0);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::IndirectCBV].InitAsConstantBuffer(1);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::PrefixSumBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::ClusterVoxelBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::AABBBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::ClusterVisibilitySRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 3);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::LitVoxelsSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 4);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::LightTransportBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 6, D3D12_SHADER_VISIBILITY_ALL, 5);
    (*LightTransportRootSignature)[(UINT)LightTransportIndirectRootParameters::RadianceBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2);

    LightTransportRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return LightTransportRootSignature;
}

void VOX::LightTransportTechnique::BuildIndirectCommandPSO()
{
    std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildIndirectRootSignature();
    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pLightTransportIndirect_CS, ARRAYSIZE(g_pLightTransportIndirect_CS));

    m_indirectLightPso = std::make_unique<ComputePipelineState>();
    m_indirectLightPso->SetRootSignature(rootSig);
    m_indirectLightPso->SetComputeShader(shaderBlob);
    m_indirectLightPso->Finalize();
    m_indirectLightPso->Name = IndirectName;
}


void VOX::LightTransportTechnique::BuildPipelineState()
{
    std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pLightTransportDepth_CS, ARRAYSIZE(g_pLightTransportDepth_CS));


    std::unique_ptr<ComputePipelineState> lightTransportComputePso = std::make_unique<ComputePipelineState>();
    lightTransportComputePso->SetRootSignature(rootSig);
    lightTransportComputePso->SetComputeShader(shaderBlob);
    lightTransportComputePso->Finalize();
    lightTransportComputePso->Name = Name;

    BuildIndirectCommandPSO();

	m_techniquePSO = std::move(lightTransportComputePso);
}

void VOX::LightTransportTechnique::BuildPipelineState(CD3DX12_SHADER_BYTECODE& shaderByteCode)
{
    std::shared_ptr<DX12Lib::RootSignature> rootSig = BuildRootSignature();

    std::unique_ptr<ComputePipelineState> lightTransportComputePso = std::make_unique<ComputePipelineState>();
    lightTransportComputePso->SetRootSignature(rootSig);
    lightTransportComputePso->SetComputeShader(shaderByteCode);
    lightTransportComputePso->Finalize();
    lightTransportComputePso->Name = Name;

    BuildIndirectCommandPSO();

    m_techniquePSO = std::move(lightTransportComputePso);
}

void VOX::LightTransportTechnique::ClearRadianceBuffers(DX12Lib::ComputeContext& context, bool resetRadiance)
{
    PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"Clear buffers");

    CreateFrustumPlanes();

    m_cbFrustumCulling.AABBGroupCount = m_data->AABBGeometryGroupCount;
    m_cbFrustumCulling.CameraPosition = m_data->GetCamera()->Node->GetPosition();
    m_cbFrustumCulling.CurrentStep = 0;
    m_cbFrustumCulling.VoxelCount = m_data->GetVoxelCount();
    m_cbFrustumCulling.FaceCount = m_data->FaceCount;
	m_cbFrustumCulling.ResetRadianceBuffers = (UINT)resetRadiance;

    // Reset buffers
    TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->FaceCount / (2.0f * 128.0f)), 1, 1));

    m_cbFrustumCulling.ResetRadianceBuffers = 0;

    PIXEndEvent(context.m_commandList->Get());
}

void VOX::LightTransportTechnique::ComputeVisibleFaces(DX12Lib::ComputeContext& context)
{
    PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"Compute Visible Faces");

    CreateFrustumPlanes();

    m_cbFrustumCulling.AABBGroupCount = m_data->AABBGeometryGroupCount;
    m_cbFrustumCulling.CameraPosition = m_data->GetCamera()->Node->GetPosition();
    m_cbFrustumCulling.CurrentStep = 1;
    m_cbFrustumCulling.VoxelCount = m_data->GetVoxelCount();
    m_cbFrustumCulling.FaceCount = m_data->FaceCount;
    m_cbFrustumCulling.ResetRadianceBuffers = 0;


    TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->GetVoxelCount() / 128.0f), 1, 1));

    PIXEndEvent(context.m_commandList->Get());
}


void VOX::LightTransportTechnique::ResetRadianceBuffers(bool reset)
{
	m_cbFrustumCulling.ResetRadianceBuffers = (UINT)reset;
}

void VOX::LightTransportTechnique::LaunchIndirectLightBlock(DX12Lib::ComputeContext& context, UINT blockCount)
{
    auto* lightComponent = m_data->GetLightComponent();

    m_cbLightIndirect.LightDirection = lightComponent->Node->GetForward();
    m_cbLightIndirect.LightPosition = lightComponent->Node->GetPosition();
    m_cbLightIndirect.LightIntensity = 15.0f;
    m_cbLightIndirect.EyePosition = m_data->GetCamera()->Node->GetPosition();
	m_cbLightIndirect.DispatchNumber = blockCount;

    TechniquePassIndirect(context);
}

void VOX::LightTransportTechnique::TransferRadianceData(DX12Lib::ComputeContext& context)
{
    //if (!m_computeIndirect)
    //    return;

    context.CopyBuffer(m_faceIndicesReadBack, m_bufferManager->GetBuffer((UINT)LightTransportBufferType::IndirectLightVisibleFacesIndices));
	context.CopyBuffer(m_faceRadianceReadback, m_indirectBufferManager->GetBuffer(1));
	context.CopyBufferRegion(m_visibleFacesCountReadback, 0, m_bufferManager->GetBuffer(0), 0, sizeof(UINT32));
}

std::uint8_t* VOX::LightTransportTechnique::GetVisibleFacesIndices(UINT32 visFaceCount)
{
    void* data = m_faceIndicesReadBack.ReadBack(visFaceCount * sizeof(DirectX::XMUINT2));

    return reinterpret_cast<std::uint8_t*>(data);
}

std::uint8_t* VOX::LightTransportTechnique::GetVisibleFacesRadiance(UINT32 visFaceCount)
{
    void* data = m_faceRadianceReadback.ReadBack(visFaceCount * sizeof(DirectX::XMUINT2));

    return reinterpret_cast<std::uint8_t*>(data);
}

UINT32 VOX::LightTransportTechnique::GetVisibleFacesCount()
{
    void* data = m_visibleFacesCountReadback.ReadBack(sizeof(UINT32));

    UINT32 faceCount = 0;

	memcpy(&faceCount, data, sizeof(UINT32));

	return faceCount;
}

std::shared_ptr<DX12Lib::RootSignature> VOX::LightTransportTechnique::BuildRootSignature()
{
    SamplerDesc defaultSamplerDesc;

    SamplerDesc ShadowSamplerDesc;
    ShadowSamplerDesc.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    ShadowSamplerDesc.SetTextureAddressMode(D3D12_TEXTURE_ADDRESS_MODE_BORDER);
    ShadowSamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    ShadowSamplerDesc.MipLODBias = 0.0f;
    ShadowSamplerDesc.MaxAnisotropy = 1;
    ShadowSamplerDesc.SetBorderColor(Color::Black());

    std::shared_ptr<DX12Lib::RootSignature> LightTransportRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)LightTransportTechniqueRootParameters::Count, 2);
    LightTransportRootSignature->InitStaticSampler(0, defaultSamplerDesc);
    LightTransportRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::VoxelCommonsCBV].InitAsConstantBuffer(0);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::LightTransportCBV].InitAsConstantBuffer(1);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::CameraCBV].InitAsConstantBuffer(2);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::OffsetCameraCBV].InitAsConstantBuffer(3);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::PrefixSumBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::DepthMapSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*LightTransportRootSignature)[(UINT)LightTransportTechniqueRootParameters::LightTransportBuffersUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, (UINT)LightTransportBufferType::Count, D3D12_SHADER_VISIBILITY_ALL, 0);

    LightTransportRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return LightTransportRootSignature;
}


const std::wstring VOX::LightTransportTechnique::Name = L"LightTransportTechnique";
const std::wstring VOX::LightTransportTechnique::IndirectName = L"LightTransportIndirectTechnique";
