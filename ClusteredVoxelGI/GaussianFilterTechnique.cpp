#include "GaussianFilterTechnique.h"
#include "DX12Lib/pch.h"
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "FaceCountTechnique.h"
#include "FacePenaltyTechnique.h"
#include "LightTransportTechnique.h"
#include "GaussianFilter_CS.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

CVGI::GaussianFilterTechnique::GaussianFilterTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->AddBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::GaussianFilterTechnique::InitializeBuffers()
{
	// Empty buffer for uav barriers
	m_bufferManager->AddByteAddressBuffer();
}

void CVGI::GaussianFilterTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR_DEFAULT, Name.c_str());

	m_cbGaussianFilter.CurrentPhase = 0;
    m_cbGaussianFilter.KernelSize = 5;
	m_cbGaussianFilter.FaceCount = m_data->FaceCount;
	m_cbGaussianFilter.VoxelCount = m_data->VoxelCount;

	// Called with execute indirect, group size is determined by previous pass.
	TechniquePass(context, DirectX::XMUINT3(1, 1, 1));

	m_cbGaussianFilter.CurrentPhase = 1;
	// Called with execute indirect, group size is determined by previous pass.
	TechniquePass(context, DirectX::XMUINT3(1, 1, 1));

	PIXEndEvent(context.m_commandList->Get());
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::GaussianFilterTechnique::BuildPipelineState()
{
    std::shared_ptr<DX12Lib::RootSignature> gaussianFilterRootSig = BuildRootSignature();

    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pGaussianFilter_CS, ARRAYSIZE(g_pGaussianFilter_CS));

    std::shared_ptr<DX12Lib::ComputePipelineState> gaussianFilterPso = std::make_shared<DX12Lib::ComputePipelineState>();
    gaussianFilterPso->SetRootSignature(gaussianFilterRootSig);
    gaussianFilterPso->SetComputeShader(shaderBlob);
    gaussianFilterPso->Finalize();
    gaussianFilterPso->Name = Name;

    return gaussianFilterPso;
}

void CVGI::GaussianFilterTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetPipelineState(Renderer::s_PSOs[Name].get());
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());

	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& prefixSumBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& faceDataBufferManager = m_data->GetBufferManager(FaceCountTechnique::Name);
	auto& facePenaltyBufferManager = m_data->GetBufferManager(FacePenaltyTechnique::Name);
	auto& visibleFacesBufferManager = m_data->GetBufferManager(LightTransportTechnique::Name);
	auto& voxelRadianceBufferManager = m_data->GetBufferManager(LightTransportTechnique::IndirectName);

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	faceDataBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	facePenaltyBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	voxelRadianceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	
	visibleFacesBufferManager.TransitionBuffer((UINT)LightTransportTechnique::LightTransportBufferType::VisibleFaceCounter, context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	visibleFacesBufferManager.TransitionBuffer((UINT)LightTransportTechnique::LightTransportBufferType::VisibleFaceIndices, context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	visibleFacesBufferManager.TransitionBuffer((UINT)LightTransportTechnique::LightTransportBufferType::IndirectDispatchBuffer, context, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
	
	context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0));
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)GaussianFilterRootParameters::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)GaussianFilterRootParameters::GaussianFilterCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbGaussianFilter).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::VoxelDataSRV, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::PrefixSumSRV, prefixSumBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::FaceDataSRV, faceDataBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::FacePenaltySRV, facePenaltyBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::VoxelVisibleFaceSRV, visibleFacesBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::VoxelRadianceUAV, voxelRadianceBufferManager.GetUAVHandle());
	
	auto& indirectDispatchBuffer = visibleFacesBufferManager.GetBuffer((UINT)LightTransportTechnique::LightTransportBufferType::IndirectDispatchBuffer);
	context.m_commandList->Get()->ExecuteIndirect(m_commandSignature.Get(), 1, indirectDispatchBuffer.Get(), sizeof(D3D12_DISPATCH_ARGUMENTS), nullptr, 0);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::GaussianFilterTechnique::BuildRootSignature()
{

    std::shared_ptr<DX12Lib::RootSignature> gaussianRootSig = std::make_shared<DX12Lib::RootSignature>((UINT)GaussianFilterRootParameters::Count, 0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelCommonCBV].InitAsConstantBuffer(0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::GaussianFilterCBV].InitAsConstantBuffer(1);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelDataSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::PrefixSumSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::FaceDataSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 2);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::FacePenaltySRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 3);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelVisibleFaceSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 4);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelRadianceUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

    gaussianRootSig->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return gaussianRootSig;
}

const std::wstring CVGI::GaussianFilterTechnique::Name = L"GaussianFilterTechnique";