#include "GaussianFilterTechnique.h"
#include "DX12Lib/pch.h"
#include "LightTransportTechnique.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "../Data/Shaders/Include/GaussianFilter_CS.h"

using namespace VOX;
using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

#define SIDE 2
#define KERNEL_SIZE (SIDE * 2 + 1)

VOX::GaussianFilterTechnique::GaussianFilterTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	m_readBufferManager = std::make_shared<BufferManager>();

	data->SetBufferManager(Name, m_bufferManager);
	data->SetBufferManager(ReadName, m_readBufferManager);

	m_data = data;
}

void VOX::GaussianFilterTechnique::InitializeBuffers()
{
	// First filter pass
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(UINT32));

	// Gaussian precomputed values
	m_bufferManager->AddStructuredBuffer(KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE, sizeof(float));

	m_readBufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2), L"Gaussian Read Buffer");

	m_bufferManager->AllocateBuffers();
	m_readBufferManager->AllocateBuffers();

	m_radianceReadBack.Create(m_data->FaceCount, sizeof(DirectX::XMUINT2));

	m_radianceData.resize(m_data->FaceCount);

	m_cbGaussianFilter.UsePreComputedGaussian = 1;

	m_cbGaussianFilter.KernelSize = 5;
	m_cbGaussianFilter.Sigma = 2.0f;
	m_cbGaussianFilter.PassCount = 2;
}

void VOX::GaussianFilterTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR_DEFAULT, Name.c_str());

	m_cbGaussianFilter.FaceCount = m_data->FaceCount;
	m_cbGaussianFilter.VoxelCount = m_data->GetVoxelCount();
	m_cbGaussianFilter.EyePosition = m_data->GetCamera()->Node->GetPosition();

	m_cbGaussianFilter.CurrentPhase = 1;
	// Called with execute indirect, group size is determined by previous pass.
	TechniquePass(context, DirectX::XMUINT3(1, 1, 1));


	PIXEndEvent(context.m_commandList->Get());
}

void VOX::GaussianFilterTechnique::InitializeGaussianConstants(DX12Lib::ComputeContext& context)
{
	m_cbGaussianFilter.CurrentPhase = 0;
	TechniquePass(context, DirectX::XMUINT3(1, 1, 1));
}

void VOX::GaussianFilterTechnique::BuildPipelineState()
{
    std::shared_ptr<DX12Lib::RootSignature> gaussianFilterRootSig = BuildRootSignature();

    auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pGaussianFilter_CS, ARRAYSIZE(g_pGaussianFilter_CS));

    std::unique_ptr<DX12Lib::ComputePipelineState> gaussianFilterPso = std::make_unique<DX12Lib::ComputePipelineState>();
    gaussianFilterPso->SetRootSignature(gaussianFilterRootSig);
    gaussianFilterPso->SetComputeShader(shaderBlob);
    gaussianFilterPso->Finalize();
    gaussianFilterPso->Name = Name;

	m_techniquePSO = std::move(gaussianFilterPso);
}



void VOX::GaussianFilterTechnique::SetGaussianBlock(UINT32 block)
{
	m_cbGaussianFilter.BlockNum = block;
}

void VOX::GaussianFilterTechnique::SetGaussianKernelSize(UINT size)
{
	m_cbGaussianFilter.KernelSize = size;
	m_gaussianOptionModified = true;
}

void VOX::GaussianFilterTechnique::SetUsePrecomputedGaussian(bool usePrecomp)
{
	m_cbGaussianFilter.UsePreComputedGaussian = usePrecomp ? 1 : 0;
	m_gaussianOptionModified = true;
}

void VOX::GaussianFilterTechnique::SetGaussianPassCount(UINT passCount)
{
	m_cbGaussianFilter.PassCount = passCount;
	m_gaussianOptionModified = true;
}

void VOX::GaussianFilterTechnique::SetGaussianSigma(float sigma)
{
	m_cbGaussianFilter.Sigma = sigma;
	m_gaussianOptionModified = true;
}

void VOX::GaussianFilterTechnique::PerformTechnique2(DX12Lib::ComputeContext& context)
{
	m_cbGaussianFilter.CurrentPhase = 2;
	// Called with execute indirect, group size is determined by previous pass.
	TechniquePass(context, DirectX::XMUINT3(1, 1, 1));
}

void VOX::GaussianFilterTechnique::TransferRadianceData(DX12Lib::ComputeContext& context)
{
	//m_isTransferingRadiance = true;

	//context.CopyBuffer(m_radianceReadBack, m_writeBufferManager->GetBuffer(0));

	//m_radLength = m_data->FaceCount;
}

std::vector<DirectX::XMUINT2>& VOX::GaussianFilterTechnique::GetRadianceData()
{
	void* data = m_radianceReadBack.ReadBack(m_data->FaceCount * sizeof(DirectX::XMUINT2));
	
	memcpy(m_radianceData.data(), data, m_data->FaceCount * sizeof(DirectX::XMUINT2));

	return m_radianceData;
}

std::uint8_t* VOX::GaussianFilterTechnique::GetRadianceDataPtr()
{
	void* data = m_radianceReadBack.ReadBack(m_data->FaceCount * sizeof(DirectX::XMUINT2));

	return reinterpret_cast<std::uint8_t*>(data);
}

UINT64 VOX::GaussianFilterTechnique::GetMemoryUsage()
{
	return m_bufferManager->GetTotalMemorySize() + m_readBufferManager->GetTotalMemorySize();
}

void VOX::GaussianFilterTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetPipelineState(m_techniquePSO.get());
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());

	auto& voxelBufferManager = m_data->GetBufferManager(VOXELIZE_SCENE);
	auto& prefixSumBufferManager = m_data->GetBufferManager(PREFIX_SUM);
	auto& visibleFacesBufferManager = m_data->GetBufferManager(LightTransportTechnique::Name);
	auto& faceRadianceBufferManager = m_data->GetBufferManager(LightTransportTechnique::IndirectName);

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	prefixSumBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	faceRadianceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	for (UINT i = 0; i < visibleFacesBufferManager.GetBufferCount(); i++)
	{
		if (i != (UINT)LightTransportTechnique::LightTransportBufferType::IndirectLightDispatchBuffer &&
			i != (UINT)LightTransportTechnique::LightTransportBufferType::GaussianDispatchBuffer)
		{
			visibleFacesBufferManager.TransitionBuffer(i, context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		}
	}

	auto& indirectCommandBuffer = visibleFacesBufferManager.GetBuffer((UINT)LightTransportTechnique::LightTransportBufferType::GaussianDispatchBuffer);
	context.TransitionResource(indirectCommandBuffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
	
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_readBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	context.AddUAVIfNoBarriers();
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)GaussianFilterRootParameters::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)GaussianFilterRootParameters::GaussianFilterCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbGaussianFilter).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::VoxelDataSRV, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::PrefixSumSRV, prefixSumBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::VoxelVisibleFaceSRV, visibleFacesBufferManager.GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::RadianceBufferSRV, faceRadianceBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::GaussianBufferUAV, m_bufferManager->GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)GaussianFilterRootParameters::WriteBufferUAV, m_readBufferManager->GetUAVHandle());
	
	if (m_cbGaussianFilter.CurrentPhase == 0)
		context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
	else
		context.m_commandList->Get()->ExecuteIndirect(m_commandSignature.Get(), 1, indirectCommandBuffer.Get(), 0, nullptr, 0);

	m_gaussianOptionModified = false;
}

std::shared_ptr<DX12Lib::RootSignature> VOX::GaussianFilterTechnique::BuildRootSignature()
{

    std::shared_ptr<DX12Lib::RootSignature> gaussianRootSig = std::make_shared<DX12Lib::RootSignature>((UINT)GaussianFilterRootParameters::Count, 0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelCommonCBV].InitAsConstantBuffer(0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::GaussianFilterCBV].InitAsConstantBuffer(1);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelDataSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::PrefixSumSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::VoxelVisibleFaceSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 2);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::RadianceBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 4);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::GaussianBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);
    (*gaussianRootSig)[(UINT)GaussianFilterRootParameters::WriteBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 1);

    gaussianRootSig->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

    return gaussianRootSig;
}

const std::wstring VOX::GaussianFilterTechnique::Name = L"GaussianFilterTechnique";
const std::wstring VOX::GaussianFilterTechnique::ReadName = L"GaussianFilterRead";
