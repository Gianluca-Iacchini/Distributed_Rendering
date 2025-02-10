#include "RadianceFromNetworkTechnique.h"
#include "DX12Lib/pch.h"
#include "Data/Shaders/Include/RadianceFromNetwork_CS.h"
#include "LightTransportTechnique.h"

using namespace LI;
using namespace DX12Lib;
using namespace VOX;

enum class NetworkRadianceRootSignature
{
	VoxelCommonCBV = 0,
	NetworkRadianceCBV = 1,
	NetworkRadianceBuffersSRV = 2,
	VisibleVoxelsBitmapsUAV = 3,
	FinalRadianceBufferUAV = 4,
	Count
};

LI::RadianceFromNetworkTechnique::RadianceFromNetworkTechnique(std::shared_ptr<VOX::TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();

	data->SetBufferManager(L"RadianceFromNetworkTechnique", this->m_bufferManager);

	m_data = data;
}

void LI::RadianceFromNetworkTechnique::InitializeBuffers()
{
	// Radiance for face buffer
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2));

	m_bufferManager->AllocateBuffers();
}

void LI::RadianceFromNetworkTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(111, 220, 220), L"RadianceFromNetworkTechnique");

	m_cbRadianceFromNetwork.FaceCount = m_data->FaceCount;
	
	UINT32 nGroups = (UINT)ceilf(m_cbRadianceFromNetwork.ReceivedFaceCount / 128.0f);

	TechniquePass(context, DirectX::XMUINT3(nGroups, 1, 1));

	m_cbRadianceFromNetwork.ShouldReset = 0;

	PIXEndEvent(context.m_commandList->Get());
}

void LI::RadianceFromNetworkTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetPipelineState(m_techniquePSO.get());
	context.SetDescriptorHeap(Graphics::Renderer::s_textureHeap.get());


	auto& voxelBitmapBufferManager = m_data->GetBufferManager(LightTransportTechnique::Name);
	auto& lightTransportBufferManager = m_data->GetBufferManager(LightTransportTechnique::IndirectName);

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	voxelBitmapBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	lightTransportBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	context.AddUAVIfNoBarriers();
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)NetworkRadianceRootSignature::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)NetworkRadianceRootSignature::NetworkRadianceCBV, Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_cbRadianceFromNetwork).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)NetworkRadianceRootSignature::NetworkRadianceBuffersSRV, m_bufferManager->GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)NetworkRadianceRootSignature::VisibleVoxelsBitmapsUAV, voxelBitmapBufferManager.GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)NetworkRadianceRootSignature::FinalRadianceBufferUAV, lightTransportBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void LI::RadianceFromNetworkTechnique::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> radianceFromNetRootSig = BuildRootSignature();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pRadianceFromNetwork_CS, ARRAYSIZE(g_pRadianceFromNetwork_CS));

	std::unique_ptr<DX12Lib::ComputePipelineState> gaussianFilterPso = std::make_unique<DX12Lib::ComputePipelineState>();
	gaussianFilterPso->SetRootSignature(radianceFromNetRootSig);
	gaussianFilterPso->SetComputeShader(shaderBlob);
	gaussianFilterPso->Finalize();
	gaussianFilterPso->Name = L"RadianceFromNetworkTechnique";

	m_techniquePSO = std::move(gaussianFilterPso);
}

UINT64 LI::RadianceFromNetworkTechnique::ProcessNetworkData(DX12Lib::ComputeContext& context, DX12Lib::UploadBuffer* buffer, UINT faceCount, UINT shouldReset)
{
	m_cbRadianceFromNetwork.ShouldReset = shouldReset;
	m_cbRadianceFromNetwork.ReceivedFaceCount = faceCount;

	auto& faceIndexBuffer = m_bufferManager->GetBuffer(0);

	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(faceIndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST);
	context.FlushResourceBarriers();

	UINT32 faceIdxByteSize = faceCount * sizeof(DirectX::XMUINT2);

	context.m_commandList->Get()->CopyBufferRegion(faceIndexBuffer.Get(), 0, buffer->Get(), 0, faceIdxByteSize);

	UINT64 copyFenceVal = context.Flush();

	PerformTechnique(context);

	return copyFenceVal;
}


std::shared_ptr<DX12Lib::RootSignature> LI::RadianceFromNetworkTechnique::BuildRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> radianceFromNetRootSig = std::make_shared<DX12Lib::RootSignature>((UINT)NetworkRadianceRootSignature::Count, 0);

	(*radianceFromNetRootSig)[(UINT)NetworkRadianceRootSignature::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*radianceFromNetRootSig)[(UINT)NetworkRadianceRootSignature::NetworkRadianceCBV].InitAsConstantBuffer(1);
	(*radianceFromNetRootSig)[(UINT)NetworkRadianceRootSignature::NetworkRadianceBuffersSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
	(*radianceFromNetRootSig)[(UINT)NetworkRadianceRootSignature::VisibleVoxelsBitmapsUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5);
	(*radianceFromNetRootSig)[(UINT)NetworkRadianceRootSignature::FinalRadianceBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 1);

	radianceFromNetRootSig->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return radianceFromNetRootSig;
}


