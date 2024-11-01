#include "FaceCountTechnique.h"
#include "DX12Lib/pch.h"
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::FaceCountTechnique::InitializeBuffers()
{
	// Faces buffer (u0)
	m_bufferManager->AddStructuredBuffer(1, sizeof(DirectX::XMUINT2));
	// Face start count buffer (u1)
	m_bufferManager->AddStructuredBuffer(m_data->GetVoxelCount(), sizeof(DirectX::XMUINT2));
	// Number of faces (u2);
	m_bufferManager->AddByteAddressBuffer();

	m_bufferManager->AllocateBuffers();
}

void CVGI::FaceCountTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"FaceCount");

	UINT32 voxelCount = m_data->GetVoxelCount();

	m_cbFaceCount.GridDimension = m_data->GetVoxelGridSize();
	m_cbFaceCount.CurrentPhase = 0;
	m_cbFaceCount.VoxelCount = voxelCount;


	TechniquePass(context, DirectX::XMUINT3(ceil(voxelCount / 256), 1, 1));
	context.Flush();

	m_cbFaceCount.CurrentPhase = 1;

	m_faceCount = *m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)FaceBufferType::FaceCounterBuffer);
	m_bufferManager->ResizeBuffer((UINT)FaceBufferType::FaceDataBuffer, m_faceCount);
	m_bufferManager->ZeroBuffer(context, (UINT)FaceBufferType::FaceCounterBuffer); 

	TechniquePass(context, DirectX::XMUINT3(ceil(voxelCount / 256), 1, 1));

	PIXEndEvent(context.m_commandList->Get());

	m_data->FaceCount = m_faceCount;

	context.Flush();
}

void CVGI::FaceCountTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FaceCountRootSignature::FaceCountCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFaceCount).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::VoxelSRVTable, voxelBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::FaceCountUAVTable, m_bufferManager->GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::FaceCountTechnique::BuildPipelineState()
{
	std::shared_ptr<RootSignature> rootSig = BuildRootSignature();

	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FaceCount_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> faceCountComputePso = std::make_shared<ComputePipelineState>();
	faceCountComputePso->SetRootSignature(rootSig);
	faceCountComputePso->SetComputeShader(computeShader);
	faceCountComputePso->Finalize();
	faceCountComputePso->Name = Name;

	return faceCountComputePso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::FaceCountTechnique::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> faceCountRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)FaceCountRootSignature::Count, 1);
	faceCountRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountCBV].InitAsConstantBuffer(0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::VoxelSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 0);

	faceCountRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return faceCountRootSignature;
}

const std::wstring FaceCountTechnique::Name = L"FaceCountTechnique";