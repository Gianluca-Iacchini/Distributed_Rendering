#include "LerpRadianceTechnique.h"
#include "DX12Lib/pch.h"
#include "LerpRadiance_CS.h"
#include "LightTransportTechnique.h"
#include "GaussianFilterTechnique.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace DirectX;
using namespace Graphics;

CVGI::LerpRadianceTechnique::LerpRadianceTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->SetBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::LerpRadianceTechnique::InitializeBuffers()
{
	// New Radiance
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2), L"NewBuffer");
	// Old Radiance
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2), L"OldBuffer");
	// Old Gaussian Radiance
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2), L"OldGaussianBuffer");
	m_bufferManager->AllocateBuffers();
}

void CVGI::LerpRadianceTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(0, 255, 0), Name.c_str());

	m_cbLerpRadiance.FaceCount = m_data->FaceCount;

	TechniquePass(context, DirectX::XMUINT3(ceilf(m_data->FaceCount / 128.0f), 1, 1));

	m_cbLerpRadiance.CurrentPhase = 0;

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::LerpRadianceTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name.c_str()].get());

	auto& gaussianBufferManager = m_data->GetBufferManager(GaussianFilterTechnique::ReadName);

	gaussianBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0), true);

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LerpRadianceRootSignature::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)LerpRadianceRootSignature::LerpRadianceCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbLerpRadiance).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LerpRadianceRootSignature::GaussianFilterBufferUAV, gaussianBufferManager.GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)LerpRadianceRootSignature::LerpBufferUAV, m_bufferManager->GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void CVGI::LerpRadianceTechnique::SetMaxTime(float maxTime)
{
	m_cbLerpRadiance.maxTime = maxTime;
}

void CVGI::LerpRadianceTechnique::SetAccumulatedTime(float accumulatedTime)
{
	m_cbLerpRadiance.accumulatedTime = accumulatedTime;
}

void CVGI::LerpRadianceTechnique::SetPhase(UINT phase)
{
	m_cbLerpRadiance.CurrentPhase = phase;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::LerpRadianceTechnique::BuildPipelineState()
{
	auto lerpRootSignature = this->BuildRootSignature();

	auto lerpShaderBytecode = CD3DX12_SHADER_BYTECODE((void*)g_pLerpRadiance_CS, ARRAYSIZE(g_pLerpRadiance_CS));

	std::shared_ptr<DX12Lib::ComputePipelineState> lerpPipelineState = std::make_shared<DX12Lib::ComputePipelineState>();
	lerpPipelineState->SetRootSignature(lerpRootSignature);
	lerpPipelineState->SetComputeShader(lerpShaderBytecode);
	lerpPipelineState->Finalize();
	lerpPipelineState->Name = Name.c_str();

	return lerpPipelineState;
}


std::shared_ptr<DX12Lib::RootSignature> CVGI::LerpRadianceTechnique::BuildRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> lerpRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)LerpRadianceRootSignature::Count, 0);

	(*lerpRootSignature)[(UINT)LerpRadianceRootSignature::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*lerpRootSignature)[(UINT)LerpRadianceRootSignature::LerpRadianceCBV].InitAsConstantBuffer(1);
	(*lerpRootSignature)[(UINT)LerpRadianceRootSignature::GaussianFilterBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*lerpRootSignature)[(UINT)LerpRadianceRootSignature::LerpBufferUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 1);

	lerpRootSignature->Finalize();

	return lerpRootSignature;
}

const std::wstring CVGI::LerpRadianceTechnique::Name = L"LerpRadianceTechnique";