#include "DX12Lib/pch.h"
#include "PipelineState.h"
#include "Shader.h"
#include "RootSignature.h"

using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;



void PipelineState::SetRootSignature(std::shared_ptr<RootSignature> rootSignature)
{
	m_rootSignature = rootSignature;
}

void GraphicsPipelineState::SetShader(std::shared_ptr<Shader> shader, ShaderType shaderType)
{
	if (shaderType == ShaderType::Count)
	{
		return;
	}

	m_shaders[shaderType] = shader;

	auto shaderByteCode = CD3DX12_SHADER_BYTECODE(shader->GetShaderByteBlob().Get());


	switch (shaderType)
	{
	case ShaderType::Vertex:
		m_psoDesc.VS = shaderByteCode;
		break;
	case ShaderType::Pixel:
		m_psoDesc.PS = shaderByteCode;
		break;
	case ShaderType::Geometry:
		m_psoDesc.GS = shaderByteCode;
		break;
	case ShaderType::Hull:
		m_psoDesc.HS = shaderByteCode;
		break;
	case ShaderType::Domain:
		m_psoDesc.DS = shaderByteCode;
		break;
	case ShaderType::Count:
		break;
	default:
		break;
	}
}

void GraphicsPipelineState::InitializeDefaultStates()
{
	this->SetBlendState(CD3DX12_BLEND_DESC(D3D12_DEFAULT));
	this->SetRasterizerState(CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT));
	this->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));

}

void DX12Lib::GraphicsPipelineState::UseRootSignature(CommandList& commandList) const
{
	commandList.Get()->SetGraphicsRootSignature(m_rootSignature->Get());
}

void DX12Lib::GraphicsPipelineState::Use(CommandList& commandList) const
{
	commandList.Get()->SetPipelineState(m_pipelineState.Get());
}

void GraphicsPipelineState::SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout)
{
	m_psoDesc.InputLayout.NumElements = static_cast<UINT>(inputLayout.size());
	m_psoDesc.InputLayout.pInputElementDescs = inputLayout.data();
}

void GraphicsPipelineState::SetInputLayout(const D3D12_INPUT_ELEMENT_DESC* inputLayout, UINT numElements)
{
	m_psoDesc.InputLayout.NumElements = numElements;
	m_psoDesc.InputLayout.pInputElementDescs = inputLayout;
}

void GraphicsPipelineState::SetRenderTargetFormats(UINT numRTVs, const DXGI_FORMAT* RTVFormats, DXGI_FORMAT DSVFormat, UINT msaaCount, UINT msaaQuality)
{
	assert(numRTVs == 0 || RTVFormats != nullptr && "Array data and array size do not match");

	for (UINT i = 0; i < numRTVs; ++i)
	{
		assert(RTVFormats[i] != DXGI_FORMAT_UNKNOWN && "Invalid RTV format");
		m_psoDesc.RTVFormats[i] = RTVFormats[i];
	}

	for (UINT i = numRTVs; i < m_psoDesc.NumRenderTargets; ++i)
	{
		m_psoDesc.RTVFormats[i] = DXGI_FORMAT_UNKNOWN;
	}

	m_psoDesc.NumRenderTargets = numRTVs;
	m_psoDesc.DSVFormat = DSVFormat;
	m_psoDesc.SampleDesc.Count = msaaCount;
	m_psoDesc.SampleDesc.Quality = msaaQuality;
}

void GraphicsPipelineState::Finalize()
{
	m_psoDesc.pRootSignature = m_rootSignature->Get();
	ThrowIfFailed(s_device->GetComPtr()->CreateGraphicsPipelineState(&m_psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())));
}

void DX12Lib::GraphicsPipelineState::operator=(const GraphicsPipelineState& rhs)
{
	m_psoDesc = rhs.m_psoDesc;
	m_rootSignature = rhs.m_rootSignature;

	this->m_shaders.clear();

	for (auto& shader : rhs.m_shaders)
	{
		this->m_shaders[shader.first] = shader.second;
	}
}

void DX12Lib::ComputePipelineState::UseRootSignature(CommandList& commandList) const
{
	commandList.Get()->SetComputeRootSignature(m_rootSignature->Get());
}

void DX12Lib::ComputePipelineState::Use(CommandList& commandList) const
{
	commandList.Get()->SetPipelineState(m_pipelineState.Get());
}

void DX12Lib::ComputePipelineState::SetComputeShader(std::shared_ptr<Shader> computeShader)
{
	m_computeShader = computeShader;
	m_psoDesc.CS = CD3DX12_SHADER_BYTECODE(m_computeShader->GetShaderByteBlob().Get());
}

void DX12Lib::ComputePipelineState::SetComputeShader(const D3D12_SHADER_BYTECODE& computeShader)
{
	m_psoDesc.CS = computeShader;
}

void DX12Lib::ComputePipelineState::Finalize()
{
	m_psoDesc.pRootSignature = m_rootSignature->Get();
	ThrowIfFailed(s_device->GetComPtr()->CreateComputePipelineState(&m_psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())));
}
