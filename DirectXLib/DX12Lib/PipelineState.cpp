#include "PipelineState.h"
#include "Device.h"
#include "Shader.h"
#include <iostream>
#include "GraphicsCore.h"
#include "RootSignature.h"

using namespace Microsoft::WRL;
using namespace Graphics;

//PipelineState::PipelineState(DXGI_FORMAT backBufferFormat, DXGI_FORMAT depthStencilFormat, D3D12_GRAPHICS_PIPELINE_STATE_DESC* psoDesc)
//{
//	if (psoDesc != nullptr)
//	{
//		m_psoDesc = *psoDesc;
//	}
//	else
//	{
//		ZeroMemory(&m_psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
//		m_psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
//		m_psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
//		m_psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
//		m_psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
//		m_psoDesc.NumRenderTargets = 1;
//		m_psoDesc.RTVFormats[0] = backBufferFormat;
//		m_psoDesc.SampleDesc.Count = 1;
//		m_psoDesc.SampleDesc.Quality = 0;
//		m_psoDesc.DSVFormat = depthStencilFormat;
//	}
//}

void PipelineState::SetShader(std::shared_ptr<Shader> shader, ShaderType shaderType)
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

void PipelineState::InitializeDefaultStates()
{
	this->SetBlendState(CD3DX12_BLEND_DESC(D3D12_DEFAULT));
	this->SetRasterizerState(CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT));
	this->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
}

void PipelineState::SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout)
{
	m_psoDesc.InputLayout.NumElements = static_cast<UINT>(inputLayout.size());
	m_psoDesc.InputLayout.pInputElementDescs = inputLayout.data();
}

void PipelineState::SetRootSignature(std::shared_ptr<RootSignature> rootSignature)
{
	m_rootSignature = rootSignature;
	m_psoDesc.pRootSignature = rootSignature->Get();
}

void PipelineState::SetRenderTargetFormats(UINT numRTVs, const DXGI_FORMAT* RTVFormats, DXGI_FORMAT DSVFormat, UINT msaaCount, UINT msaaQuality)
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

void PipelineState::Finalize()
{
	ThrowIfFailed(s_device->GetComPtr()->CreateGraphicsPipelineState(&m_psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())))
}





