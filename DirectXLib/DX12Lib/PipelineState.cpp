#include "PipelineState.h"
#include "Device.h"
#include "Shader.h"
#include <iostream>

using namespace Microsoft::WRL;

PipelineState::PipelineState(DXGI_FORMAT backBufferFormat, DXGI_FORMAT depthStencilFormat, D3D12_GRAPHICS_PIPELINE_STATE_DESC* psoDesc)
{
	if (psoDesc != nullptr)
	{
		m_psoDesc = *psoDesc;
	}
	else
	{
		ZeroMemory(&m_psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
		m_psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		m_psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		m_psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		m_psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		m_psoDesc.NumRenderTargets = 1;
		m_psoDesc.RTVFormats[0] = backBufferFormat;
		m_psoDesc.SampleDesc.Count = 1;
		m_psoDesc.SampleDesc.Quality = 0;
		m_psoDesc.DSVFormat = depthStencilFormat;
	}
}

void PipelineState::SetShader(std::shared_ptr<Shader> shader, ShaderType shaderType)
{
	if (shaderType == ShaderType::Count)
	{
		return;
	}

	m_shaders[shaderType] = shader;

	auto shaderByteCode = CD3DX12_SHADER_BYTECODE(shader.get()->GetShaderByteBlob().Get());


	switch (shaderType)	
	{
	case ShaderType::Vertex:
		m_psoDesc.VS = shaderByteCode;
		//m_psoDesc.InputLayout.NumElements = shader.get()->InputLayout.size();
		//m_psoDesc.InputLayout.pInputElementDescs = shader.get()->InputLayout.data();
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
	//case ShaderType::Compute:
	//	m_cpsoDesc.CS = shaderByteCode;
	//	break;
	case ShaderType::Count:
		break;
	default:
		break;
	}
}

void PipelineState::SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout)
{
	m_psoDesc.InputLayout.NumElements = static_cast<UINT>(inputLayout.size());
	m_psoDesc.InputLayout.pInputElementDescs = inputLayout.data();
}

void PipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
{
	m_psoDesc.pRootSignature = rootSignature;
}

void PipelineState::Finalize(Device& device)
{
	ThrowIfFailed(device.GetComPtr()->CreateGraphicsPipelineState(&m_psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())))
}





