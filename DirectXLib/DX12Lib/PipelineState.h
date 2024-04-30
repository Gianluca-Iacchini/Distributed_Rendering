#include "Helpers.h"

#ifndef PIPELINE_STATE_H
#define PIPELINE_STATE_H

class Shader;
class RootSignature;

enum class ShaderType
{
	Vertex = 0,
	Pixel,
	Geometry,
	Hull,
	Domain,
	Count
};

class PipelineState
{
public:
	PipelineState() 
	{
		ZeroMemory(&m_psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
		m_psoDesc.NodeMask = 0;
		m_psoDesc.SampleMask = UINT_MAX;
		m_psoDesc.SampleDesc.Count = 1;
		m_psoDesc.InputLayout.NumElements = 0;
	}
	~PipelineState() {};
	
	void SetShader(std::shared_ptr<Shader> shader, ShaderType type);
	void InitializeDefaultStates();
	void SetRootSignature(std::shared_ptr<RootSignature> rootSignature);
	void SetBlendState(const D3D12_BLEND_DESC& blendDesc) { m_psoDesc.BlendState = blendDesc; }
	void SetRasterizerState(const D3D12_RASTERIZER_DESC& rasterizerDesc) { m_psoDesc.RasterizerState = rasterizerDesc; }
	void SetDepthStencilState(const D3D12_DEPTH_STENCIL_DESC& depthStencilDesc) { m_psoDesc.DepthStencilState = depthStencilDesc; }
	void SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout);
	void SetInputLayout(const D3D12_INPUT_ELEMENT_DESC* inputLayout, UINT numElements);
	void SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE type) { m_psoDesc.PrimitiveTopologyType = type; }
	void SetDepthTargetFormat(DXGI_FORMAT format, UINT msaaCount, UINT msaaQuality) { SetRenderTargetFormats(0, nullptr, format, msaaCount, msaaQuality); }
	void SetRenderTargetFormat(DXGI_FORMAT rtvFormat, DXGI_FORMAT dsvFormat, UINT msaaCount, UINT msaaQuality) { SetRenderTargetFormats(1, &rtvFormat, dsvFormat, msaaCount, msaaQuality); }
	void SetRenderTargetFormats(UINT numRTVs, const DXGI_FORMAT* RTVFormats, DXGI_FORMAT DSVFOrmat, UINT msaaCount, UINT msaaQuality);
	void Finalize();

private:
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
	D3D12_GRAPHICS_PIPELINE_STATE_DESC m_psoDesc;
	std::unordered_map<ShaderType, std::shared_ptr<Shader>> m_shaders;
	std::shared_ptr<RootSignature> m_rootSignature;

public:
	PipelineState(PipelineState&& rhs) = default;
	PipelineState& operator=(PipelineState&& rhs) = default;

	PipelineState(const PipelineState& rhs) = delete;
	PipelineState& operator=(const PipelineState& rhs) = delete;

	ID3D12PipelineState* Get() const { return m_pipelineState.Get(); }
	ID3D12PipelineState** GetAddressOf() { return m_pipelineState.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12PipelineState> GetComPtr() const { return m_pipelineState; }


};

#endif // !PIPELINE_STATE_H



