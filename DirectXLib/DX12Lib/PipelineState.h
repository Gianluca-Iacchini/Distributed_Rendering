#include "Helpers.h"

#ifndef PIPELINE_STATE_H
#define PIPELINE_STATE_H

class Device;
class Shader;

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
	PipelineState(DXGI_FORMAT backBufferFormat, DXGI_FORMAT depthStencilFormat, D3D12_GRAPHICS_PIPELINE_STATE_DESC* psoDesc = nullptr);
	~PipelineState() {};
	void SetShader(std::shared_ptr<Shader> shader, ShaderType type);
	void SetBlendState(const D3D12_BLEND_DESC& blendDesc) { m_psoDesc.BlendState = blendDesc; }
	void SetRasterizerState(const D3D12_RASTERIZER_DESC& rasterizerDesc) { m_psoDesc.RasterizerState = rasterizerDesc; }
	void SetDepthStencilState(const D3D12_DEPTH_STENCIL_DESC& depthStencilDesc) { m_psoDesc.DepthStencilState = depthStencilDesc; }
	void SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout);
	void SetRootSignature(ID3D12RootSignature* rootSignature);
	void Finalize(Device& device);

private:
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
	D3D12_GRAPHICS_PIPELINE_STATE_DESC m_psoDesc;
	std::unordered_map<ShaderType, std::shared_ptr<Shader>> m_shaders;

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



