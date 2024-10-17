#pragma once

#include "wrl/client.h"
#include <d3d12.h>
#include <VertexTypes.h>
#include "DX12Lib/Commons/Helpers.h"
#include "unordered_map"

namespace DX12Lib {

	class Shader;
	class RootSignature;
	class CommandList;
	class CommandContext;

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
		friend class CommandContext;

	public:
		PipelineState()
		{
			ThrowIfFailed(CoCreateGuid(&m_guid));
		}
		virtual ~PipelineState() {}



		void SetRootSignature(std::shared_ptr<RootSignature> rootSignature);
		std::shared_ptr<RootSignature> GetRootSignature() const { return m_rootSignature; }

		std::wstring GetGUID() const { return Utils::GUIDToWstring(m_guid); }

	protected:
		virtual void UseRootSignature(CommandList& commandList) const = 0;
		virtual void Use(CommandList& commandList) const = 0;

	public:
		std::wstring Name;

	protected:
		Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
		std::shared_ptr<RootSignature> m_rootSignature;
		GUID m_guid;

	public:
		bool operator==(const PipelineState& rhs) const { return m_guid == rhs.m_guid; }

		ID3D12PipelineState* Get() const { return m_pipelineState.Get(); }
		ID3D12PipelineState** GetAddressOf() { return m_pipelineState.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12PipelineState> GetComPtr() const { return m_pipelineState; }

	};


	class GraphicsPipelineState : public PipelineState
	{
	public:
		GraphicsPipelineState() : PipelineState()
		{
			ZeroMemory(&m_psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
			m_psoDesc.NodeMask = 0;
			m_psoDesc.SampleMask = UINT_MAX;
			m_psoDesc.SampleDesc.Count = 1;
			m_psoDesc.InputLayout.NumElements = 0;
		}

		virtual ~GraphicsPipelineState() = default;

		void SetDesc(D3D12_GRAPHICS_PIPELINE_STATE_DESC desc) { m_psoDesc = desc; }

		void SetShader(std::shared_ptr<Shader> shader, ShaderType type);
		void InitializeDefaultStates();

		virtual void UseRootSignature(CommandList& commandList) const override;
		virtual void Use(CommandList& commandList) const override;

		void SetBlendState(const D3D12_BLEND_DESC& blendDesc) { m_psoDesc.BlendState = blendDesc; }
		void SetRasterizerState(const D3D12_RASTERIZER_DESC& rasterizerDesc) { m_psoDesc.RasterizerState = rasterizerDesc; }
		void SetDepthStencilState(const D3D12_DEPTH_STENCIL_DESC& depthStencilDesc) { m_psoDesc.DepthStencilState = depthStencilDesc; }
		void SetInputLayout(std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout);
		void SetInputLayout(const D3D12_INPUT_ELEMENT_DESC* inputLayout, UINT numElements);
		void SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE type) { m_psoDesc.PrimitiveTopologyType = type; }
		void SetDepthTargetFormat(DXGI_FORMAT format, UINT msaaCount, UINT msaaQuality) { SetRenderTargetFormats(0, nullptr, format, msaaCount, msaaQuality); }
		void SetRenderTargetFormat(DXGI_FORMAT rtvFormat, DXGI_FORMAT dsvFormat, UINT msaaCount, UINT msaaQuality) { SetRenderTargetFormats(1, &rtvFormat, dsvFormat, msaaCount, msaaQuality); }
		void SetRenderTargetFormats(UINT numRTVs, const DXGI_FORMAT* RTVFormats, DXGI_FORMAT DSVFormat, UINT msaaCount = 1, UINT msaaQuality = 0);
		void SetCullMode(D3D12_CULL_MODE cullMode) { m_psoDesc.RasterizerState.CullMode = cullMode; }
		void Finalize();

		D3D12_GRAPHICS_PIPELINE_STATE_DESC GetDesc() const { return m_psoDesc; }
	private:
		D3D12_GRAPHICS_PIPELINE_STATE_DESC m_psoDesc;
		std::unordered_map<ShaderType, std::shared_ptr<Shader>> m_shaders;

	public:
		void operator=(const GraphicsPipelineState& rhs);
	};

	class ComputePipelineState : public PipelineState
	{
	public:
		ComputePipelineState() : PipelineState()
		{
			ZeroMemory(&m_psoDesc, sizeof(D3D12_COMPUTE_PIPELINE_STATE_DESC));
		}

		void UseRootSignature(CommandList& commandList) const override;

		virtual void Use(CommandList& commandList) const override;

		void SetComputeShader(std::shared_ptr<Shader> computeShader);
		void SetComputeShader(const D3D12_SHADER_BYTECODE& computeShader);

		void Finalize();

		virtual ~ComputePipelineState() = default;

	private:
		D3D12_COMPUTE_PIPELINE_STATE_DESC m_psoDesc;
		std::shared_ptr<Shader> m_computeShader;
	};
}



