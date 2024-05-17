#include "DX12Lib/pch.h"
#include "Renderer.h"
#include "DX12Lib/Models/ModelRenderer.h"

using namespace DX12Lib;
using namespace Graphics;

namespace Graphics::Renderer
{
	std::vector<DX12Lib::ModelRenderer*> m_renderers;
	std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap = nullptr;
	std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory = nullptr;
	std::unique_ptr<TextureManager> s_textureManager = nullptr;
	std::unique_ptr<MaterialManager> s_materialManager = nullptr;
	std::unordered_map<std::wstring, std::shared_ptr<PipelineState>> s_PSOs;
	std::unordered_map<std::wstring, std::shared_ptr<Shader>> s_shaders;

	void CreateDefaultPSOs();
	void CreateDefaultShaders();

	void Initialize()
	{
		s_textureHeap = std::make_shared<DescriptorHeap>();
		s_textureHeap->Create(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);

		s_graphicsMemory = std::make_unique<DirectX::GraphicsMemory>(*s_device);
		s_textureManager = std::make_unique<TextureManager>();
		s_materialManager = std::make_unique<MaterialManager>();

		CreateDefaultShaders();
		CreateDefaultPSOs();
	}

	void AddRendererToQueue(ModelRenderer* renderer)
	{
		m_renderers.push_back(renderer);
	}

	void RenderLayers(CommandContext* context)
	{
		for (auto& pso : s_PSOs)
		{
			context->m_commandList->SetPipelineState(pso.second);
			context->m_commandList->Get()->SetGraphicsRootSignature(pso.second->GetRootSignature()->Get());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				mRenderer->DrawMeshes(context->m_commandList->Get(), pso.first);
			}
		}

		m_renderers.clear();
	}

	void Shutdown()
	{
		m_renderers.clear();
		s_textureHeap = nullptr;
		s_graphicsMemory = nullptr;
		s_textureManager = nullptr;
		s_materialManager = nullptr;
		s_PSOs.clear();
		s_shaders.clear();
	}

	void CreateDefaultShaders()
	{
		std::wstring srcDir = Utils::ToWstring(SOURCE_DIR);
		std::wstring VSshaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_VS.hlsl";
		std::wstring PSshaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_PS.hlsl";
		std::wstring PBRPSShaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\BasicPBR_PS.hlsl";

		std::shared_ptr<Shader> baseVertexShader = std::make_shared<Shader>(VSshaderFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> basePixelShader = std::make_shared<Shader>(PSshaderFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> PBRPixelShader = std::make_shared<Shader>(PBRPSShaderFile, "PS", "ps_5_1");

		baseVertexShader->Compile();
		basePixelShader->Compile();
		PBRPixelShader->Compile();

		s_shaders[L"basicVS"] = std::move(baseVertexShader);
		s_shaders[L"basicPS"] = std::move(basePixelShader);
		s_shaders[L"PBRBasicPS"] = std::move(PBRPixelShader);
	}

	void CreateDefaultPSOs()
	{
		SamplerDesc DefaultSamplerDesc;
		DefaultSamplerDesc.MaxAnisotropy = 8;

		std::shared_ptr<RootSignature> baseRootSignature = std::make_shared<RootSignature>(4, 1);
		baseRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		(*baseRootSignature)[0].InitAsConstantBuffer(0);
		(*baseRootSignature)[1].InitAsConstantBuffer(1);
		(*baseRootSignature)[2].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*baseRootSignature)[3].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PHONG_TEXTURES);
		baseRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> pbrRootSignature = std::make_shared<RootSignature>(4, 1);
		pbrRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		(*pbrRootSignature)[0].InitAsConstantBuffer(0);
		(*pbrRootSignature)[1].InitAsConstantBuffer(1);
		(*pbrRootSignature)[2].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*pbrRootSignature)[3].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);
		pbrRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);





		std::shared_ptr<PipelineState> phongPso = std::make_shared<PipelineState>();
		phongPso->InitializeDefaultStates();
		phongPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		phongPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		phongPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		phongPso->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		phongPso->SetShader(s_shaders[L"basicPS"], ShaderType::Pixel);
		phongPso->SetRootSignature(baseRootSignature);
		phongPso->Finalize();

		// Duplicate content of opaquePSO
		std::shared_ptr<PipelineState> pbrPso = std::make_shared<PipelineState>();
		pbrPso->InitializeDefaultStates();
		pbrPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		pbrPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		pbrPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		pbrPso->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		pbrPso->SetShader(s_shaders[L"PBRBasicPS"], ShaderType::Pixel);
		pbrPso->SetRootSignature(pbrRootSignature);
		pbrPso->Finalize();





		s_PSOs[PSO_PHONG_OPAQUE] = std::move(phongPso);
		s_PSOs[PSO_PBR_OPAQUE] = std::move(pbrPso);
	}
}
