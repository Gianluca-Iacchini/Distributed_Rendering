#include "DX12Lib/pch.h"
#include "Renderer.h"
#include "DX12Lib/Models/ModelRenderer.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/SceneCamera.h"

#define PSO_SHADOW_OPAQUE L"ShadowOpaquePso"
#define PSO_SHADOW_ALPHA_TEST L"ShadowAlphaTestPso"

#define PSO_SHADOW_TEST L"ShadowTestPso"

using namespace DX12Lib;
using namespace Graphics;

namespace Graphics::Renderer
{
	D3D12_VIEWPORT s_screenViewport;
	D3D12_RECT s_scissorRect;

	std::vector<DX12Lib::ModelRenderer*> m_renderers;
	std::vector<DX12Lib::LightComponent*> m_shadowLights;
	SceneCamera* m_mainCamera = nullptr;

	std::unique_ptr<Swapchain> s_swapchain = nullptr;
	std::unique_ptr<DepthBuffer> s_depthStencilBuffer = nullptr;
	std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap = nullptr;
	std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory = nullptr;
	std::unique_ptr<TextureManager> s_textureManager = nullptr;
	std::unique_ptr<MaterialManager> s_materialManager = nullptr;
	std::unordered_map<std::wstring, std::shared_ptr<PipelineState>> s_PSOs;
	std::unordered_map<std::wstring, std::shared_ptr<Shader>> s_shaders;

	int s_clientWidth = 1920;
	int s_clientHeight = 1080;

	std::unique_ptr<DX12Lib::ShadowBuffer> s_shadowBuffer = nullptr;

	UINT64 backBufferFences[3] = { 0, 0, 0 };

	DescriptorHandle m_shadowMapSRVHandle;
	CostantBufferCommons m_costantBufferCommons;

	void CreateDefaultPSOs();
	void CreateDefaultShaders();

	void InitializeApp()
	{
		s_depthStencilBuffer = std::make_unique<DepthBuffer>();

		s_textureHeap = std::make_shared<DescriptorHeap>();
		s_textureHeap->Create(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);

		s_graphicsMemory = std::make_unique<DirectX::GraphicsMemory>(*s_device);
		s_textureManager = std::make_unique<TextureManager>();
		s_materialManager = std::make_unique<MaterialManager>();

		s_shadowBuffer = std::make_unique<ShadowBuffer>();
		s_shadowBuffer->Create(4096, 4096);

		m_shadowMapSRVHandle = s_textureHeap->Alloc(1);
		s_device->Get()->CopyDescriptorsSimple(1, m_shadowMapSRVHandle, s_shadowBuffer->GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		CreateDefaultShaders();
		CreateDefaultPSOs();
	}

	void AddRendererToQueue(ModelRenderer* renderer)
	{
		m_renderers.push_back(renderer);
	}

	void AddLightToQueue(DX12Lib::LightComponent* light)
	{
		m_shadowLights.push_back(light);
	}

	void AddMainCamera(DX12Lib::SceneCamera* camera)
	{
		assert(camera != nullptr && "Main camera cannot be null");

		m_mainCamera = camera;
	}

	void SetUpRenderFrame(DX12Lib::CommandContext& context)
	{
		ID3D12DescriptorHeap* heaps[] = { Renderer::s_textureHeap->Get() };
		context.m_commandList->Get()->SetDescriptorHeaps(1, heaps);

		// Using Phong as default PSO and root signature just in case
		context.m_commandList->SetPipelineState(s_PSOs[PSO_PHONG_OPAQUE]);
		context.m_commandList->Get()->SetGraphicsRootSignature(s_PSOs[PSO_PHONG_OPAQUE]->GetRootSignature()->Get());
	}

	void RenderLayers(CommandContext& context)
	{ 
		// Shadow pass opaque objects
		s_shadowBuffer->RenderShadowStart(context);

		auto shadowPso = s_PSOs[PSO_SHADOW_OPAQUE];
		for (auto& light : m_shadowLights)
		{
			context.m_commandList->SetPipelineState(shadowPso);
			context.m_commandList->Get()->SetGraphicsRootSignature(shadowPso->GetRootSignature()->Get());

			auto shadowCamera = light->GetShadowCamera();
			
			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)RootSignatureSlot::CameraCBV, shadowCamera->GetShadowCB().GpuAddress());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				mRenderer->DrawOpaque(context);
			}
		}

		shadowPso = s_PSOs[PSO_SHADOW_ALPHA_TEST];

		// Shadow pass transparent objects
		for (auto& light : m_shadowLights)
		{
			context.m_commandList->SetPipelineState(shadowPso);
			context.m_commandList->Get()->SetGraphicsRootSignature(shadowPso->GetRootSignature()->Get());

			auto shadowCamera = light->GetShadowCamera();
			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)RootSignatureSlot::CameraCBV, shadowCamera->GetShadowCB().GpuAddress());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				mRenderer->DrawTransparent(context);
			}
		}
		s_shadowBuffer->RenderShadowEnd(context);

		context.SetViewportAndScissor(s_screenViewport, s_scissorRect);

		auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

		context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
		context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

		context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

		if (m_shadowLights.size() > 0)
		{
			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)RootSignatureSlot::CommonTextureSRV, m_shadowMapSRVHandle);
		}

		m_costantBufferCommons.totalTime = GameTime::GetTotalTime();
		m_costantBufferCommons.deltaTime = GameTime::GetDeltaTime();
		m_costantBufferCommons.numLights = LightComponent::GetLightCount();
		m_costantBufferCommons.renderShadows = m_shadowLights.size() > 0;

		context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::RootSignatureSlot::CommonCBV, s_graphicsMemory->AllocateConstant(m_costantBufferCommons).GpuAddress());

		static bool useMain = true;


		if (s_kbTracker->IsKeyPressed(DirectX::Keyboard::Space))
			useMain = !useMain;

		if (useMain)
			m_mainCamera->UseCamera(context);

		// Main pass opaque
		for (auto& pso : s_PSOs)
		{
			context.m_commandList->SetPipelineState(pso.second);
			context.m_commandList->Get()->SetGraphicsRootSignature(pso.second->GetRootSignature()->Get());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				mRenderer->DrawBatchOpaque(context, pso.first);
			}
		}
		
		// Main pass transparent
		for (auto& pso : s_PSOs)
		{
			context.m_commandList->SetPipelineState(pso.second);
			context.m_commandList->Get()->SetGraphicsRootSignature(pso.second->GetRootSignature()->Get());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				mRenderer->DrawBatchTransparent(context, pso.first);
			}
		}

		m_renderers.clear();
		m_shadowLights.clear();
		m_mainCamera = nullptr;
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

	void InitializeSwapchain(DX12Lib::DX12Window* window)
	{
		s_swapchain = std::make_unique<Swapchain>(*window, m_backBufferFormat);
		s_swapchain->InitializeApp(s_commandQueueManager->GetGraphicsQueue());
	}

	void WaitForSwapchainBuffers()
	{
		UINT64 currentFrame = backBufferFences[s_swapchain->CurrentBufferIndex];
		if (currentFrame != 0)
		{
			s_commandQueueManager->GetGraphicsQueue().WaitForFence(currentFrame);
		}
	}

	DX12Lib::ColorBuffer& GetCurrentBackBuffer()
	{
		return s_swapchain->GetCurrentBackBuffer();
	}

	void ResizeSwapchain(CommandContext* context, int newWidth, int newHeight)
	{
		s_clientWidth = newWidth;
		s_clientHeight = newHeight;

		s_swapchain->Resize(s_clientWidth, s_clientHeight);

		s_swapchain->CurrentBufferIndex = 0;


		Renderer::s_depthStencilBuffer->GetComPtr().Reset();
		Renderer::s_depthStencilBuffer->Create(s_clientWidth, s_clientHeight, m_depthStencilFormat);

		context->TransitionResource(*Renderer::s_depthStencilBuffer, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);

		Renderer::s_screenViewport.TopLeftX = 0;
		Renderer::s_screenViewport.TopLeftY = 0;
		Renderer::s_screenViewport.Width = static_cast<float>(s_clientWidth);
		Renderer::s_screenViewport.Height = static_cast<float>(s_clientHeight);
		Renderer::s_screenViewport.MinDepth = 0.0f;
		Renderer::s_screenViewport.MaxDepth = 1.0f;

		Renderer::s_scissorRect = { 0, 0, s_clientWidth, s_clientHeight};

		m_costantBufferCommons.renderTargetSize = DirectX::XMFLOAT2((float)s_clientWidth, (float)s_clientHeight);
		m_costantBufferCommons.invRenderTargetSize = DirectX::XMFLOAT2(1.0f / s_clientWidth, 1.0f / s_clientHeight);
	}

	void Present(UINT64 fenceVal)
	{
		backBufferFences[s_swapchain->CurrentBufferIndex] = fenceVal;

		auto& curBackBuf = GetCurrentBackBuffer();


		HRESULT hr = s_swapchain->GetComPointer()->Present(0, 0);

		if (FAILED(hr))
		{
			hr = s_device->GetComPtr()->GetDeviceRemovedReason();
			if (FAILED(hr))
			{
				DeviceRemovedHandler();
				ThrowIfFailed(hr);
			}
		}

		CommandContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE_DIRECT);
		s_swapchain->CurrentBufferIndex = (s_swapchain->CurrentBufferIndex + 1) % s_swapchain->BufferCount;
	}

	void CreateDefaultShaders()
	{
		std::wstring srcDir = Utils::ToWstring(SOURCE_DIR);
		std::wstring baseVSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_VS.hlsl";
		std::wstring phongPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_PS.hlsl";
		std::wstring pbrPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\BasicPBR_PS.hlsl";
		std::wstring depthVSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Depth_VS.hlsl";

		std::wstring alphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTest_PS.hlsl";
		std::wstring alphaTestPBRPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestPBR_PS.hlsl";
		std::wstring depthAlphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestDepth_PS.hlsl";

		std::shared_ptr<Shader> basicVS = std::make_shared<Shader>(baseVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> phongPS = std::make_shared<Shader>(phongPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> pbrPS = std::make_shared<Shader>(pbrPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> depthVS = std::make_shared<Shader>(depthVSFile, "VS", "vs_5_1");

		std::shared_ptr<Shader> phongAlphaTestPS = std::make_shared<Shader>(alphaTestPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> pbrAlphaTestPS = std::make_shared<Shader>(alphaTestPBRPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> depthAlphaTestPS = std::make_shared<Shader>(depthAlphaTestPSFile, "PS", "ps_5_1");

		basicVS->Compile();
		phongPS->Compile();
		pbrPS->Compile();
		depthVS->Compile();
		phongAlphaTestPS->Compile();
		pbrAlphaTestPS->Compile();
		depthAlphaTestPS->Compile();

		s_shaders[L"basicVS"] = std::move(basicVS);
		s_shaders[L"phongPS"] = std::move(phongPS);
		s_shaders[L"pbrPS"] = std::move(pbrPS);
		s_shaders[L"depthVS"] = std::move(depthVS);
		s_shaders[L"phongAlphaTestPS"] = std::move(phongAlphaTestPS);
		s_shaders[L"pbrAlphaTestPS"] = std::move(pbrAlphaTestPS);
		s_shaders[L"depthAlphaTestPS"] = std::move(depthAlphaTestPS);
	}

	void CreateDefaultPSOs()
	{
		SamplerDesc DefaultSamplerDesc;
		DefaultSamplerDesc.MaxAnisotropy = 8;

		SamplerDesc ShadowSamplerDesc;
		ShadowSamplerDesc.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
		ShadowSamplerDesc.SetTextureAddressMode(D3D12_TEXTURE_ADDRESS_MODE_BORDER);
		ShadowSamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
		ShadowSamplerDesc.MipLODBias = 0.0f;
		ShadowSamplerDesc.MaxAnisotropy = 16;
		ShadowSamplerDesc.SetBorderColor(Color::Black());
		
		

		std::shared_ptr<RootSignature> baseRootSignature = std::make_shared<RootSignature>((UINT)RootSignatureSlot::Count, 2);
		baseRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		baseRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
		(*baseRootSignature)[(UINT)RootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);
		(*baseRootSignature)[(UINT)RootSignatureSlot::ObjectCBV].InitAsConstantBuffer(1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::CameraCBV].InitAsConstantBuffer(2);
		(*baseRootSignature)[(UINT)RootSignatureSlot::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::MaterialSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::CommonTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, NUM_PHONG_TEXTURES);
		baseRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> pbrRootSignature = std::make_shared<RootSignature>((UINT)RootSignatureSlot::Count, 2);
		pbrRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		pbrRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::ObjectCBV].InitAsConstantBuffer(1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::CameraCBV].InitAsConstantBuffer(2);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::MaterialSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::CommonTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, NUM_PBR_TEXTURES);
		pbrRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);


		std::shared_ptr<PipelineState> phongPso = std::make_shared<PipelineState>();
		phongPso->InitializeDefaultStates();
		phongPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		phongPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		phongPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		phongPso->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		phongPso->SetShader(s_shaders[L"phongPS"], ShaderType::Pixel);
		phongPso->SetRootSignature(baseRootSignature);
		phongPso->Finalize();

		// Duplicate content of opaquePSO
		std::shared_ptr<PipelineState> pbrPso = std::make_shared<PipelineState>();
		*pbrPso = *phongPso;
		pbrPso->SetShader(s_shaders[L"pbrPS"], ShaderType::Pixel);
		pbrPso->Finalize();

		std::shared_ptr<PipelineState> phongAlphaTestPso = std::make_shared<PipelineState>();
		*phongAlphaTestPso = *phongPso;
		phongAlphaTestPso->InitializeDefaultStates();
		phongAlphaTestPso->SetCullMode(D3D12_CULL_MODE_NONE);
		phongAlphaTestPso->SetShader(s_shaders[L"phongAlphaTestPS"], ShaderType::Pixel);
		phongAlphaTestPso->Finalize();

		std::shared_ptr<PipelineState> pbrAlphaTestPso = std::make_shared<PipelineState>();
		*pbrAlphaTestPso = *phongAlphaTestPso;
		pbrAlphaTestPso->SetShader(s_shaders[L"pbrAlphaTestPS"], ShaderType::Pixel);
		pbrAlphaTestPso->Finalize();

		D3D12_RASTERIZER_DESC shadowRastDesc = {};
		shadowRastDesc.FillMode = D3D12_FILL_MODE_SOLID;
		shadowRastDesc.CullMode = D3D12_CULL_MODE_BACK;
		shadowRastDesc.FrontCounterClockwise = FALSE;
		shadowRastDesc.DepthBias = 300;
		shadowRastDesc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
		shadowRastDesc.SlopeScaledDepthBias = 1.0;
		shadowRastDesc.DepthClipEnable = TRUE;
		shadowRastDesc.MultisampleEnable = FALSE;
		shadowRastDesc.AntialiasedLineEnable = FALSE;
		shadowRastDesc.ForcedSampleCount = 0;
		shadowRastDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

		

		D3D12_BLEND_DESC blendNoColorWrite = {};
		blendNoColorWrite.IndependentBlendEnable = FALSE;
		blendNoColorWrite.RenderTarget[0].BlendEnable = FALSE;
		blendNoColorWrite.RenderTarget[0].LogicOpEnable = FALSE;
		blendNoColorWrite.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
		blendNoColorWrite.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
		blendNoColorWrite.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
		blendNoColorWrite.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
		blendNoColorWrite.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
		blendNoColorWrite.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
		blendNoColorWrite.RenderTarget[0].RenderTargetWriteMask = 0;

		D3D12_DEPTH_STENCIL_DESC depthTestDesc = {};
		depthTestDesc.DepthEnable = TRUE;
		depthTestDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
		depthTestDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
		depthTestDesc.StencilEnable = FALSE;
		depthTestDesc.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
		depthTestDesc.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
		depthTestDesc.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
		depthTestDesc.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
		depthTestDesc.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
		depthTestDesc.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
		depthTestDesc.BackFace = depthTestDesc.FrontFace;

		std::shared_ptr<PipelineState> shadowPso = std::make_shared<PipelineState>();
		shadowPso->InitializeDefaultStates();
		shadowPso->SetRasterizerState(shadowRastDesc);
		shadowPso->SetBlendState(blendNoColorWrite);
		shadowPso->SetDepthStencilState(depthTestDesc);
		shadowPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
					DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		shadowPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		shadowPso->SetRenderTargetFormats(0, nullptr, s_shadowBuffer->GetFormat());
		shadowPso->SetShader(s_shaders[L"depthVS"], ShaderType::Vertex);
		shadowPso->SetRootSignature(pbrRootSignature);
		shadowPso->Finalize();

		D3D12_RASTERIZER_DESC shadowAlphaTestRastDesc = shadowRastDesc;
		shadowAlphaTestRastDesc.CullMode = D3D12_CULL_MODE_NONE;
		std::shared_ptr<PipelineState> shadowAlphaTested = std::make_shared<PipelineState>();
		*shadowAlphaTested = *shadowPso;
		shadowAlphaTested->SetRasterizerState(shadowAlphaTestRastDesc);
		shadowAlphaTested->SetShader(s_shaders[L"depthAlphaTestPS"], ShaderType::Pixel);
		shadowAlphaTested->Finalize();

		std::shared_ptr<PipelineState> shadowTestPso = std::make_shared<PipelineState>();
		shadowTestPso->InitializeDefaultStates();
		shadowTestPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
					DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		shadowTestPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		shadowTestPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		shadowTestPso->SetShader(s_shaders[L"depthVS"], ShaderType::Vertex);
		shadowTestPso->SetShader(s_shaders[L"depthAlphaTestPS"], ShaderType::Pixel);
		shadowTestPso->SetRootSignature(pbrRootSignature);
		shadowTestPso->Finalize();

		s_PSOs[PSO_PHONG_OPAQUE] = std::move(phongPso);
		s_PSOs[PSO_PBR_OPAQUE] = std::move(pbrPso);

		s_PSOs[PSO_PHONG_ALPHA_TEST] = std::move(phongAlphaTestPso);
		s_PSOs[PSO_PBR_ALPHA_TEST] = std::move(pbrAlphaTestPso);

		s_PSOs[PSO_SHADOW_OPAQUE] = std::move(shadowPso);
		s_PSOs[PSO_SHADOW_ALPHA_TEST] = std::move(shadowAlphaTested);

		s_PSOs[PSO_SHADOW_TEST] = std::move(shadowTestPso);
	}
}
