#include "DX12Lib/pch.h"
#include "Renderer.h"
#include "DX12Lib/Models/ModelRenderer.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include <WinPixEventRuntime/pix3.h>

#define USE_RTGI 1

#include "../ClusteredVoxelGI/Technique.h"

#define PSO_SHADOW_OPAQUE L"ShadowOpaquePso"
#define PSO_SHADOW_ALPHA_TEST L"ShadowAlphaTestPso"

#define PSO_SHADOW_TEST L"ShadowTestPso"

using namespace DX12Lib;
using namespace Graphics;

namespace Graphics::Renderer
{
	enum class DeferredRootSignatureSlot
	{
		CommonCBV = 0,
		CameraCBV = 1,
		VoxelRTGICBV,
		LightSRV,
		MaterialSRV,
		CommonTextureSRV,
		GBufferSRV,
		VoxelBufferSRV,
		CompactBufferSRV,
		ClusterBufferSRV,
		RadianceBufferSRV,
		Count
	};

	// Render targets for deferred rendering, this could be incorporeted together
	// (e.g. Storing material in same texture as world pos)
	// But keeping them separate helps with debugging and understanding the process
	enum RenderTargetType
	{
		WorldPosMatIndex,		// World position in xyz, material index in w
		Normal,					// Normal in xy, third component computed in shader
		Diffuse,				// Diffuse albedo
		MetallicRoughnessAO,	// AO in the R channel, Roughness in the G channel and Metallic in the B channel
		Count
	};



	D3D12_VIEWPORT s_screenViewport;
	D3D12_RECT s_scissorRect;

	std::vector<DX12Lib::ModelRenderer*> m_renderers;
	std::vector<DX12Lib::ShadowCamera*> m_shadowLights;
	SceneCamera* m_mainCamera = nullptr;

	std::unique_ptr<Swapchain> s_swapchain = nullptr;
	std::unique_ptr<DepthBuffer> s_depthStencilBuffer = nullptr;
	std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap = nullptr;
	std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory = nullptr;
	std::unique_ptr<TextureManager> s_textureManager = nullptr;
	std::unique_ptr<MaterialManager> s_materialManager = nullptr;
	std::unordered_map<std::wstring, std::shared_ptr<PipelineState>> s_PSOs;
	std::unordered_map<std::wstring, std::shared_ptr<Shader>> s_shaders;

	DescriptorHandle m_commonTextureSRVHandle;

	int s_clientWidth = 1920;
	int s_clientHeight = 1080;

	bool sEnableRenderMainPass = true;
	bool sEnableRenderShadows = true;

	std::unique_ptr<DX12Lib::ShadowBuffer> s_rtgiShadowBuffer = nullptr;
	DX12Lib::ColorBuffer* s_voxel3DTexture = nullptr;

	std::vector<std::unique_ptr<DX12Lib::ColorBuffer>> s_renderTargets;

	UINT64 backBufferFences[3] = { 0, 0, 0 };

	// Ping ponging shadow textures for RTGI
	DescriptorHandle m_rtgiShadowTextureHandle;


	DescriptorHandle m_gbufferStartHandle;
	CostantBufferCommons m_costantBufferCommons;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;

	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

	bool m_useRTGI = false;
	std::weak_ptr<CVGI::TechniqueData> m_rtgiData;

	// For debug only
	ConstantBufferVoxelCommons m_cbVoxelCommons;

	void CreateDefaultPSOs();
	void CreateDefaultShaders();
	void BuildRenderQuadGeometry();

	void InitializeApp()
	{
		s_depthStencilBuffer = std::make_unique<DepthBuffer>();
		s_depthStencilBuffer->Create(s_clientWidth, s_clientHeight, m_depthStencilFormat);

		s_textureHeap = std::make_shared<DescriptorHeap>();
		s_textureHeap->Create(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);

		s_graphicsMemory = std::make_unique<DirectX::GraphicsMemory>(*s_device);
		s_textureManager = std::make_unique<TextureManager>();
		s_materialManager = std::make_unique<MaterialManager>();

		s_renderTargets.resize((UINT)RenderTargetType::Count);

		s_renderTargets[(UINT)RenderTargetType::WorldPosMatIndex] = std::make_unique<ColorBuffer>();
		s_renderTargets[(UINT)RenderTargetType::WorldPosMatIndex]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R32G32B32A32_FLOAT);

		// We compute the third normal component in the shader, so we only need to store two components
		s_renderTargets[(UINT)RenderTargetType::Normal] = std::make_unique<ColorBuffer>();
		s_renderTargets[(UINT)RenderTargetType::Normal]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R16G16_FLOAT);

		s_renderTargets[(UINT)RenderTargetType::Diffuse] = std::make_unique<ColorBuffer>(Color::LightSteelBlue());
		s_renderTargets[(UINT)RenderTargetType::Diffuse]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

		s_renderTargets[(UINT)RenderTargetType::MetallicRoughnessAO] = std::make_unique<ColorBuffer>(Color::Red());
		s_renderTargets[(UINT)RenderTargetType::MetallicRoughnessAO]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

		s_rtgiShadowBuffer = std::make_unique<ShadowBuffer>();
		s_rtgiShadowBuffer->Create(2048, 2048);

		m_commonTextureSRVHandle = s_textureHeap->Alloc(2);
		m_rtgiShadowTextureHandle = s_textureHeap->Alloc(1);

		s_device->Get()->CopyDescriptorsSimple(1, m_rtgiShadowTextureHandle, s_rtgiShadowBuffer->GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);


		m_gbufferStartHandle = s_textureHeap->Alloc((UINT)RenderTargetType::Count);
		UINT gbufferStartIndex = s_textureHeap->GetOffsetOfHandle(m_gbufferStartHandle);

		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			s_device->Get()->CopyDescriptorsSimple(1, 
				(*s_textureHeap)[gbufferStartIndex + i],
				s_renderTargets[i]->GetSRV(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		CreateDefaultShaders();
		CreateDefaultPSOs();
		BuildRenderQuadGeometry();
		s_materialManager->LoadDefaultMaterials(*s_textureManager);

	}

	void BuildRenderQuadGeometry()
	{
		float vertices[] = {
			  // Positions        // Texture Coords
			-1.0f, -1.0f, 0.0f,		0.0f, 1.0f,		    // Bottom Left
			-1.0f,  1.0f, 0.0f,		0.0f, 0.0f,			// Top Left
			 1.0f, -1.0f, 0.0f,		1.0f, 1.0f,		    // Bottom Right
			 1.0f,  1.0f, 0.0f,		1.0f, 0.0f			// Top Right
		};

		UINT16 indices[] = {
			0, 1, 2,
			2, 1, 3
		};

		UINT numVertices = 4;
		UINT numIndices = 6;

		UINT vertexStride = sizeof(float) * 5;
		m_vertexBufferResource = Utils::CreateDefaultBuffer(vertices, vertexStride * numVertices);
		m_vertexBufferView.BufferLocation = m_vertexBufferResource->GetGPUVirtualAddress();
		m_vertexBufferView.StrideInBytes = vertexStride;
		m_vertexBufferView.SizeInBytes = vertexStride * numVertices;

		m_indexBufferResource = Utils::CreateDefaultBuffer(indices, sizeof(UINT16) * numIndices);
		m_indexBufferView.BufferLocation = m_indexBufferResource->GetGPUVirtualAddress();
		m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
		m_indexBufferView.SizeInBytes = sizeof(UINT16) * numIndices;
	}

	void AddRendererToQueue(ModelRenderer* renderer)
	{
		m_renderers.push_back(renderer);
	}

	void AddShadowCamera(DX12Lib::ShadowCamera* shadowCamera)
	{
		m_shadowLights.push_back(shadowCamera);
	}

	DX12Lib::DescriptorHandle& GetShadowMapSrv()
	{
		return m_commonTextureSRVHandle;
	}

	void AddMainCamera(DX12Lib::SceneCamera* camera)
	{
		assert(camera != nullptr && "Main camera cannot be null");

		m_mainCamera = camera;
	}

	std::vector<DX12Lib::ModelRenderer*> GetRenderers()
	{
		return m_renderers;
	}

	void SetUpRenderFrame(DX12Lib::CommandContext& context)
	{
		context.SetDescriptorHeap(s_textureHeap.get());

		// Using Phong as default PSO and root signature just in case
		context.SetPipelineState(s_PSOs[PSO_PHONG_OPAQUE].get());

		m_costantBufferCommons.totalTime = GameTime::GetTotalTime();
		m_costantBufferCommons.deltaTime = GameTime::GetDeltaTime();
		m_costantBufferCommons.numLights = LightComponent::GetLightCount();
	}

	void ShadowPass(DX12Lib::GraphicsContext& context)
	{

		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(40, 40, 40), L"ShadowPass");

		if (m_shadowLights.size() <= 0 || !sEnableRenderShadows)
		{
			return;
		}

		for (UINT32 i = 0; i < m_shadowLights.size(); i++)
		{
			auto sl = m_shadowLights[i];
			auto descriptorStartSize = m_commonTextureSRVHandle + s_textureHeap->GetDescriptorSize() * i;
			s_device->Get()->CopyDescriptorsSimple(1, descriptorStartSize, sl->GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		// Shadow pass opaque objects
		auto shadowPso = s_PSOs[PSO_SHADOW_OPAQUE];
		for (auto sl : m_shadowLights)
		{
			context.SetPipelineState(shadowPso.get());

			sl->GetShadowBuffer().RenderShadowStart(context, true);

			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)RootSignatureSlot::CameraCBV, sl->GetShadowCB().GpuAddress());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				auto batch = mRenderer->GetAllOpaque();

				mRenderer->Model->UseBuffers(context);

				for (auto& mesh : batch)
				{
					context.m_commandList->Get()->SetGraphicsRootDescriptorTable((UINT)RootSignatureSlot::MaterialTextureSRV, mesh->GetMaterialTextureSRV());
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)RootSignatureSlot::ObjectCBV, mesh->GetObjectCB().GpuAddress());
					mesh->DrawMesh(context);
				}
			}

			sl->GetShadowBuffer().RenderShadowEnd(context);
		}

		shadowPso = s_PSOs[PSO_SHADOW_ALPHA_TEST];

		// Shadow pass transparent objects
		for (auto& sl : m_shadowLights)
		{
			context.SetPipelineState(shadowPso.get());

			sl->GetShadowBuffer().RenderShadowStart(context, false);

			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)RootSignatureSlot::CameraCBV, sl->GetShadowCB().GpuAddress());

			for (ModelRenderer* mRenderer : m_renderers)
			{
				auto batch = mRenderer->GetAllTransparent();

				for (auto& mesh : batch)
				{
					context.m_commandList->Get()->SetGraphicsRootDescriptorTable((UINT)RootSignatureSlot::MaterialTextureSRV, mesh->GetMaterialTextureSRV());
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)RootSignatureSlot::ObjectCBV, mesh->GetObjectCB().GpuAddress());
					mesh->DrawMesh(context);
				}
			}

			sl->GetShadowBuffer().RenderShadowEnd(context);
		}


		PIXEndEvent(context.m_commandList->Get());
	}



	void MainRenderPass(DX12Lib::GraphicsContext& context)
	{
		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(255, 255, 255), L"MainRenderPass");

		context.SetViewportAndScissor(s_screenViewport, s_scissorRect);

		context.SetPipelineState(s_PSOs[PSO_PBR_OPAQUE].get());

		D3D12_CPU_DESCRIPTOR_HANDLE rtvs[(UINT)RenderTargetType::Count];
		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			context.TransitionResource(*s_renderTargets[i], D3D12_RESOURCE_STATE_RENDER_TARGET);
			context.ClearColor(*s_renderTargets[i], s_renderTargets[i]->GetClearColor().GetPtr(), nullptr);
			rtvs[i] = s_renderTargets[i]->GetRTV();
		}

		context.ClearDepthAndStencil(*s_depthStencilBuffer);



		context.SetRenderTargets(
			(UINT)RenderTargetType::Count,
			rtvs,
			s_depthStencilBuffer->GetDSV());



		// Main pass opaque
		for (auto& pso : s_PSOs)
		{
			context.SetPipelineState(pso.second.get());

			if (pso.first == PSO_PHONG_OPAQUE || pso.first == PSO_PBR_OPAQUE)
			{
				context.SetPipelineState(pso.second.get());

				context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
					(UINT)Renderer::RootSignatureSlot::CommonCBV, s_graphicsMemory->AllocateConstant(m_costantBufferCommons).GpuAddress());


				if (m_mainCamera != nullptr && m_mainCamera->IsEnabled)
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
						(UINT)Renderer::RootSignatureSlot::CameraCBV, m_mainCamera->GetCameraBuffer().GpuAddress());

				context.FlushResourceBarriers();
			}

			for (ModelRenderer* mRenderer : m_renderers)
			{
				auto batch = mRenderer->GetAllOpaqueForPSO(pso.first);
				
				mRenderer->Model->UseBuffers(context);

				for (auto& mesh : batch)
				{
					context.m_commandList->Get()->SetGraphicsRootDescriptorTable((UINT)RootSignatureSlot::MaterialTextureSRV, mesh->GetMaterialTextureSRV());
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)RootSignatureSlot::ObjectCBV, mesh->GetObjectCB().GpuAddress());
					mesh->DrawMesh(context);
				}
			}
		}

		// Main pass transparent
		for (auto& pso : s_PSOs)
		{
			context.SetPipelineState(pso.second.get());

			if (pso.first == PSO_PHONG_ALPHA_TEST || pso.first == PSO_PBR_ALPHA_TEST)
			{

				context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
					(UINT)Renderer::RootSignatureSlot::CommonCBV, s_graphicsMemory->AllocateConstant(m_costantBufferCommons).GpuAddress());


				if (m_mainCamera != nullptr && m_mainCamera->IsEnabled)
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
						(UINT)Renderer::RootSignatureSlot::CameraCBV, m_mainCamera->GetCameraBuffer().GpuAddress());

				context.FlushResourceBarriers();
			}

			for (ModelRenderer* mRenderer : m_renderers)
			{
				auto batch = mRenderer->GetAllTransparentForPSO(pso.first);

				mRenderer->Model->UseBuffers(context);

				for (auto& mesh : batch)
				{
					context.m_commandList->Get()->SetGraphicsRootDescriptorTable((UINT)RootSignatureSlot::MaterialTextureSRV, mesh->GetMaterialTextureSRV());
					context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)RootSignatureSlot::ObjectCBV, mesh->GetObjectCB().GpuAddress());
					mesh->DrawMesh(context);
				}
			}
		}

		PIXEndEvent(context.m_commandList->Get());
	}

	void DeferredPass(GraphicsContext& context)
	{
		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"DeferredPass");

		auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

		context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
		context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

		context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

		context.SetPipelineState(s_PSOs[L"deferredPso"].get());

		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			context.TransitionResource(*s_renderTargets[i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}

		if (auto techPtr = m_rtgiData.lock())
		{
			auto& voxelSceneBufferManager = techPtr->GetBufferManager(L"VoxelizeScene");
			auto& prefixBufferManager = techPtr->GetBufferManager(L"PrefixSumVoxels");
			auto& clusterBufferManager = techPtr->GetBufferManager(L"ClusterVoxels");
			auto& gaussianBufferManager = techPtr->GetBufferManager(L"GaussianFilterRead");

			voxelSceneBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			prefixBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			gaussianBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)DeferredRootSignatureSlot::VoxelRTGICBV, s_graphicsMemory->AllocateConstant(m_cbVoxelCommons).GpuAddress());

			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::VoxelBufferSRV, voxelSceneBufferManager.GetSRVHandle());

			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::CompactBufferSRV, prefixBufferManager.GetSRVHandle());

			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::ClusterBufferSRV, clusterBufferManager.GetSRVHandle());

			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::RadianceBufferSRV, gaussianBufferManager.GetSRVHandle());
		}

		context.FlushResourceBarriers();


		m_costantBufferCommons.useRTGI = (UINT)m_useRTGI;

		context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::DeferredRootSignatureSlot::CommonCBV, s_graphicsMemory->AllocateConstant(m_costantBufferCommons).GpuAddress());

		if (m_mainCamera != nullptr && m_mainCamera->IsEnabled)
			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)Renderer::DeferredRootSignatureSlot::CameraCBV, m_mainCamera->GetCameraBuffer().GpuAddress());
		
		context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
			(UINT)DeferredRootSignatureSlot::LightSRV, LightComponent::GetLightBufferSRV().GpuAddress());

		context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
			(UINT)DeferredRootSignatureSlot::MaterialSRV, s_materialManager->GetMaterialStructuredBuffer().GpuAddress()
		);

		if (m_shadowLights.size() > 0)
		{
			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::CommonTextureSRV, m_commonTextureSRVHandle);
		}

		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			(UINT)DeferredRootSignatureSlot::GBufferSRV, m_gbufferStartHandle);


		DrawScreenQuad(context);

		PIXEndEvent(context.m_commandList->Get());
	}

	void RenderLayers(GraphicsContext& context)
	{ 

		ShadowPass(context);

		if (sEnableRenderMainPass)
			MainRenderPass(context);

		DeferredPass(context);
	}

	void PostDrawCleanup(CommandContext& context)
	{
		m_renderers.clear();
		m_shadowLights.clear();
		m_mainCamera = nullptr;
	}

	void DrawScreenQuad(DX12Lib::GraphicsContext& context)
	{
		context.m_commandList->Get()->IASetVertexBuffers(0, 1, &m_vertexBufferView);
		context.m_commandList->Get()->IASetIndexBuffer(&m_indexBufferView);

		context.m_commandList->Get()->DrawIndexedInstanced(6, 1, 0, 0, 0);
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

		GraphicsContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE_DIRECT);
		s_swapchain->CurrentBufferIndex = (s_swapchain->CurrentBufferIndex + 1) % s_swapchain->BufferCount;
	}

	void SetScissorAndViewportSize(int width, int height)
	{
		Renderer::s_screenViewport.TopLeftX = 0;
		Renderer::s_screenViewport.TopLeftY = 0;
		Renderer::s_screenViewport.Width = static_cast<float>(width);
		Renderer::s_screenViewport.Height = static_cast<float>(height);
		Renderer::s_screenViewport.MinDepth = 0.0f;
		Renderer::s_screenViewport.MaxDepth = 1.0f;

		Renderer::s_scissorRect = { 0, 0, width, height};
	}

	void UseRTGI(bool useRTGI)
	{
		m_useRTGI = useRTGI;
	}

	DirectX::XMMATRIX DebugVoxelToWorldMatrix(ConstantBufferVoxelCommons cb, DirectX::XMFLOAT3 originalSceneMin, DirectX::XMFLOAT3 originalSceneMax)
	{
		DirectX::XMFLOAT3 SceneAABBMax = cb.SceneAABBMax;
		DirectX::XMFLOAT3 SceneAABBMin = cb.SceneAABBMin;
		DirectX::XMUINT3 VoxelGridSize = cb.voxelTextureDimensions;

		DirectX::XMFLOAT3 extents = DirectX::XMFLOAT3(SceneAABBMax.x - SceneAABBMin.x,
			SceneAABBMax.y - SceneAABBMin.y,
			SceneAABBMax.z - SceneAABBMin.z);

		DirectX::XMFLOAT3 originalExtents = DirectX::XMFLOAT3(originalSceneMax.x - originalSceneMin.x,
			originalSceneMax.y - originalSceneMin.y,
			originalSceneMax.z - originalSceneMin.z);

		DirectX::XMMATRIX normalizeMatrix = DirectX::XMMatrixScaling(1.0f / VoxelGridSize.x,
			1.0f / VoxelGridSize.y,
			1.0f / VoxelGridSize.z);



		DirectX::XMMATRIX scaleMatrix = DirectX::XMMatrixScaling(extents.x,
			extents.y,
			extents.z);

		DirectX::XMMATRIX scaleOriginalSceneBounds = DirectX::XMMatrixScaling(originalExtents.x / extents.x,
			originalExtents.y / extents.y,
			originalExtents.z / extents.z);

		DirectX::XMMATRIX translateMatrix = DirectX::XMMatrixTranslation(extents.x / VoxelGridSize.x * 0.5f,
			extents.y / VoxelGridSize.y * 0.5f,
			extents.z / VoxelGridSize.z * 0.5f);


		DirectX::XMMATRIX translateToOrigin = DirectX::XMMatrixTranslation(
			SceneAABBMin.x,
			SceneAABBMin.y,
			SceneAABBMin.z);

		//DirectX::XMMATRIX voxelToWorld = normalizeMatrix * scaleMatrix * translateMatrix * translateToOrigin;

		DirectX::XMMATRIX voxelToWorld = normalizeMatrix * scaleMatrix * translateMatrix * translateToOrigin;

		return voxelToWorld;
	}

	void SetRTGIData(std::shared_ptr<CVGI::TechniqueData> techniqueData, DirectX::XMFLOAT3 originalSceneMin, DirectX::XMFLOAT3 originalSceneMax)
	{
		m_rtgiData = techniqueData;
		m_cbVoxelCommons = techniqueData->GetVoxelCommons();
	}

	void SwapShadowBuffers()
	{
		//std::swap(s_shadowBuffer, s_rtgiShadowBuffer);
		//std::swap(m_shadowTextureHandle, m_rtgiShadowTextureHandle);
	}

	void CreateDefaultShaders()
	{
		std::wstring srcDir = Utils::ToWstring(SOURCE_DIR);
		std::wstring baseVSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_VS.hlsl";
		std::wstring phongPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_PS.hlsl";
		std::wstring pbrPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\BasicPBR_PS.hlsl";
		std::wstring depthVSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Depth_VS.hlsl";
		std::wstring gbufferPBRpsFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\GBufferPBR_PS.hlsl";
		std::wstring deferredVSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Deferred_VS.hlsl";

		std::wstring alphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTest_PS.hlsl";
		std::wstring alphaTestPBRPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestPBR_PS.hlsl";
		std::wstring depthAlphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestDepth_PS.hlsl";

		std::shared_ptr<Shader> basicVS = std::make_shared<Shader>(baseVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> phongPS = std::make_shared<Shader>(phongPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> pbrPS = std::make_shared<Shader>(pbrPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> gbufferPBRPS = std::make_shared<Shader>(gbufferPBRpsFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> deferredVS = std::make_shared<Shader>(deferredVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> depthVS = std::make_shared<Shader>(depthVSFile, "VS", "vs_5_1");

		std::shared_ptr<Shader> phongAlphaTestPS = std::make_shared<Shader>(alphaTestPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> pbrAlphaTestPS = std::make_shared<Shader>(alphaTestPBRPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> depthAlphaTestPS = std::make_shared<Shader>(depthAlphaTestPSFile, "PS", "ps_5_1");

		basicVS->Compile();
		phongPS->Compile();
		pbrPS->Compile();
		gbufferPBRPS->Compile();
		deferredVS->Compile();

		depthVS->Compile();
		phongAlphaTestPS->Compile();
		pbrAlphaTestPS->Compile();
		depthAlphaTestPS->Compile();

		s_shaders[L"basicVS"] = std::move(basicVS);
		s_shaders[L"phongPS"] = std::move(phongPS);
		s_shaders[L"pbrPS"] = std::move(pbrPS);
		s_shaders[L"gbufferPBRPS"] = std::move(gbufferPBRPS);
		s_shaders[L"deferredVS"] = std::move(deferredVS);
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
		ShadowSamplerDesc.MaxAnisotropy = 1;
		ShadowSamplerDesc.SetBorderColor(Color::Black());
		
		

		std::shared_ptr<RootSignature> baseRootSignature = std::make_shared<RootSignature>((UINT)RootSignatureSlot::Count, 2);
		baseRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		baseRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
		(*baseRootSignature)[(UINT)RootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);
		(*baseRootSignature)[(UINT)RootSignatureSlot::CameraCBV].InitAsConstantBuffer(1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::ObjectCBV].InitAsConstantBuffer(2);
		(*baseRootSignature)[(UINT)RootSignatureSlot::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::MaterialSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*baseRootSignature)[(UINT)RootSignatureSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PHONG_TEXTURES);
		baseRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> pbrRootSignature = std::make_shared<RootSignature>((UINT)RootSignatureSlot::Count, 2);
		pbrRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		pbrRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::CameraCBV].InitAsConstantBuffer(1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::ObjectCBV].InitAsConstantBuffer(2);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::MaterialSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*pbrRootSignature)[(UINT)RootSignatureSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);

		pbrRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);


		std::shared_ptr<RootSignature> deferredRootSignature = std::make_shared<RootSignature>((UINT)DeferredRootSignatureSlot::Count, 2);
		deferredRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		deferredRootSignature->InitStaticSampler(1, ShadowSamplerDesc);
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);										
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::CameraCBV].InitAsConstantBuffer(1);										
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::VoxelRTGICBV].InitAsConstantBuffer(2);										
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);				
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::MaterialSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_ALL, 1);				
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::CommonTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2);	
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::GBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, (UINT)RenderTargetType::Count);	
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::VoxelBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 2);
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::CompactBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 3);
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::ClusterBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 4);
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::RadianceBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 5);
		deferredRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<GraphicsPipelineState> phongPso = std::make_shared<GraphicsPipelineState>();
		phongPso->InitializeDefaultStates();
		phongPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		phongPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		phongPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		phongPso->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		phongPso->SetShader(s_shaders[L"phongPS"], ShaderType::Pixel);
		phongPso->SetRootSignature(baseRootSignature);
		phongPso->Finalize();
		phongPso->Name = PSO_PHONG_OPAQUE;


		DXGI_FORMAT rtvFormats[(UINT)RenderTargetType::Count];

		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			rtvFormats[i] = s_renderTargets[i]->GetFormat();
		}

		// Duplicate content of opaquePSO
		std::shared_ptr<GraphicsPipelineState> pbrPso = std::make_shared<GraphicsPipelineState>();
		*pbrPso = *phongPso;
		pbrPso->SetShader(s_shaders[L"gbufferPBRPS"], ShaderType::Pixel);
		pbrPso->SetRenderTargetFormats((UINT)RenderTargetType::Count, rtvFormats, s_depthStencilBuffer->GetFormat());
		pbrPso->Finalize();
		pbrPso->Name = PSO_PBR_OPAQUE;

		std::shared_ptr<GraphicsPipelineState> deferredPso = std::make_shared<GraphicsPipelineState>();
		deferredPso->InitializeDefaultStates();
		deferredPso->SetInputLayout(DirectX::VertexPositionTexture::InputLayout.pInputElementDescs, \
					DirectX::VertexPositionTexture::InputLayout.NumElements);
		deferredPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		deferredPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		deferredPso->SetShader(s_shaders[L"deferredVS"], ShaderType::Vertex);
		deferredPso->SetShader(s_shaders[L"pbrPS"], ShaderType::Pixel);
		deferredPso->SetRootSignature(deferredRootSignature);
		deferredPso->Finalize();
		deferredPso->Name = L"deferredPso";


		std::shared_ptr<GraphicsPipelineState> phongAlphaTestPso = std::make_shared<GraphicsPipelineState>();
		*phongAlphaTestPso = *phongPso;
		phongAlphaTestPso->InitializeDefaultStates();
		phongAlphaTestPso->SetCullMode(D3D12_CULL_MODE_NONE);
		phongAlphaTestPso->SetShader(s_shaders[L"phongAlphaTestPS"], ShaderType::Pixel);
		phongAlphaTestPso->Finalize();
		phongAlphaTestPso->Name = PSO_PHONG_ALPHA_TEST;


		std::shared_ptr<GraphicsPipelineState> pbrAlphaTestPso = std::make_shared<GraphicsPipelineState>();
		*pbrAlphaTestPso = *phongAlphaTestPso;
		pbrAlphaTestPso->SetRenderTargetFormats((UINT)RenderTargetType::Count, rtvFormats, s_depthStencilBuffer->GetFormat());
		pbrAlphaTestPso->SetShader(s_shaders[L"pbrAlphaTestPS"], ShaderType::Pixel);
		pbrAlphaTestPso->Finalize();
		pbrAlphaTestPso->Name = PSO_PBR_ALPHA_TEST;

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

		std::shared_ptr<GraphicsPipelineState> shadowPso = std::make_shared<GraphicsPipelineState>();
		shadowPso->InitializeDefaultStates();
		shadowPso->SetRasterizerState(shadowRastDesc);
		shadowPso->SetBlendState(blendNoColorWrite);
		shadowPso->SetDepthStencilState(depthTestDesc);
		shadowPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
					DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		shadowPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		shadowPso->SetRenderTargetFormats(0, nullptr, DXGI_FORMAT_D16_UNORM);
		shadowPso->SetShader(s_shaders[L"depthVS"], ShaderType::Vertex);
		shadowPso->SetRootSignature(pbrRootSignature);
		shadowPso->Finalize();
		shadowPso->Name = PSO_SHADOW_OPAQUE;

		D3D12_RASTERIZER_DESC shadowAlphaTestRastDesc = shadowRastDesc;
		shadowAlphaTestRastDesc.CullMode = D3D12_CULL_MODE_NONE;
		std::shared_ptr<GraphicsPipelineState> shadowAlphaTested = std::make_shared<GraphicsPipelineState>();
		*shadowAlphaTested = *shadowPso;
		shadowAlphaTested->SetRasterizerState(shadowAlphaTestRastDesc);
		shadowAlphaTested->SetShader(s_shaders[L"depthAlphaTestPS"], ShaderType::Pixel);
		shadowAlphaTested->Finalize();
		shadowAlphaTested->Name = PSO_SHADOW_ALPHA_TEST;

		std::shared_ptr<GraphicsPipelineState> shadowTestPso = std::make_shared<GraphicsPipelineState>();
		shadowTestPso->InitializeDefaultStates();
		shadowTestPso->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
					DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		shadowTestPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		shadowTestPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		shadowTestPso->SetShader(s_shaders[L"depthVS"], ShaderType::Vertex);
		shadowTestPso->SetShader(s_shaders[L"depthAlphaTestPS"], ShaderType::Pixel);
		shadowTestPso->SetRootSignature(pbrRootSignature);
		shadowTestPso->Finalize();
		shadowTestPso->Name = PSO_SHADOW_TEST;
	


		s_PSOs[phongPso->Name] = std::move(phongPso);
		s_PSOs[pbrPso->Name] = std::move(pbrPso);

		s_PSOs[phongAlphaTestPso->Name] = std::move(phongAlphaTestPso);
		s_PSOs[pbrAlphaTestPso->Name] = std::move(pbrAlphaTestPso);

		s_PSOs[shadowPso->Name] = std::move(shadowPso);
		s_PSOs[shadowAlphaTested->Name] = std::move(shadowAlphaTested);

		s_PSOs[shadowTestPso->Name] = std::move(shadowTestPso);

		s_PSOs[deferredPso->Name] = std::move(deferredPso);
	}
}
