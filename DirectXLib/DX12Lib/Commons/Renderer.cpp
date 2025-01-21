#include "DX12Lib/pch.h"
#include "Renderer.h"
#include "DX12Lib/Models/ModelRenderer.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include <WinPixEventRuntime/pix3.h>

#include "imgui.h"
#include "backends/imgui_impl_dx12.h"

#include "UIHelpers.h"

#define USE_RTGI 1

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
		RTGISRV,
		LerpRadianceSRV,
		Count
	};

	enum class PostProcessRootSignatureSlot
	{
		CommonCBV = 0,
		CameraCBV = 1,
		PostProcessCBV = 2,
		GBufferSRV,
		DeferredResultSRV,
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

	struct ConstantBufferLerp
	{
		UINT32 CurrentPhase = 0;
		UINT32 FaceCount = 0;
		DirectX::XMUINT2 ScreenSize = DirectX::XMUINT2(480, 360);

		float accumulatedTime = 0.0f;
		float maxTime = 0.0f;
		float pad0 = 0.0f;
		float pad1 = 0.0f;

	} m_cbLerp;

	struct ConstantBufferPostProcess
	{
		float SigmaSpatial = 20.0f; // Controls spatial smoothing (e.g., pixel distance)
		float SigmaIntensity = 12.0f; // Controls intensity smoothing (e.g., color difference)
		float MaxWorldPostDistance = 2.0f; // Maximum allowed difference in world coordinates
		int KernelSize = 3;
	} m_cbPostProcess;

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

	DX12Lib::DescriptorHandle m_commonTextureSRVHandle;

	int s_clientWidth = 1920;
	int s_clientHeight = 1080;

	bool sEnableRenderMainPass = true;
	bool sEnableRenderShadows = true;

	std::unique_ptr<DX12Lib::ShadowBuffer> s_rtgiShadowBuffer = nullptr;
	DX12Lib::ColorBuffer* s_voxel3DTexture = nullptr;

	std::vector<std::unique_ptr<DX12Lib::ColorBuffer>> s_renderTargets;

	std::vector<std::unique_ptr<DX12Lib::ColorBuffer>> s_defferedOutputRenderTargets;

	UINT64 backBufferFences[3] = { 0, 0, 0 };

	// Ping ponging shadow textures for RTGI
	DX12Lib::DescriptorHandle m_rtgiHandleSRV;

	DX12Lib::DescriptorHandle m_postProcessHandleUAV;
	DX12Lib::DescriptorHandle m_postProcessHandleSRV;


	// Post process buffers (used for lerp in radiance);
	DX12Lib::StructuredBuffer m_postProcessBuffers[5];

	DX12Lib::DescriptorHandle m_gbufferStartHandle;
	DX12Lib::DescriptorHandle m_deferredRtStartHandleSRV;

	CostantBufferCommons m_costantBufferCommons;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;

	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

	bool m_useRTGI = false;

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

		s_defferedOutputRenderTargets.resize(2);

		s_defferedOutputRenderTargets[0] = std::make_unique<ColorBuffer>();
		s_defferedOutputRenderTargets[0]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

		s_defferedOutputRenderTargets[1] = std::make_unique<ColorBuffer>();
		s_defferedOutputRenderTargets[1]->Create2D(s_clientWidth, s_clientHeight, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

		s_rtgiShadowBuffer = std::make_unique<ShadowBuffer>();
		s_rtgiShadowBuffer->Create(2048, 2048);

		m_commonTextureSRVHandle = s_textureHeap->Alloc(2);
		m_rtgiHandleSRV = s_textureHeap->Alloc(6);

		m_postProcessHandleSRV = s_textureHeap->Alloc(5);
		m_postProcessHandleUAV = s_textureHeap->Alloc(5);

		m_gbufferStartHandle = s_textureHeap->Alloc((UINT)RenderTargetType::Count);
		UINT gbufferStartIndex = s_textureHeap->GetOffsetOfHandle(m_gbufferStartHandle);

		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			s_device->Get()->CopyDescriptorsSimple(1, 
				(*s_textureHeap)[gbufferStartIndex + i],
				s_renderTargets[i]->GetSRV(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		m_deferredRtStartHandleSRV = s_textureHeap->Alloc(2);


		size_t offset = s_textureHeap->GetDescriptorSize();

		for (UINT i = 0; i < 2; i++)
		{
			s_device->Get()->CopyDescriptorsSimple(1,
				m_deferredRtStartHandleSRV + offset * i,
				s_defferedOutputRenderTargets[i]->GetSRV(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		}

		CreateDefaultShaders();
		CreateDefaultPSOs();
		BuildRenderQuadGeometry();
		s_materialManager->LoadDefaultMaterials(*s_textureManager);

		m_cbVoxelCommons.VoxelCount = 0;
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

		m_cbLerp.ScreenSize = DirectX::XMUINT2(s_clientWidth, s_clientHeight);
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

	void SetPostProcessSpatialSigma(float sigma)
	{
		m_cbPostProcess.SigmaSpatial = sigma;
	}

	float GetPostProcessSpatialSigma()
	{
		return m_cbPostProcess.SigmaSpatial;
	}
	void SetPostProcessIntensitySigma(float sigma)
	{
		m_cbPostProcess.SigmaIntensity = sigma;
	}
	float GetPostProcessIntensitySigma()
	{
		return m_cbPostProcess.SigmaIntensity;
	}
	void SetPostProcessWorldThreshold(float treshold)
	{
		m_cbPostProcess.MaxWorldPostDistance = treshold;
	}
	float GetPostProcessWorldThreshold()
	{
		return m_cbPostProcess.MaxWorldPostDistance;
	}
	void SetPostProcessKernelSize(int kernelSize)
	{
		m_cbPostProcess.KernelSize = kernelSize;
	}
	int GetPostProcessKernelSize()
	{
		return m_cbPostProcess.KernelSize;
	}

	void LerpRadiancePass(DX12Lib::GraphicsContext& context)
	{
		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"LerpRadiancePass");

		context.SetPipelineState(s_PSOs[L"lerpRadiancePso"].get());

		for (UINT i = 0; i < 2; i++)
		{
			context.TransitionResource(m_postProcessBuffers[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		}

		context.TransitionResource(m_postProcessBuffers[2 + s_swapchain->CurrentBufferIndex], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

		context.FlushResourceBarriers();

		// Gaussian radiance data should already be in the NON_PIXEL_SHADER_RESOURCE state

		context.m_commandList->Get()->SetGraphicsRootConstantBufferView(0, s_graphicsMemory->AllocateConstant(m_cbLerp).GpuAddress());

		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(1, m_rtgiHandleSRV + s_textureHeap->GetDescriptorSize() * 5);

		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			2, m_postProcessHandleUAV);

		context.m_commandList->Get()->SetGraphicsRootUnorderedAccessView(
			3, m_postProcessBuffers[2 + s_swapchain->CurrentBufferIndex].GetGpuVirtualAddress());

		context.SetRenderTargets(0, nullptr, Renderer::s_depthStencilBuffer->GetDSV());

		DrawScreenQuad(context);

		if (m_cbLerp.CurrentPhase == 1)
			m_cbLerp.CurrentPhase = 0;
		
		PIXEndEvent(context.m_commandList->Get());
	}

	void SetUpRenderFrame(DX12Lib::CommandContext& context)
	{
		context.SetDescriptorHeap(s_textureHeap.get());

		// Using Phong as default PSO and root signature just in case
		context.SetPipelineState(s_PSOs[PSO_PHONG_OPAQUE].get());

		m_costantBufferCommons.totalTime = GameTime::GetTotalTime();
		m_costantBufferCommons.deltaTime = GameTime::GetDeltaTime();
		m_costantBufferCommons.numLights = LightComponent::GetLightCount();

		Commons::UIHelpers::StartFrame();
	}

	void ShadowPassForCamera(DX12Lib::GraphicsContext& context, DX12Lib::ShadowCamera* shadowCamera)
	{
		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(40, 40, 40), L"ShadowPass");
		// Shadow pass opaque objects
		auto shadowPso = s_PSOs[PSO_SHADOW_OPAQUE];

		context.SetPipelineState(shadowPso.get());

		shadowCamera->GetShadowBuffer().RenderShadowStart(context, true);

		context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
			(UINT)RootSignatureSlot::CameraCBV, shadowCamera->GetShadowCB().GpuAddress());

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

		shadowCamera->GetShadowBuffer().RenderShadowEnd(context);
		

		shadowPso = s_PSOs[PSO_SHADOW_ALPHA_TEST];

		// Shadow pass transparent objects

		context.SetPipelineState(shadowPso.get());

		shadowCamera->GetShadowBuffer().RenderShadowStart(context, false);

		context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
			(UINT)RootSignatureSlot::CameraCBV, shadowCamera->GetShadowCB().GpuAddress());

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

		shadowCamera->GetShadowBuffer().RenderShadowEnd(context);


		PIXEndEvent(context.m_commandList->Get());
	}

	void ShadowPass(DX12Lib::GraphicsContext& context)
	{
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

		for (UINT32 i = 0; i < m_shadowLights.size(); i++)
		{
			auto sl = m_shadowLights[i];
			ShadowPassForCamera(context, sl);
		}


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

		context.SetPipelineState(s_PSOs[L"deferredPso"].get());

		for (UINT i = 0; i < (UINT)RenderTargetType::Count; i++)
		{
			context.TransitionResource(*s_renderTargets[i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}

		if (m_useRTGI)
		{
			context.TransitionResource(m_postProcessBuffers[2 + s_swapchain->CurrentBufferIndex], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)DeferredRootSignatureSlot::VoxelRTGICBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons).GpuAddress());

			context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
				(UINT)DeferredRootSignatureSlot::RTGISRV, m_rtgiHandleSRV);

			context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
				(UINT)DeferredRootSignatureSlot::LerpRadianceSRV, m_postProcessBuffers[2 + s_swapchain->CurrentBufferIndex].GetGpuVirtualAddress());

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


		D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2];
		for (UINT i = 0; i < 2; i++)
		{
			context.TransitionResource(*s_defferedOutputRenderTargets[i], D3D12_RESOURCE_STATE_RENDER_TARGET);
			context.ClearColor(*s_defferedOutputRenderTargets[i], s_defferedOutputRenderTargets[i]->GetClearColor().GetPtr(), nullptr);
			rtvs[i] = s_defferedOutputRenderTargets[i]->GetRTV();
		}

		context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

		context.SetRenderTargets(2, rtvs, Renderer::s_depthStencilBuffer->GetDSV());

		DrawScreenQuad(context);

		PIXEndEvent(context.m_commandList->Get());
	}

	void PostProcessPass(DX12Lib::GraphicsContext& context)
	{
		PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(50, 128, 90), L"PostProcessPass");

		auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

		context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		context.SetPipelineState(s_PSOs[L"postProcessPso"].get());

		for (UINT i = 0; i < 2; i++)
		{
			context.TransitionResource(*s_defferedOutputRenderTargets[i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		}

		context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET);

		context.FlushResourceBarriers();


		context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::PostProcessRootSignatureSlot::CommonCBV, s_graphicsMemory->AllocateConstant(m_costantBufferCommons).GpuAddress());

		if (m_mainCamera != nullptr && m_mainCamera->IsEnabled)
			context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
				(UINT)Renderer::PostProcessRootSignatureSlot::CameraCBV, m_mainCamera->GetCameraBuffer().GpuAddress());

		context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::PostProcessRootSignatureSlot::PostProcessCBV, s_graphicsMemory->AllocateConstant(m_cbPostProcess).GpuAddress());

		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			(UINT)PostProcessRootSignatureSlot::GBufferSRV, m_gbufferStartHandle);


		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			(UINT)PostProcessRootSignatureSlot::DeferredResultSRV, m_deferredRtStartHandleSRV);

		context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
		context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

		context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

		DrawScreenQuad(context);

		PIXEndEvent(context.m_commandList->Get());
	}

	void UIPass(DX12Lib::GraphicsContext& context, bool clearBackBuffer)
	{
		auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

		context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		if (clearBackBuffer)
		{
			auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

			context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
			context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);
			context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());
		}

		ImGui::Render();


		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), context.m_commandList->Get());
	}

	void RenderLayers(GraphicsContext& context)
	{ 

		ShadowPass(context);

		if (sEnableRenderMainPass)
			MainRenderPass(context);

		if (m_useRTGI)
			LerpRadiancePass(context);

		DeferredPass(context);
		PostProcessPass(context);
		UIPass(context);
	}

	void PostDrawCleanup(CommandContext& context)
	{
		ImGui::EndFrame();
		m_renderers.clear();
		m_shadowLights.clear();
		m_mainCamera = nullptr;
	}

	void DrawScreenQuad(DX12Lib::GraphicsContext& context)
	{
		context.m_commandList->Get()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
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


		HRESULT hr = s_swapchain->GetComPointer()->Present(1, 0);

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

	void SetRTGIData(ConstantBufferVoxelCommons voxelCommons)
	{
		UINT32 prevVoxelCount = m_cbVoxelCommons.VoxelCount;

		m_cbVoxelCommons = voxelCommons;

		if (prevVoxelCount != voxelCommons.VoxelCount)
		{
			for (UINT i = 0; i < 5; i++)
			{
				m_postProcessBuffers[i].Create(voxelCommons.VoxelCount * 6, sizeof(DirectX::XMUINT2));

				if (prevVoxelCount == 0)
				{
					size_t offset = s_textureHeap->GetDescriptorSize() * i;

					s_device->Get()->CopyDescriptorsSimple(1, m_postProcessHandleSRV + offset, m_postProcessBuffers[i].GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
					s_device->Get()->CopyDescriptorsSimple(1, m_postProcessHandleUAV + offset, m_postProcessBuffers[i].GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				}
			}

			m_cbLerp.FaceCount = voxelCommons.VoxelCount * 6;
		}
	}

	DX12Lib::DescriptorHandle& GetRTGIHandleSRV()
	{
		return m_rtgiHandleSRV;
	}

	void ResetLerpTime()
	{
		m_cbLerp.CurrentPhase = 1;
	}

	void SetDeltaLerpTime(float delta)
	{
		m_cbLerp.accumulatedTime = delta;
	}

	void SetLerpMaxTime(float maxTime)
	{
		m_cbLerp.maxTime = maxTime;
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
		std::wstring postProcessPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\PostProcess_PS.hlsl";
		std::wstring lerpRadiancePSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\LerpRadiance_PS.hlsl";

		std::wstring alphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTest_PS.hlsl";
		std::wstring alphaTestPBRPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestPBR_PS.hlsl";
		std::wstring depthAlphaTestPSFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\AlphaTestDepth_PS.hlsl";

		std::shared_ptr<Shader> basicVS = std::make_shared<Shader>(baseVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> phongPS = std::make_shared<Shader>(phongPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> pbrPS = std::make_shared<Shader>(pbrPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> gbufferPBRPS = std::make_shared<Shader>(gbufferPBRpsFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> deferredVS = std::make_shared<Shader>(deferredVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> depthVS = std::make_shared<Shader>(depthVSFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> postProcessPS = std::make_shared<Shader>(postProcessPSFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> lerpRadiancePS = std::make_shared<Shader>(lerpRadiancePSFile, "PS", "ps_5_1");


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
		postProcessPS->Compile();
		lerpRadiancePS->Compile();

		s_shaders[L"basicVS"] = std::move(basicVS);
		s_shaders[L"phongPS"] = std::move(phongPS);
		s_shaders[L"pbrPS"] = std::move(pbrPS);
		s_shaders[L"gbufferPBRPS"] = std::move(gbufferPBRPS);
		s_shaders[L"deferredVS"] = std::move(deferredVS);
		s_shaders[L"depthVS"] = std::move(depthVS);
		s_shaders[L"phongAlphaTestPS"] = std::move(phongAlphaTestPS);
		s_shaders[L"pbrAlphaTestPS"] = std::move(pbrAlphaTestPS);
		s_shaders[L"depthAlphaTestPS"] = std::move(depthAlphaTestPS);
		s_shaders[L"postProcessPS"] = std::move(postProcessPS);
		s_shaders[L"lerpRadiancePS"] = std::move(lerpRadiancePS);
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
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::RTGISRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 2);
		(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::LerpRadianceSRV].InitAsBufferSRV(5, D3D12_SHADER_VISIBILITY_ALL, 2);
		//(*deferredRootSignature)[(UINT)DeferredRootSignatureSlot::LerpRadianceSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 1, D3D12_SHADER_VISIBILITY_ALL, 2);
		deferredRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> postProcessRootSignature = std::make_shared<RootSignature>((UINT)PostProcessRootSignatureSlot::Count, 1);
		postProcessRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		(*postProcessRootSignature)[(UINT)PostProcessRootSignatureSlot::CommonCBV].InitAsConstantBuffer(0);
		(*postProcessRootSignature)[(UINT)PostProcessRootSignatureSlot::CameraCBV].InitAsConstantBuffer(1);
		(*postProcessRootSignature)[(UINT)PostProcessRootSignatureSlot::PostProcessCBV].InitAsConstantBuffer(2);
		(*postProcessRootSignature)[(UINT)PostProcessRootSignatureSlot::GBufferSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)RenderTargetType::Count);
		(*postProcessRootSignature)[(UINT)PostProcessRootSignatureSlot::DeferredResultSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, (UINT)RenderTargetType::Count, 2);
		postProcessRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> lerpRadianceRootSignature = std::make_shared<RootSignature>(4, 0);
		(*lerpRadianceRootSignature)[0].InitAsConstantBuffer(0);
		(*lerpRadianceRootSignature)[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
		(*lerpRadianceRootSignature)[2].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2);
		(*lerpRadianceRootSignature)[3].InitAsBufferUAV(2);
		lerpRadianceRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

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

		DXGI_FORMAT outDeferredRtvFormats[2] = { DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM };

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
		deferredPso->SetRenderTargetFormats(2, outDeferredRtvFormats, s_depthStencilBuffer->GetFormat());
		deferredPso->SetShader(s_shaders[L"deferredVS"], ShaderType::Vertex);
		deferredPso->SetShader(s_shaders[L"pbrPS"], ShaderType::Pixel);
		deferredPso->SetRootSignature(deferredRootSignature);
		deferredPso->Finalize();
		deferredPso->Name = L"deferredPso";

		std::shared_ptr<GraphicsPipelineState> postProcessPso = std::make_shared<GraphicsPipelineState>();
		postProcessPso->InitializeDefaultStates();
		postProcessPso->SetInputLayout(DirectX::VertexPositionTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionTexture::InputLayout.NumElements);
		postProcessPso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		postProcessPso->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		postProcessPso->SetShader(s_shaders[L"deferredVS"], ShaderType::Vertex);
		postProcessPso->SetShader(s_shaders[L"postProcessPS"], ShaderType::Pixel);
		postProcessPso->SetRootSignature(postProcessRootSignature);
		postProcessPso->Finalize();
		postProcessPso->Name = L"postProcessPso";

		std::shared_ptr<GraphicsPipelineState> lerpRadiancePso = std::make_shared<GraphicsPipelineState>();
		lerpRadiancePso->InitializeDefaultStates();
		lerpRadiancePso->SetInputLayout(DirectX::VertexPositionTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionTexture::InputLayout.NumElements);
		lerpRadiancePso->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		lerpRadiancePso->SetRenderTargetFormats(0, nullptr, m_depthStencilFormat);
		lerpRadiancePso->SetShader(s_shaders[L"deferredVS"], ShaderType::Vertex);
		lerpRadiancePso->SetShader(s_shaders[L"lerpRadiancePS"], ShaderType::Pixel);
		lerpRadiancePso->SetRootSignature(lerpRadianceRootSignature);
		lerpRadiancePso->Finalize();
		lerpRadiancePso->Name = L"lerpRadiancePso";

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
		s_PSOs[postProcessPso->Name] = std::move(postProcessPso);
		s_PSOs[lerpRadiancePso->Name] = std::move(lerpRadiancePso);
	}
}
