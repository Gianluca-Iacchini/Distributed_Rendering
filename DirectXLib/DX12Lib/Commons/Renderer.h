#pragma once

#define PSO_PHONG_OPAQUE L"phongOpaquePso"
#define PSO_PHONG_TRANSPARENT L"phongTransparentPso"
#define PSO_PHONG_ALPHA_TEST L"phongAlphaTestPso"
#define PSO_PBR_OPAQUE L"pbrOpaquePso"
#define PSO_PBR_TRANSPARENT L"pbrTransparentPso"
#define PSO_PBR_ALPHA_TEST L"pbrAlphaTestPso"

#include "../VoxelUtils/Shaders/TechniquesCompat.h"

namespace DX12Lib
{
	class CommandContext;
	class GraphicsContext;
	class ModelRenderer;
	class LightComponent;
	class PipelineState;
	class Shader;
	class Swapchain;
	class DepthBuffer;
	class DX12Window;
	class ColorBuffer;
	class SceneCamera;
	class ShadowCamera;
	class ShadowBuffer;
	class ColorBuffer;
	class DescriptorHandle;
	class QueryHeap;

	class MaterialManager;
	class TextureManager;
}

namespace CVGI
{
	class TechniqueData;
}


namespace Graphics
{
	namespace Renderer
	{
		enum class RootSignatureSlot
		{
			CommonCBV = 0,
			CameraCBV = 1,
			ObjectCBV = 2,
			LightSRV,
			MaterialSRV,
			MaterialTextureSRV,
			Count
		};

		extern D3D12_VIEWPORT s_screenViewport;
		extern D3D12_RECT s_scissorRect;

		extern std::unique_ptr<DX12Lib::Swapchain> s_swapchain;
		extern std::unique_ptr<DX12Lib::DepthBuffer> s_depthStencilBuffer;
		extern std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap;
		extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
		extern std::unique_ptr<DX12Lib::TextureManager> s_textureManager;
		extern std::unique_ptr<DX12Lib::MaterialManager> s_materialManager;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::PipelineState>> s_PSOs;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::Shader>> s_shaders;

		extern int s_clientWidth;
		extern int s_clientHeight;

		extern bool sEnableRenderMainPass;
		extern bool sEnableRenderShadows;

		void InitializeApp();
		void AddRendererToQueue(DX12Lib::ModelRenderer* renderer);
		void AddShadowCamera(DX12Lib::ShadowCamera* shadowCamera);
		void AddMainCamera(DX12Lib::SceneCamera* camera);

		DX12Lib::ShadowBuffer* const GetShadowBuffer();
		DX12Lib::DescriptorHandle& GetShadowMapSrv();

		std::vector<DX12Lib::ModelRenderer*> GetRenderers();

		void SetPostProcessSpatialSigma(float sigma);
		float GetPostProcessSpatialSigma();
		void SetPostProcessIntensitySigma(float sigma);
		float GetPostProcessIntensitySigma();
		void SetPostProcessWorldThreshold(float treshold);
		float GetPostProcessWorldThreshold();
		void SetPostProcessKernelSize(int kernelSize);
		int GetPostProcessKernelSize();

		void SetUpRenderFrame(DX12Lib::CommandContext& context);
		void ShadowPassForCamera(DX12Lib::GraphicsContext& context, DX12Lib::ShadowCamera* shadowCamera);
		void ShadowPass(DX12Lib::GraphicsContext& context);
		void MainRenderPass(DX12Lib::GraphicsContext& context);
		void DeferredPass(DX12Lib::GraphicsContext& context);
		void LerpRadiancePass(DX12Lib::GraphicsContext& context);
		void PostProcessPass(DX12Lib::GraphicsContext& context);
		void UIPass(DX12Lib::GraphicsContext& context, bool clearBackBuffer = false);
		void RenderLayers(DX12Lib::GraphicsContext& context);
		void PostDrawCleanup(DX12Lib::CommandContext& context);
		void DrawScreenQuad(DX12Lib::GraphicsContext& context);

		void Shutdown();
		void InitializeSwapchain(DX12Lib::DX12Window* window);
		void WaitForSwapchainBuffers();
		DX12Lib::ColorBuffer& GetCurrentBackBuffer();
		void ResizeSwapchain(DX12Lib::CommandContext* context, int newWidth, int newHeight);
		void Present(UINT64 fenceVal);
		void SetScissorAndViewportSize(int width, int height);


		void SetRTGIData(ConstantBufferVoxelCommons voxelCommonResource);
		void UseRTGI(bool useRTGI);
		DX12Lib::DescriptorHandle& GetRTGIHandleSRV();

		void ResetLerpTime();
		void SetDeltaLerpTime(float delta);
		void SetLerpMaxTime(float maxTime);
		
	};
}

