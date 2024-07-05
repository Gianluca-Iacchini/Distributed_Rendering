#pragma once

#define PSO_PHONG_OPAQUE L"phongOpaquePso"
#define PSO_PHONG_TRANSPARENT L"phongTransparentPso"
#define PSO_PHONG_ALPHA_TEST L"phongAlphaTestPso"
#define PSO_PBR_OPAQUE L"pbrOpaquePso"
#define PSO_PBR_TRANSPARENT L"pbrTransparentPso"
#define PSO_PBR_ALPHA_TEST L"pbrAlphaTestPso"

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
	class ShadowBuffer;
	class ColorBuffer;
	class DescriptorHandle;
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
			LightSRV = 3,
			MaterialSRV = 4,
			CommonTextureSRV = 5,
			MaterialTextureSRV = 6,
			VoxelTextureUAV = 7,
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

		extern std::unique_ptr<DX12Lib::ShadowBuffer> s_shadowBuffer;
		extern std::unique_ptr<DX12Lib::ColorBuffer> s_voxel3DTexture;

		extern int s_clientWidth;
		extern int s_clientHeight;

		extern bool sEnableRenderMainPass;
		extern bool sEnableRenderShadows;

		void InitializeApp();
		void AddRendererToQueue(DX12Lib::ModelRenderer* renderer);
		void AddLightToQueue(DX12Lib::LightComponent* light);
		void AddMainCamera(DX12Lib::SceneCamera* camera);
		void BindVoxelTexture(DX12Lib::CommandContext& context, UINT rootParamSlot);
		std::vector<DX12Lib::ModelRenderer*> GetRenderers();

		void SetUpRenderFrame(DX12Lib::GraphicsContext& context);
		void ShadowPass(DX12Lib::GraphicsContext& context);
		void MainRenderPass(DX12Lib::GraphicsContext& context);
		void DeferredPass(DX12Lib::GraphicsContext& context);
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

	};
}

