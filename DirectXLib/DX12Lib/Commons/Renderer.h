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
	class ModelRenderer;
	class PipelineState;
	class Shader;
	class Swapchain;
	class DepthBuffer;
	class DX12Window;
	class ColorBuffer;
}



namespace Graphics
{
	namespace Renderer
	{
		enum class RootSignatureSlot
		{
			CommonCBV = 0,
			ObjectCBV = 1,
			CameraCBV = 2,
			LightSRV = 3,
			MaterialSRV = 4,
			TextureSRV = 5
		};

		extern std::unique_ptr<DX12Lib::Swapchain> s_swapchain;
		extern std::unique_ptr<DX12Lib::DepthBuffer> s_depthStencilBuffer;
		extern std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap;
		extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
		extern std::unique_ptr<DX12Lib::TextureManager> s_textureManager;
		extern std::unique_ptr<DX12Lib::MaterialManager> s_materialManager;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::PipelineState>> s_PSOs;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::Shader>> s_shaders;

		void Initialize();
		void AddRendererToQueue(DX12Lib::ModelRenderer* renderer);
		void SetUpRenderFrame(DX12Lib::CommandContext* context);
		void RenderLayers(DX12Lib::CommandContext* context);
		void Shutdown();
		void InitializeSwapchain(DX12Lib::DX12Window* window);
		void WaitForSwapchainBuffers();
		DX12Lib::ColorBuffer& GetCurrentBackBuffer();
		void Present(UINT64 fenceVal);
	};
}

