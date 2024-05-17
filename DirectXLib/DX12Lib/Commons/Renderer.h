#pragma once

namespace DX12Lib
{
	class CommandContext;
	class ModelRenderer;
	class PipelineState;
	class Shader;
}

namespace Graphics
{
	namespace Renderer
	{
		extern std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap;
		extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
		extern std::unique_ptr<DX12Lib::TextureManager> s_textureManager;
		extern std::unique_ptr<DX12Lib::MaterialManager> s_materialManager;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::PipelineState>> s_PSOs;
		extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::Shader>> s_shaders;

		void Initialize();
		void AddRendererToQueue(DX12Lib::ModelRenderer* renderer);
		void RenderLayers(DX12Lib::CommandContext* context);
		void Shutdown();
		
	};
}

