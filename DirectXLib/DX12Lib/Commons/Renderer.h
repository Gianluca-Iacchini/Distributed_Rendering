#pragma once

namespace DX12Lib
{
	class CommandContext;
	class ModelRenderer;
}

namespace Graphics
{
	class Renderer
	{
	public:
		Renderer() {}
		~Renderer() { m_renderers.clear(); }
		
		void AddRendererToQueue(DX12Lib::ModelRenderer* renderer);
		void Render(DX12Lib::CommandContext* context);
	private:
		std::vector<DX12Lib::ModelRenderer*> m_renderers;
	};
}

