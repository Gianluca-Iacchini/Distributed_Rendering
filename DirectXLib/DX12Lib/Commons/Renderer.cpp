#include "DX12Lib/pch.h"
#include "Renderer.h"
#include "DX12Lib/Models/ModelRenderer.h"

using namespace DX12Lib;
using namespace Graphics;

void Graphics::Renderer::AddRendererToQueue(ModelRenderer* renderer)
{
	m_renderers.push_back(renderer);
}

void Graphics::Renderer::Render(CommandContext* context)
{
	for (auto& pso : Graphics::s_PSOs)
	{
		context->m_commandList->SetPipelineState(pso.second);
		
		for (ModelRenderer* mRenderer : m_renderers)
		{
			mRenderer->DrawMeshes(context->m_commandList->Get(), pso.first);
		}
	}

	m_renderers.clear();
}
