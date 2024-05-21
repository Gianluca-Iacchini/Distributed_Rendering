#include "DX12Lib/pch.h"
#include "ShadowMap.h"

using namespace DX12Lib;
using namespace Graphics;

void DX12Lib::ShadowBuffer::RenderShadowStart(CommandContext* context)
{
	assert(context != nullptr);

	context->TransitionResource(*this, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);
	context->ClearDepth(*this);
	context->SetDepthStencilTarget(GetDSV());
	context->SetViewportAndScissor(m_bufferViewport, m_bufferScissorRect);
}

void DX12Lib::ShadowBuffer::RenderShadowEnd(CommandContext* context)
{
	assert(context != nullptr);

	context->TransitionResource(*this, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, true);
}

void DX12Lib::ShadowCamera::Update(CommandContext& context)
{

}
