#pragma once
#include "DX12Lib/DXWrapper/DepthBuffer.h"
#include "DX12Lib/Commons/Camera.h"

namespace DX12Lib
{
	class CommandContext;
}

namespace DX12Lib
{
	class ShadowBuffer : public DepthBuffer
	{
	public:
		void RenderShadowStart(CommandContext* context);
		void RenderShadowEnd(CommandContext* context);
	private:
		D3D12_VIEWPORT m_bufferViewport;
		D3D12_RECT m_bufferScissorRect;
	};

	class ShadowCamera : public Camera
	{

	};
}

