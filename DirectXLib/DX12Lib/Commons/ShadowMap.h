#pragma once
#include "DX12Lib/DXWrapper/DepthBuffer.h"
#include "DX12Lib/Commons/Camera.h"
#include "DX12Lib/Scene/Component.h"

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

	class ShadowCamera : private Camera, public Component
	{
		void UpdateViewMatrix(DirectX::XMFLOAT3 lightDir, DirectX::XMFLOAT3 shadowCenter, DirectX::XMFLOAT3 shadowBounds,
							  UINT bufferWidth, UINT bufferHeight, UINT bufferPrecision);

		void Update(CommandContext& context) override;

	private:
		DirectX::XMFLOAT3 m_lastCameraPos = { 0.0f, 0.0f, 0.0f };
	};
}

