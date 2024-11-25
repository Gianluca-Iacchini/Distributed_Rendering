#pragma once
#include "DX12Lib/DXWrapper/DepthBuffer.h"
#include "DX12Lib/Commons/Camera.h"
#include "DX12Lib/Scene/SceneNode.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib
{
	class GraphicsContext;
}

namespace DX12Lib
{
	class ShadowBuffer : public DepthBuffer
	{
	public:
		void Create(uint32_t width, uint32_t height);
		void RenderShadowStart(GraphicsContext& context, bool clearDsv = true);
		void RenderShadowEnd(GraphicsContext& context);
	private:
		D3D12_VIEWPORT m_bufferViewport;
		D3D12_RECT m_bufferScissorRect;
	};

	class ShadowCamera : public Camera
	{
	public:
		ShadowCamera() : Camera() {}
		~ShadowCamera() = default;
		void SetShadowBufferDimensions(uint32_t width, uint32_t height);
		void SetCenter(const DirectX::XMFLOAT3& center) { m_shadowCenter = center; }
		void SetBounds(const DirectX::XMFLOAT3& bounds) { m_shadowBounds = bounds; }
		
		ShadowBuffer& GetShadowBuffer() { return m_shadowBuffer; }
		DirectX::XMFLOAT3 GetBounds() const { return m_shadowBounds; }

		const ConstantBufferCamera& GetShadowConstantBuffer() const { return m_shadowCB; }
		void UpdateShadowMatrix(SceneNode& node);
		void UpdateShadowMatrix(const ConstantBufferCamera& cb);



		DirectX::GraphicsResource GetShadowCB();
		DirectX::XMFLOAT4X4 GetShadowTransform() const { return m_shadowTransform; }
		DirectX::XMFLOAT4X4 GetInvShadowTransform() const { return m_invShadowTransform; }
	private:
		DirectX::XMFLOAT3 m_shadowCenter = { 0.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT3 m_shadowBounds = { 250.f, 250.f, 250.f };
		DirectX::XMFLOAT4X4 m_shadowTransform = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 m_invShadowTransform = MathHelper::Identity4x4();
		ConstantBufferCamera m_shadowCB;
		ShadowBuffer m_shadowBuffer;
	};
}

