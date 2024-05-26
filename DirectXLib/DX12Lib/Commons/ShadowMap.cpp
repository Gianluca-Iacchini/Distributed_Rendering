#include "DX12Lib/pch.h"
#include "ShadowMap.h"
#include "DX12Lib/Scene/SceneNode.h"
#include "DX12Lib/Commons/ShadowMap.h"

using namespace DX12Lib;
using namespace Graphics;

void DX12Lib::ShadowBuffer::Create(uint32_t width, uint32_t height)
{
	DepthBuffer::Create(width, height, DXGI_FORMAT_D16_UNORM);

	m_bufferViewport.TopLeftX = 0.0f;
	m_bufferViewport.TopLeftY = 0.0f;
	m_bufferViewport.Width = static_cast<float>(width);
	m_bufferViewport.Height = static_cast<float>(height);
	m_bufferViewport.MinDepth = 0.0f;
	m_bufferViewport.MaxDepth = 1.0f;

	m_bufferScissorRect = { 1, 1, static_cast<LONG>(width) - 2, static_cast<LONG>(height) - 2 };
}

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


void DX12Lib::ShadowCamera::UpdateShadowMatrix(SceneNode& sceneNode)
{
	auto& transform = sceneNode.Transform;

	Camera::UpdateViewMatrix(transform.GetWorldPosition(), transform.GetUp(), transform.GetForward(), transform.GetRight());

	DirectX::XMVECTOR cameraWS = transform.GetWorldPosition();


	DirectX::XMMATRIX view = this->GetView();

	Camera::SetOrthogonalBounds(m_shadowCenter, m_shadowBounds);


	DirectX::XMMATRIX projection = GetProjection();

	DirectX::XMMATRIX T(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f
	);

	DirectX::XMMATRIX S = view * projection * T;
	DirectX::XMStoreFloat4x4(&m_shadowTransform, DirectX::XMMatrixTranspose(S));

	DirectX::XMMATRIX viewProjection = view * projection;
	DirectX::XMMATRIX invView = DirectX::XMMatrixInverse(nullptr, view);
	DirectX::XMMATRIX invProjection = DirectX::XMMatrixInverse(nullptr, projection);
	DirectX::XMMATRIX invViewProjection = DirectX::XMMatrixInverse(nullptr, viewProjection);

	DirectX::XMStoreFloat4x4(&m_shadowCB.view, DirectX::XMMatrixTranspose(view));
	DirectX::XMStoreFloat4x4(&m_shadowCB.invView, DirectX::XMMatrixTranspose(invView));
	DirectX::XMStoreFloat4x4(&m_shadowCB.projection, DirectX::XMMatrixTranspose(projection));
	DirectX::XMStoreFloat4x4(&m_shadowCB.invProjection, DirectX::XMMatrixTranspose(invProjection));
	DirectX::XMStoreFloat4x4(&m_shadowCB.viewProjection, DirectX::XMMatrixTranspose(viewProjection));
	DirectX::XMStoreFloat4x4(&m_shadowCB.invViewProjection, DirectX::XMMatrixTranspose(invViewProjection));
	DirectX::XMStoreFloat3(&m_shadowCB.eyePosition, cameraWS);

	m_shadowCB.nearPlane = m_nearZ;
	m_shadowCB.farPlane = m_farZ;
}

DirectX::GraphicsResource DX12Lib::ShadowCamera::GetShadowCB()
{
	return Renderer::s_graphicsMemory->AllocateConstant(m_shadowCB);
}
