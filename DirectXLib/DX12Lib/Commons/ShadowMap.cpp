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

void DX12Lib::ShadowBuffer::RenderShadowStart(GraphicsContext& context, bool clearDsv)
{
	context.TransitionResource(*this, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);

	if (clearDsv)
	{
		context.ClearDepth(*this);
	}

	context.SetDepthStencilTarget(GetDSV());
	context.SetViewportAndScissor(m_bufferViewport, m_bufferScissorRect);
}

void DX12Lib::ShadowBuffer::RenderShadowEnd(GraphicsContext& context)
{
	context.TransitionResource(*this, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, true);
}


void DX12Lib::ShadowCamera::SetShadowBufferDimensions(uint32_t width, uint32_t height)
{
	m_shadowBuffer.Create(width, height);
}

void DX12Lib::ShadowCamera::UpdateShadowMatrix(SceneNode& sceneNode)
{
	auto& transform = sceneNode.Transform;

	this->UpdateShadowMatrix(transform.GetWorldPosition3f(), transform.GetUp3f(), transform.GetForward3f(), transform.GetRight3f());
}

void DX12Lib::ShadowCamera::UpdateShadowMatrix(const ConstantBufferCamera& cb)
{
	m_shadowCB.view = cb.view;
	m_shadowCB.invView = cb.invView;
	m_shadowCB.projection = cb.projection;
	m_shadowCB.invProjection = cb.invProjection;
	m_shadowCB.viewProjection = cb.viewProjection;
	m_shadowCB.invViewProjection = cb.invViewProjection;

	DirectX::XMMATRIX view = DirectX::XMMatrixTranspose(DirectX::XMLoadFloat4x4(&cb.view));
	DirectX::XMMATRIX projection = DirectX::XMMatrixTranspose(DirectX::XMLoadFloat4x4(&cb.projection));

	DirectX::XMMATRIX T(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f
	);

	DirectX::XMMATRIX S = view * projection * T;
	DirectX::XMStoreFloat4x4(&m_shadowTransform, DirectX::XMMatrixTranspose(S));
	DirectX::XMStoreFloat4x4(&m_invShadowTransform, DirectX::XMMatrixTranspose(DirectX::XMMatrixInverse(nullptr, S)));

	m_shadowCB.eyePosition = DirectX::XMFLOAT3(
		-m_shadowCB.view._14,
		-m_shadowCB.view._24,
		-m_shadowCB.view._34
	);

	m_shadowCB.nearPlane = m_nearZ;
	m_shadowCB.farPlane = m_farZ;
}

void DX12Lib::ShadowCamera::UpdateShadowMatrix(Transform& transform)
{
	this->UpdateShadowMatrix(transform.GetWorldPosition3f(), transform.GetUp3f(), transform.GetForward3f(), transform.GetRight3f());
}

void DX12Lib::ShadowCamera::UpdateShadowMatrix(DirectX::XMFLOAT3 pos, DirectX::XMFLOAT3 up, DirectX::XMFLOAT3 forward, DirectX::XMFLOAT3 right)
{
	DirectX::XMVECTOR xmPos = DirectX::XMLoadFloat3(&pos);
	DirectX::XMVECTOR xmUp = DirectX::XMLoadFloat3(&up);
	DirectX::XMVECTOR xmForward = DirectX::XMLoadFloat3(&forward);
	DirectX::XMVECTOR xmRight = DirectX::XMLoadFloat3(&right);

	Camera::UpdateViewMatrix(xmPos, xmUp, xmForward, xmRight);

	DirectX::XMMATRIX view = this->GetView();

	DirectX::XMMATRIX projection = GetProjection();

	DirectX::XMMATRIX T(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.0f, 1.0f
	);

	DirectX::XMMATRIX S = view * projection * T;
	DirectX::XMStoreFloat4x4(&m_shadowTransform, DirectX::XMMatrixTranspose(S));
	DirectX::XMStoreFloat4x4(&m_invShadowTransform, DirectX::XMMatrixTranspose(DirectX::XMMatrixInverse(nullptr, S)));

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
	DirectX::XMStoreFloat3(&m_shadowCB.eyePosition, xmPos);

	m_shadowCB.nearPlane = m_nearZ;
	m_shadowCB.farPlane = m_farZ;
}

DirectX::GraphicsResource DX12Lib::ShadowCamera::GetShadowCB()
{
	return Renderer::s_graphicsMemory->AllocateConstant(m_shadowCB);
}
