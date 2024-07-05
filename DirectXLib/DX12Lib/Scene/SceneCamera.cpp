#include "DX12Lib/pch.h"
#include "SceneCamera.h"
#include "Mouse.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

void DX12Lib::SceneCamera::Init(CommandContext& context)
{
	if (m_isOrthographic)
	{
		Camera::SetOrthogonalBounds(m_orthogonalBounds.x, m_orthogonalBounds.y, m_orthogonalBounds.z, m_orthogonalBounds.w);
	}
	else
	{
		Camera::SetLens(0.25f * DirectX::XM_PI, ((float)Renderer::s_clientWidth / Renderer::s_clientHeight), 1.0f, 1000.0f);
	}


}

void DX12Lib::SceneCamera::Update(CommandContext& context)
{
	auto& transform = this->Node->Transform;

	if (Node->IsTransformDirty())
	{
		Camera::UpdateViewMatrix(transform.GetWorldPosition(), transform.GetUp(), transform.GetForward(), transform.GetRight());
	}

}

void DX12Lib::SceneCamera::Render(CommandContext& context)
{

}

void DX12Lib::SceneCamera::OnResize(CommandContext& context, int newWidth, int newHeight)
{
	if (m_isOrthographic)
	{
		Camera::SetOrthogonalBounds(m_orthogonalBounds.x, m_orthogonalBounds.y, m_orthogonalBounds.z, m_orthogonalBounds.w);
	}
	else
	{
		Camera::SetLens(0.25f * DirectX::XM_PI, ((float)newWidth / newHeight), 1.0f, 1000.0f);
	}
}

DirectX::GraphicsResource DX12Lib::SceneCamera::GetCameraBuffer()
{
	DirectX::XMMATRIX view = this->GetView();
	DirectX::XMMATRIX projection = this->GetProjection();
	DirectX::XMMATRIX viewProjection = view * projection;
	DirectX::XMMATRIX invView = DirectX::XMMatrixInverse(nullptr, view);
	DirectX::XMMATRIX invProjection = DirectX::XMMatrixInverse(nullptr, projection);
	DirectX::XMMATRIX invViewProjection = DirectX::XMMatrixInverse(nullptr, viewProjection);

	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.view, DirectX::XMMatrixTranspose(view));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invView, DirectX::XMMatrixTranspose(invView));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.projection, DirectX::XMMatrixTranspose(projection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invProjection, DirectX::XMMatrixTranspose(invProjection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.viewProjection, DirectX::XMMatrixTranspose(viewProjection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invViewProjection, DirectX::XMMatrixTranspose(invViewProjection));
	m_constantBufferCamera.eyePosition = this->Node->GetPosition();
	m_constantBufferCamera.farPlane = this->m_farZ;
	m_constantBufferCamera.nearPlane = this->m_nearZ;

					

	return Renderer::s_graphicsMemory->AllocateConstant(m_constantBufferCamera);
}

void DX12Lib::SceneCamera::SetOrthogonal(DirectX::XMFLOAT4 bounds)
{
	m_isOrthographic = true;

	m_orthogonalBounds = bounds;

	if (m_isOrthographic)
	{
		Camera::SetOrthogonalBounds(m_orthogonalBounds.x, m_orthogonalBounds.y, m_orthogonalBounds.z, m_orthogonalBounds.w);
	}
	else
	{
		Camera::SetLens(0.25f * DirectX::XM_PI, ((float)Renderer::s_clientWidth / Renderer::s_clientHeight), 1.0f, 1000.0f);
	}
}

void DX12Lib::SceneCamera::SetPerspective(float fov, float aspectRatio, float nearZ, float farZ)
{
	m_isOrthographic = false;

	Camera::SetLens(fov, aspectRatio, nearZ, farZ);
}


