#include "DX12Lib/pch.h"
#include "SceneCamera.h"
#include "Mouse.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

void DX12Lib::SceneCamera::Init(CommandContext& context)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, ((float)Renderer::s_clientWidth / Renderer::s_clientHeight), 1.0f, 1000.0f);
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
	Renderer::AddMainCamera(this);
}

void DX12Lib::SceneCamera::OnResize(CommandContext& context, int newWidth, int newHeight)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, ((float)newWidth / newHeight), 1.0f, 1000.0f);
}

void DX12Lib::SceneCamera::UseCamera(CommandContext& context)
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

	auto cbCamera = Renderer::s_graphicsMemory->AllocateConstant(m_constantBufferCamera);

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
		(UINT)Renderer::RootSignatureSlot::CameraCBV,
		cbCamera.GpuAddress()
	);
}
