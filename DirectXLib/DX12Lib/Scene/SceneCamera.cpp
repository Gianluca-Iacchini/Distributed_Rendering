#include "DX12Lib/pch.h"
#include "SceneCamera.h"
#include "Mouse.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

void DX12Lib::SceneCamera::Init(CommandContext& context)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, (1920.f / 1080.f), 1.0f, 1000.0f);
	Camera::SetPosition(this->Node->GetPosition());
	Camera::UpdateViewMatrix();
}

void DX12Lib::SceneCamera::Update(CommandContext& context)
{
	auto& transform = this->Node->Transform;
	float deltaTime = this->Node->Scene.Time().DeltaTime();

	auto state = s_mouse->GetState();
	if (state.positionMode == Mouse::MODE_RELATIVE)
	{

		float dx = XMConvertToRadians(0.25f * static_cast<float>(state.x));
		float dy = XMConvertToRadians(0.25f * static_cast<float>(state.y));

		auto rotation = this->Node->GetRotationEulerAngles();
		rotation.x += state.y * deltaTime;
		rotation.y += state.x * deltaTime;

		this->Node->SetRotationEulerAngles(rotation);
	}

	auto kbState = s_kbTracker->GetLastState();

	if (s_kbTracker->IsKeyPressed(Keyboard::Escape))
		PostQuitMessage(0);

	auto pos = this->Node->GetPosition();

	XMVECTOR posXM = XMLoadFloat3(&pos);
	XMVECTOR look = this->GetLook();

	if (kbState.W)
		posXM += transform.GetForward() * deltaTime;
	if (kbState.S)
		posXM -= transform.GetForward() * deltaTime;
	if (kbState.A)
		posXM -= transform.GetRight() * deltaTime;
	if (kbState.D)
		posXM += transform.GetRight() * deltaTime;
	if (kbState.E)
		posXM += XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f) * deltaTime;
	if (kbState.Q)
		posXM -= XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f ) * deltaTime;

	XMStoreFloat3(&pos, posXM);

	this->Node->SetPosition(pos);

	auto m_right = transform.GetRight3f();

	if (Node->IsTransformDirty())
		Camera::UpdateViewMatrix(transform.GetWorldPosition(), transform.GetUp(), transform.GetForward(), transform.GetRight());
}

void DX12Lib::SceneCamera::Render(CommandContext& context)
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
	DirectX::XMFLOAT3 eyePosition = this->Node->GetPosition();
	float nearPlane = this->m_nearZ;
	float farPlane = this->m_farZ;

	auto cbCamera = Renderer::s_graphicsMemory->AllocateConstant(m_constantBufferCamera);

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView(2, cbCamera.GpuAddress());
}

void DX12Lib::SceneCamera::OnResize(CommandContext& context)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, (1920.f / 1080.f), 1.0f, 1000.0f);
}
