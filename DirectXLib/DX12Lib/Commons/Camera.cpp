#include "DX12Lib/pch.h"
#include "Camera.h"

using namespace DirectX;
using namespace DX12Lib;

Camera::Camera()
{
	SetLens(0.25f * XM_PI, 1.0f, 1.0f, 1000.0f);
}

Camera::~Camera()
{
}


float Camera::GetNearZ() const
{
	return m_nearZ;
}

float Camera::GetFarZ() const
{
	return m_farZ;
}

float Camera::GetAspect() const
{
	return m_aspect;
}

float Camera::GetFovY() const
{
	return m_fovY;
}

float Camera::GetFovX() const
{
	float halfWidth = 0.5f * GetNearWindowWidth();
	return 2.0f * atan(halfWidth / m_nearZ);
}

float Camera::GetNearWindowWidth() const
{
	return m_aspect * m_nearWindowHeight;
}

float Camera::GetNearWindowHeight() const
{
	return m_nearWindowHeight;
}

float Camera::GetFarWindowWidth() const
{
	return m_aspect * m_farWindowHeight;
}

float Camera::GetFarWindowHeight() const
{
	return m_farWindowHeight;
}

void Camera::SetLens(float fovY, float aspect, float zn, float zf)
{
	m_fovY = fovY;
	m_aspect = aspect;
	m_nearZ = zn;
	m_farZ = zf;

	m_nearWindowHeight = 2.0f * m_nearZ * tan(0.5f * m_fovY);
	m_farWindowHeight = 2.0f * m_farZ * tan(0.5f * m_fovY);

	XMMATRIX P = XMMatrixPerspectiveFovLH(m_fovY, m_aspect, m_nearZ, m_farZ);
	XMStoreFloat4x4(&m_proj, P);
}

void DX12Lib::Camera::SetOrthogonalBounds(DirectX::XMFLOAT3 center, DirectX::XMFLOAT3 halfExtents)
{
	float l = center.x - halfExtents.x;
	float r = center.x + halfExtents.x;
	float b = center.y - halfExtents.y;
	float t = center.y + halfExtents.y;
	float n = center.z - halfExtents.z;
	float f = center.z + halfExtents.z;

	m_nearZ = n;
	m_farZ = f;

	XMMATRIX P = XMMatrixOrthographicOffCenterLH(l, r, b, t, n, f);

	XMStoreFloat4x4(&m_proj, P);
}

void DX12Lib::Camera::SetOrthogonalBounds(float width, float height, float nearZ, float farZ)
{
	m_nearZ = nearZ;
	m_farZ = farZ;

	XMMATRIX P = XMMatrixOrthographicLH(width, height, nearZ, farZ);

	XMStoreFloat4x4(&m_proj, P);
}


XMMATRIX Camera::GetView() const
{
	return XMLoadFloat4x4(&m_view);
}

XMMATRIX Camera::GetProjection() const
{
	return XMLoadFloat4x4(&m_proj);
}

XMFLOAT4X4 Camera::GetView4x4f() const
{
	return m_view;
}

XMFLOAT4X4 Camera::GetProjection4x4f() const
{
	return m_proj;
}

void DX12Lib::Camera::UpdateViewMatrix(DirectX::FXMVECTOR pos, DirectX::FXMVECTOR up, DirectX::FXMVECTOR forward, DirectX::GXMVECTOR right)
{

	XMVECTOR R = right;
	XMVECTOR U = up;
	XMVECTOR L = forward;
	XMVECTOR P = pos;

	L = XMVector3Normalize(L);
	U = XMVector3Normalize(XMVector3Cross(L, R));
	R = XMVector3Cross(U, L);

	float x = -XMVectorGetX(XMVector3Dot(P, R));
	float y = -XMVectorGetX(XMVector3Dot(P, U));
	float z = -XMVectorGetX(XMVector3Dot(P, L));


	DirectX::XMFLOAT3 right3f;
	DirectX::XMFLOAT3 up3f;
	DirectX::XMFLOAT3 look3f;

	XMStoreFloat3(&right3f, R);
	XMStoreFloat3(&up3f, U);
	XMStoreFloat3(&look3f, L);

	m_view(0, 0) = right3f.x;
	m_view(1, 0) = right3f.y;
	m_view(2, 0) = right3f.z;
	m_view(3, 0) = x;

	m_view(0, 1) = up3f.x;
	m_view(1, 1) = up3f.y;
	m_view(2, 1) = up3f.z;
	m_view(3, 1) = y;

	m_view(0, 2) = look3f.x;
	m_view(1, 2) = look3f.y;
	m_view(2, 2) = look3f.z;
	m_view(3, 2) = z;

	m_view(0, 3) = 0.0f;
	m_view(1, 3) = 0.0f;
	m_view(2, 3) = 0.0f;
	m_view(3, 3) = 1.0f;


	
}
