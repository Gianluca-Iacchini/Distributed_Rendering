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

XMVECTOR Camera::GetPosition() const
{
	return XMLoadFloat3(&m_position);
}

XMFLOAT3 Camera::GetPosition3f() const
{
	return m_position;
}

void Camera::SetPosition(float x, float y, float z)
{
	m_position = XMFLOAT3(x, y, z);
	m_viewDirty = true;
}

void Camera::SetPosition(const XMFLOAT3& v)
{
	m_position = v;
	m_viewDirty = true;
}

XMVECTOR Camera::GetRight() const
{
	return XMLoadFloat3(&m_right);
}

XMFLOAT3 Camera::GetRight3f() const
{
	return m_right;
}

XMVECTOR Camera::GetUp() const
{
	return XMLoadFloat3(&m_up);
}

XMFLOAT3 Camera::GetUp3f() const
{
	return m_up;
}

XMVECTOR Camera::GetLook() const
{
	return XMLoadFloat3(&m_look);
}

XMFLOAT3 Camera::GetLook3f() const
{
	return m_look;
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

void Camera::LookAt(FXMVECTOR pos, FXMVECTOR target, FXMVECTOR worldUp)
{
	XMVECTOR L = XMVector3Normalize(XMVectorSubtract(target, pos));
	XMVECTOR R = XMVector3Normalize(XMVector3Cross(worldUp, L));
	XMVECTOR U = XMVector3Cross(L, R);

	XMStoreFloat3(&m_position, pos);
	XMStoreFloat3(&m_look, L);
	XMStoreFloat3(&m_right, R);
	XMStoreFloat3(&m_up, U);

	m_viewDirty = true;
}

void Camera::LookAt(const XMFLOAT3& pos, const XMFLOAT3& target, const XMFLOAT3& up)
{
	XMVECTOR P = XMLoadFloat3(&pos);
	XMVECTOR T = XMLoadFloat3(&target);
	XMVECTOR U = XMLoadFloat3(&up);

	LookAt(P, T, U);
}

XMMATRIX Camera::GetView() const
{
	assert(!m_viewDirty);
	return XMLoadFloat4x4(&m_view);
}

XMMATRIX Camera::GetProjection() const
{
	return XMLoadFloat4x4(&m_proj);
}

XMFLOAT4X4 Camera::GetView4x4f() const
{
	assert(!m_viewDirty);
	return m_view;
}

XMFLOAT4X4 Camera::GetProjection4x4f() const
{
	return m_proj;
}

void Camera::Strafe(float d)
{
	XMVECTOR s = XMVectorReplicate(d);
	XMVECTOR r = XMLoadFloat3(&m_right);
	XMVECTOR p = XMLoadFloat3(&m_position);
	XMStoreFloat3(&m_position, XMVectorMultiplyAdd(s, r, p));

	m_viewDirty = true;
}

void Camera::Walk(float d)
{
	XMVECTOR s = XMVectorReplicate(d);
	XMVECTOR l = XMLoadFloat3(&m_look);
	XMVECTOR p = XMLoadFloat3(&m_position);
	XMStoreFloat3(&m_position, XMVectorMultiplyAdd(s, l, p));

	m_viewDirty = true;
}

void Camera::Lift(float d)
{
	XMVECTOR s = XMVectorReplicate(d);
	XMVECTOR u = XMLoadFloat3(&m_up);
	XMVECTOR p = XMLoadFloat3(&m_position);
	XMStoreFloat3(&m_position, XMVectorMultiplyAdd(s, u, p));

	m_viewDirty = true;
}

void Camera::Pitch(float angle)
{
	XMMATRIX R = XMMatrixRotationAxis(XMLoadFloat3(&m_right), angle);

	XMStoreFloat3(&m_up, XMVector3TransformNormal(XMLoadFloat3(&m_up), R));
	XMStoreFloat3(&m_look, XMVector3TransformNormal(XMLoadFloat3(&m_look), R));

	m_viewDirty = true;
}

void Camera::Yaw(float angle)
{
	XMMATRIX R = XMMatrixRotationY(angle);

	XMStoreFloat3(&m_right, XMVector3TransformNormal(XMLoadFloat3(&m_right), R));
	XMStoreFloat3(&m_up, XMVector3TransformNormal(XMLoadFloat3(&m_up), R));
	XMStoreFloat3(&m_look, XMVector3TransformNormal(XMLoadFloat3(&m_look), R));

	m_viewDirty = true;
}

void Camera::UpdateViewMatrix()
{
	if (m_viewDirty)
	{
		XMVECTOR R = XMLoadFloat3(&m_right);
		XMVECTOR U = XMLoadFloat3(&m_up);
		XMVECTOR L = XMLoadFloat3(&m_look);
		XMVECTOR P = XMLoadFloat3(&m_position);

		L = XMVector3Normalize(L);
		U = XMVector3Normalize(XMVector3Cross(L, R));
		R = XMVector3Cross(U, L);

		float x = -XMVectorGetX(XMVector3Dot(P, R));
		float y = -XMVectorGetX(XMVector3Dot(P, U));
		float z = -XMVectorGetX(XMVector3Dot(P, L));

		XMStoreFloat3(&m_right, R);
		XMStoreFloat3(&m_up, U);
		XMStoreFloat3(&m_look, L);

		m_view(0, 0) = m_right.x;
		m_view(1, 0) = m_right.y;
		m_view(2, 0) = m_right.z;
		m_view(3, 0) = x;

		m_view(0, 1) = m_up.x;
		m_view(1, 1) = m_up.y;
		m_view(2, 1) = m_up.z;
		m_view(3, 1) = y;

		m_view(0, 2) = m_look.x;
		m_view(1, 2) = m_look.y;
		m_view(2, 2) = m_look.z;
		m_view(3, 2) = z;

		m_view(0, 3) = 0.0f;
		m_view(1, 3) = 0.0f;
		m_view(2, 3) = 0.0f;
		m_view(3, 3) = 1.0f;

		m_viewDirty = false;
	}
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
