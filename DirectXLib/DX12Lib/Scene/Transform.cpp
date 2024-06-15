#include "DX12Lib/pch.h"
#include "Transform.h"

using namespace DX12Lib;
using namespace DirectX;


DirectX::XMVECTOR DX12Lib::Transform::GetWorldPosition()
{

    XMVECTOR worldPos = XMLoadFloat3(&m_worldPos);

    if (m_dirtyFlags & (UINT)DirtyFlags::Position)
    {
        XMVECTOR relativePos = XMLoadFloat3(&m_relativePos);
        XMVECTOR parentPos = m_parent != nullptr ? m_parent->GetWorldPosition() : DirectX::XMVectorZero();
        worldPos = DirectX::XMVectorAdd(parentPos, relativePos);

        XMStoreFloat3(&m_worldPos, worldPos);

        m_dirtyFlags &= ~(UINT)DirtyFlags::Position;
    }

    return worldPos;
}

DirectX::XMVECTOR DX12Lib::Transform::GetWorldRotation()
{

    XMVECTOR worldRot = XMLoadFloat4(&m_worldRot);

    if (m_dirtyFlags & (UINT)DirtyFlags::Rotation)
    {
        XMVECTOR parentQuaternion = m_parent != nullptr ? m_parent->GetWorldRotation() : XMQuaternionIdentity();
        XMVECTOR relativeQuat = XMLoadFloat4(&m_relativeRot);
        worldRot = DirectX::XMQuaternionMultiply(parentQuaternion, relativeQuat);
        worldRot = DirectX::XMQuaternionNormalize(worldRot);

        XMStoreFloat4(&m_worldRot, worldRot);

        m_dirtyFlags &= ~(UINT)DirtyFlags::Rotation;
    }

    return worldRot;
}

DirectX::XMVECTOR DX12Lib::Transform::GetWorldScale()
{

    XMVECTOR worldScale = XMLoadFloat3(&m_worldScale);

    if (m_dirtyFlags & (UINT)DirtyFlags::Scale)
    {
        XMVECTOR parentScale = m_parent != nullptr ? m_parent->GetWorldScale() : DirectX::XMVectorSet(1.f, 1.f, 1.f, 0.f);
        XMVECTOR relativeScale = XMLoadFloat3(&m_relativeScale);
        worldScale = DirectX::XMVectorMultiply(parentScale, XMLoadFloat3(&m_relativeScale));

        XMStoreFloat3(&m_worldScale, worldScale);

        m_dirtyFlags &= ~(UINT)DirtyFlags::Scale;
    }

    return worldScale;
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetWorldPosition3f()
{
    DirectX::XMStoreFloat3(&m_worldPos, GetWorldPosition());
    return m_worldPos;
}

DirectX::XMFLOAT4 DX12Lib::Transform::GetWorldRotation4f()
{
    DirectX::XMStoreFloat4(&m_worldRot, GetWorldRotation());
    return m_worldRot;
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetWorldRotationEuler3f()
{
    DirectX::XMStoreFloat4(&m_worldRot, GetWorldRotation());
    return QuaternionToEuler(m_worldRot);
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetWorldScale3f()
{
    DirectX::XMStoreFloat3(&m_worldScale, GetWorldScale());
    return m_worldScale;
}

DirectX::XMMATRIX DX12Lib::Transform::GetRelativeWorld()
{
    XMMATRIX worldMatrix = XMLoadFloat4x4(&m_relativeWorld);

    if (m_dirtyFlags != 0)
    {
		XMVECTOR scale = XMLoadFloat3(&m_relativeScale);
		XMVECTOR rotation = XMLoadFloat4(&m_relativeRot);
		XMVECTOR position = XMLoadFloat3(&m_relativePos);

        worldMatrix = XMMatrixScalingFromVector(scale) * XMMatrixRotationQuaternion(rotation) * XMMatrixTranslationFromVector(position);

		m_dirtyFlags = 0;
    }

    return worldMatrix;
}

DirectX::XMMATRIX DX12Lib::Transform::GetWorld()
{

    XMMATRIX worldMatrix = XMLoadFloat4x4(&m_world);

    if (m_dirtyFlags > 0)
    {
        XMVECTOR scale = GetWorldScale();
        XMVECTOR rotation = GetWorldRotation();
        XMVECTOR position = GetWorldPosition();

        worldMatrix = XMMatrixScalingFromVector(scale) * XMMatrixRotationQuaternion(rotation) * XMMatrixTranslationFromVector(position);
        DirectX::XMStoreFloat4x4(&m_world, worldMatrix);

        m_dirtyFlags = 0;
    }

    return worldMatrix;
}

DirectX::XMVECTOR DX12Lib::Transform::GetRight()
{
    auto rotation = this->GetWorldRotation();

    auto right = XMVector3Rotate(XMVectorSet(1.f, 0.f, 0.f, 0.f), rotation);

    return XMVector3Normalize(right);

}

DirectX::XMVECTOR DX12Lib::Transform::GetUp()
{
    auto rotation = this->GetWorldRotation();

	auto up = XMVector3Rotate(XMVectorSet(0.f, 1.f, 0.f, 0.f), rotation);

	return XMVector3Normalize(up);
}

DirectX::XMVECTOR DX12Lib::Transform::GetForward()
{
    auto rotation = this->GetWorldRotation();

	auto forward = XMVector3Rotate(XMVectorSet(0.f, 0.f, 1.f, 0.f), rotation);

	return XMVector3Normalize(forward);
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetRight3f()
{
    auto right = GetRight();

    XMStoreFloat3(&m_right, right);

    return m_right;
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetUp3f()
{
    auto up = GetUp();

	XMStoreFloat3(&m_up, up);

	return m_up;
}

DirectX::XMFLOAT3 DX12Lib::Transform::GetForward3f()
{
    auto forward = GetForward();

	XMStoreFloat3(&m_forward, forward);

	return m_forward;
}

void DX12Lib::Transform::SetRelativePosition(DirectX::FXMVECTOR pos)
{
    // World position will be updated when GetWorldPosition is called since dirty flag is set

	XMStoreFloat3(&m_relativePos, pos);
    SetDirty(DirtyFlags::Position);
}

void DX12Lib::Transform::SetRelativeRotation(DirectX::FXMVECTOR rot)
{
    // World rotation will be updated when GetWorldRotation is called since dirty flag is set

    XMStoreFloat4(&m_relativeRot, rot);
    SetDirty(DirtyFlags::Rotation);
}

void DX12Lib::Transform::SetRelativeScale(DirectX::FXMVECTOR scale)
{
    // World scale will be updated when GetWorldScale is called since dirty flag is set

	XMStoreFloat3(&m_relativeScale, scale);
    SetDirty(DirtyFlags::Scale);
}

void DX12Lib::Transform::SetRelativePosition(DirectX::XMFLOAT3 pos)
{
	XMVECTOR newPos = XMLoadFloat3(&pos);
	this->SetRelativePosition(newPos);
}

void DX12Lib::Transform::SetRelativeRotation(DirectX::XMFLOAT4 rot)
{
    XMVECTOR newRot = XMLoadFloat4(&rot);
	this->SetRelativeRotation(newRot);
}

void DX12Lib::Transform::SetRelativeScale(DirectX::XMFLOAT3 scale)
{
    XMVECTOR newScale = XMLoadFloat3(&scale);
	this->SetRelativeScale(newScale);
}

// We don't care if the parent is dirty since we are delegating to the relative position function
void DX12Lib::Transform::SetWorldPosition(DirectX::FXMVECTOR wpos)
{
    XMVECTOR parentPos = m_parent != nullptr ? m_parent->GetWorldPosition() : DirectX::XMVectorZero();

    // WPos = ParentPos + ChildPos
    // NewChildPos = NewWpos - ParentPos

    XMVECTOR newRelativePos = DirectX::XMVectorSubtract(wpos, parentPos);
    this->SetRelativePosition(newRelativePos);
}

void DX12Lib::Transform::SetWorldRotation(DirectX::FXMVECTOR wrot)
{ 
    XMVECTOR parentRot = m_parent != nullptr ? m_parent->GetWorldRotation() : XMQuaternionIdentity();

    // qw = qp * qc
    // qc = qp^-1 * qw

    // Conjuate equals inverse for unit quaternions
    XMVECTOR invParentRot = DirectX::XMQuaternionConjugate(parentRot);

    this->SetRelativeRotation(DirectX::XMQuaternionNormalize(DirectX::XMQuaternionMultiply(invParentRot, wrot)));
}

void DX12Lib::Transform::SetWorldScale(DirectX::FXMVECTOR wscale)
{
    XMVECTOR parentScale = m_parent != nullptr ? m_parent->GetWorldScale() : DirectX::XMVectorSet(1.f, 1.f, 1.f, 0.f);

    // Assert that no component of the scale is zero
    assert(!XMComparisonAnyTrue(XMVector3EqualR(wscale, XMVectorZero())));
    
    // WScale = ParentScale * ChildScale
    // NewChildScale = WScale / ParentScale

    XMVECTOR newRelativeScale = DirectX::XMVectorDivide(wscale, parentScale);
    this->SetRelativeScale(newRelativeScale);
}

void DX12Lib::Transform::SetWorldPosition(DirectX::XMFLOAT3 pos)
{
    XMVECTOR newPos = XMLoadFloat3(&pos);
	this->SetWorldPosition(newPos);
}

void DX12Lib::Transform::SetWorldRotation(DirectX::XMFLOAT4 rot)
{
    XMVECTOR newRot = XMLoadFloat4(&rot);
	this->SetWorldRotation(newRot);
}

void DX12Lib::Transform::SetWorldScale(DirectX::XMFLOAT3 scale)
{
    XMVECTOR newScale = XMLoadFloat3(&scale);
	this->SetWorldScale(newScale);
}

void DX12Lib::Transform::Update()
{
    GetWorldPosition();
    GetWorldRotation();
    GetWorldScale();
}

DirectX::XMFLOAT3 DX12Lib::Transform::QuaternionToEuler(DirectX::XMFLOAT4 quaternion) const
{
    // Extract quaternion components
    float x = quaternion.x;
    float y = quaternion.y;
    float z = quaternion.z;
    float w = quaternion.w;

    float roll, pitch, yaw;

    // Calculate the Euler angles
    // Pitch (x-axis rotation)
    float sinr_cosp = 2 * (w * x + y * z);
    float cosr_cosp = 1 - 2 * (x * x + y * y);
    pitch = std::atan2(sinr_cosp, cosr_cosp);

    // Yaw (y-axis rotation)
    float sinp = 2 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        yaw = std::copysign(XM_PI / 2, sinp); // Use 90 degrees if out of range
    else
        yaw = std::asin(sinp);

    // Roll (z-axis rotation)
    float siny_cosp = 2 * (w * z + x * y);
    float cosy_cosp = 1 - 2 * (y * y + z * z);
    roll = std::atan2(siny_cosp, cosy_cosp);

    return DirectX::XMFLOAT3(pitch, yaw, roll);

    //const float xx = quaternion.x * quaternion.x;
    //const float yy = quaternion.y * quaternion.y;
    //const float zz = quaternion.z * quaternion.z;

    //const float m31 = 2.f * quaternion.x * quaternion.z + 2.f * quaternion.y * quaternion.w;
    //const float m32 = 2.f * quaternion.y * quaternion.z - 2.f * quaternion.x * quaternion.w;
    //const float m33 = 1.f - 2.f * xx - 2.f * yy;

    //const float cy = sqrtf(m33 * m33 + m31 * m31);
    //const float cx = atan2f(-m32, cy);
    //if (cy > 16.f * FLT_EPSILON)
    //{
    //    const float m12 = 2.f * quaternion.x * quaternion.y + 2.f * quaternion.z * quaternion.w;
    //    const float m22 = 1.f - 2.f * xx - 2.f * zz;

    //    return DirectX::XMFLOAT3(cx, atan2f(m31, m33), atan2f(m12, m22));
    //}
    //else
    //{
    //    const float m11 = 1.f - 2.f * yy - 2.f * zz;
    //    const float m21 = 2.f * quaternion.x * quaternion.y - 2.f * quaternion.z * quaternion.w;

    //    return DirectX::XMFLOAT3(cx, 0.f, atan2f(-m21, m11));
    //}
    
}

void DX12Lib::Transform::SetDirty(DirtyFlags flag)
{
    m_dirtyFlags |= (UINT)flag;
    m_dirtForFrame |= (UINT)flag;
}

void DX12Lib::Transform::SetDirty(UINT32 flags)
{
    m_dirtyFlags |= flags;
	m_dirtForFrame |= flags;
}
