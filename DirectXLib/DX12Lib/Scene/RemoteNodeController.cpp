#include "DX12Lib/pch.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "RemoteNodeController.h"

DX12Lib::RemoteNodeController::RemoteNodeController()
{
	m_lastVelocity = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
}

void DX12Lib::RemoteNodeController::FeedRemoteData(DirectX::XMFLOAT3 velocity, DirectX::XMFLOAT3 absPos, DirectX::XMFLOAT4 absRot, UINT64 timestamp)
{
    if (!m_isRemoteControlled)
    {
        return;
    }

    m_lastVelocity = velocity;

    UINT64 deltaTime = GameTime::GetTimeSinceEpoch() - timestamp;
    float dt = deltaTime / 1000000.0f;

    // Move based on velocity and deltaTime
    this->Node->Translate(velocity.x * dt, velocity.y * dt, velocity.z * dt);

    // Calculate the distance between the current position and the target position
    DirectX::XMFLOAT3 currentPos = this->Node->GetPosition();
    float distance = MathHelper::Distance(currentPos, absPos);

    // Use a threshold to determine if smoothing should happen
    if (distance > 0.1f)
    {
        // Smooth the movement using lerp (linear interpolation)
        float smoothingFactor = 0.5f; // Control the smoothing speed (higher is slower, lower is faster)
        DirectX::XMFLOAT3 smoothedPos;

        // Interpolate between the current position and the target position
        smoothedPos.x = MathHelper::Lerp(currentPos.x, absPos.x, smoothingFactor);
        smoothedPos.y = MathHelper::Lerp(currentPos.y, absPos.y, smoothingFactor);
        smoothedPos.z = MathHelper::Lerp(currentPos.z, absPos.z, smoothingFactor);

        // Set the position to the smoothed value
        this->Node->SetPosition(smoothedPos);
    }
    else
    {
        // If the distance is small enough, just snap to the target position
        this->Node->SetPosition(absPos);
    }

    // Smooth rotation using SLERP
    DirectX::XMFLOAT4 currentRotation = this->Node->GetRotationQuaternion();
    DirectX::XMFLOAT4 smoothedRotation;

    // Calculate the angle difference between the current and target rotation
    DirectX::XMVECTOR currentQuat = DirectX::XMLoadFloat4(&currentRotation);
    DirectX::XMVECTOR targetQuat = DirectX::XMLoadFloat4(&absRot);
    float dotProduct = DirectX::XMVectorGetX(DirectX::XMQuaternionDot(currentQuat, targetQuat));

    // If the angle difference is too large, snap to the target rotation
	const float angleThreshold = 0.86602540378f;  // Cosine of 30 degrees

    if (dotProduct < angleThreshold)  // If the angle is too large, snap the rotation
    {
        smoothedRotation = absRot;  // Snap to the absolute rotation
    }
    else
    {
        // Otherwise, interpolate between the current rotation and the target rotation
        DirectX::XMStoreFloat4(&smoothedRotation, DirectX::XMQuaternionSlerp(
            currentQuat,
            targetQuat,
            0.1f  // Smoothing factor for rotation
        ));
    }

    // Set the interpolated or snapped rotation
    this->Node->SetRotationQuaternion(smoothedRotation);

}

void DX12Lib::RemoteNodeController::FeedRemoteData(DirectX::XMFLOAT2 mouseDeltaXY, std::uint8_t inputBitmap, UINT64 timestamp, float clientDeltaTime)
{
	DirectX::XMFLOAT3 velocity = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

    // W
    if (inputBitmap & (1 << 0))
	{
		velocity.z += 1.0f;
	}
    // S
	if (inputBitmap & (1 << 1))
	{
		velocity.z += -1.0f;
	}
    // A
    if (inputBitmap & (1 << 2))
    {
        velocity.x += -1.0f;
	}
    // D
    if (inputBitmap & (1 << 3))
	{
		velocity.x += 1.0f;
	}
    // E
    if (inputBitmap & (1 << 4))
	{
		velocity.y += 1.0f;
	}
	// Q
    if (inputBitmap & (1 << 5))
    {
		velocity.y += -1.0f;
    }
    // Shift
	if (inputBitmap & (1 << 6))
    {
		velocity.x *= 3.0f;
		velocity.y *= 3.0f;
		velocity.z *= 3.0f;
	}

    UINT64 deltaTime = GameTime::GetTimeSinceEpoch() - timestamp;
    float dt = deltaTime / 1000000.0f;
    dt = dt / clientDeltaTime;

	DirectX::XMFLOAT3 currentPos = this->Node->GetPosition();

	this->Node->Translate(this->Node->GetForward(), velocity.z * dt);
	this->Node->Translate(this->Node->GetRight(), velocity.x * dt);
	this->Node->Translate({ 0.0f, 1.0f, 0.0f }, velocity.y * dt);

    float mouseX = mouseDeltaXY.x * Graphics::Renderer::s_clientWidth * dt;
    float mouseY = mouseDeltaXY.y * Graphics::Renderer::s_clientHeight * dt;



    this->Node->Rotate(Node->GetRight(), mouseY);
    this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, mouseX);


	DirectX::XMFLOAT3 newPos = this->Node->GetPosition();

	m_lastVelocity.x = (newPos.x - currentPos.x) / dt;
	m_lastVelocity.y = (newPos.y - currentPos.y) / dt;
	m_lastVelocity.z = (newPos.z - currentPos.z) / dt;
}

void DX12Lib::RemoteNodeController::Update(DX12Lib::CommandContext& context)
{
    if (m_isRemoteControlled)
    {
		float deltaTime = GameTime::GetDeltaTime();

        this->Node->Translate(m_lastVelocity.x * deltaTime, m_lastVelocity.y * deltaTime, m_lastVelocity.z * deltaTime);
    }
}
