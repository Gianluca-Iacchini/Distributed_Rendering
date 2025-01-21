#include "DX12Lib/pch.h"
#include "CameraController.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace DirectX;

void CVGI::CameraController::Init(DX12Lib::CommandContext& context)
{
	Graphics::s_mouse->SetMode(Mouse::MODE_ABSOLUTE);

}

void CameraController::Update(DX12Lib::CommandContext& context)
{
	float speed = 3.0f;
	float deltaTime = GameTime::GetDeltaTime();

	if (Graphics::s_kbTracker->IsKeyPressed(Keyboard::Escape))
		PostQuitMessage(0);

	auto tracker = Graphics::s_mouseTracker.get();

	if (tracker->rightButton == Mouse::ButtonStateTracker::PRESSED)
	{
		Graphics::s_mouse->SetMode(Mouse::MODE_RELATIVE);
	}
	else if (tracker->rightButton == Mouse::ButtonStateTracker::RELEASED)
	{
		Graphics::s_mouse->SetMode(Mouse::MODE_ABSOLUTE);
	}

	if (IsRemote)
		PredictInput(speed, deltaTime);
	else
		Move(speed, deltaTime);

}

void CVGI::CameraController::SetRemoteInput(CameraState cameraState, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 clientTimestamp)
{
	
	m_cameraInputState.UpdateState(cameraState);

	DirectX::XMFLOAT3 currentPos3f = this->Node->GetPosition();

	DirectX::XMVECTOR clientPos = DirectX::XMLoadFloat3(&clientAbsPos);
	DirectX::XMVECTOR currentPos = DirectX::XMLoadFloat3(&currentPos3f);

	DirectX::XMVECTOR posDelta = DirectX::XMVectorSubtract(clientPos, currentPos);

	if (DirectX::XMVectorGetX(DirectX::XMVector3Length(posDelta)) > m_maxPositionError)
	{
		this->Node->SetPosition(clientAbsPos);
	}
	else 
	{
		// Smoothly interpolate the position
		DirectX::XMVECTOR newPos = DirectX::XMVectorLerp(currentPos, clientPos, 0.95f);

		//TrimHistory(clientTimestamp);

		//for (auto& delta : m_inputDeltaHistory)
		//{
		//	newPos = DirectX::XMVectorAdd(newPos, DirectX::XMLoadFloat3(&delta.PositionDelta));
		//}

		//DirectX::XMFLOAT3 pos;
		//DirectX::XMStoreFloat3(&pos, newPos);
		this->Node->SetPosition(clientAbsPos);
	}

	DirectX::XMFLOAT4 currentRot4f = this->Node->GetRotationQuaternion();

	DirectX::XMVECTOR clientQuatRot = DirectX::XMLoadFloat4(&clientAbsRot);
	DirectX::XMVECTOR currentQuatRot = DirectX::XMLoadFloat4(&currentRot4f);

	// Compute the dot product between the two quaternions
	float dot = DirectX::XMVectorGetX(DirectX::XMQuaternionDot(clientQuatRot, currentQuatRot));

	dot = MathHelper::Clamp(dot, -1.0f, 1.0f);

	// Calculate the angle difference (shortest path)
	float quatDiff = 2.0f * std::acosf(std::fabs(dot)); // Use fabs to ensure shortest path

	if (quatDiff > m_maxRotationError)
	{
		this->Node->SetRotationQuaternion(clientAbsRot);
	}
	else
	{
		// Smoothly interpolate the rotation
		DirectX::XMVECTOR newRot = DirectX::XMQuaternionSlerp(currentQuatRot, clientQuatRot, 1.0f);

		DirectX::XMFLOAT4 rot;
		DirectX::XMStoreFloat4(&rot, newRot);
		this->Node->SetRotationQuaternion(rot);
	}
}

void CVGI::CameraController::SetRemoteInput(std::uint8_t cameraBitmask, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 timestamp)
{
	CameraState newState;
	newState.CameraForward = (cameraBitmask & 0x01) ? RemoteInput::ON : RemoteInput::OFF;
	newState.CameraBackward = (cameraBitmask & 0x02) ? RemoteInput::ON : RemoteInput::OFF;
	newState.CameraLeft = (cameraBitmask & 0x04) ? RemoteInput::ON : RemoteInput::OFF;
	newState.CameraRight = (cameraBitmask & 0x08) ? RemoteInput::ON : RemoteInput::OFF;
	newState.CameraUp = (cameraBitmask & 0x10) ? RemoteInput::ON : RemoteInput::OFF;
	newState.CameraDown = (cameraBitmask & 0x20) ? RemoteInput::ON : RemoteInput::OFF;

	SetRemoteInput(newState, clientAbsPos, clientAbsRot, timestamp);
}

void CVGI::CameraController::Move(float speed, float deltaTime)
{
	auto kbState = Graphics::s_kbTracker->GetLastState();
	auto mouseState = Graphics::s_mouseTracker->GetLastState();


	if (mouseState.positionMode == Mouse::MODE_RELATIVE)
	{
		if (mouseState.x != 0 || mouseState.y != 0)
		{
			auto rotation = this->Node->GetRotationEulerAngles();
			this->Node->Rotate(Node->GetRight(), mouseState.y * deltaTime);
			this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, mouseState.x * deltaTime);
		}
	}

	if (kbState.LeftShift)
		speed *= 3.0f;

	if (kbState.W)
		this->Node->Translate(Node->GetForward(), deltaTime * speed);
	if (kbState.S)
		this->Node->Translate(Node->GetForward(), -deltaTime * speed);
	if (kbState.A)
		this->Node->Translate(Node->GetRight(), -deltaTime * speed);
	if (kbState.D)
		this->Node->Translate(Node->GetRight(), deltaTime * speed);
	if (kbState.E)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, deltaTime * speed);
	if (kbState.Q)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, -deltaTime * speed);
}

void CVGI::CameraController::PredictInput(float speed, float deltaTime)
{
	DirectX::XMFLOAT3 currentPos = this->Node->GetPosition();

	if (m_cameraInputState.CameraForward == RemoteInput::ON)
		this->Node->Translate(Node->GetForward(), deltaTime * speed);
	if (m_cameraInputState.CameraBackward == RemoteInput::ON)
		this->Node->Translate(Node->GetForward(), -deltaTime * speed);
	if (m_cameraInputState.CameraLeft == RemoteInput::ON)
		this->Node->Translate(Node->GetRight(), -deltaTime * speed);
	if (m_cameraInputState.CameraRight == RemoteInput::ON)
		this->Node->Translate(Node->GetRight(), deltaTime * speed);
	if (m_cameraInputState.CameraUp == RemoteInput::ON)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, deltaTime * speed);
	if (m_cameraInputState.CameraDown == RemoteInput::ON)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, -deltaTime * speed);

	DirectX::XMFLOAT3 newPos = this->Node->GetPosition();

	DirectX::XMFLOAT3 deltaPos = { newPos.x - currentPos.x, newPos.y - currentPos.y, newPos.z - currentPos.z };

	// Using epoch time instead of GameTime, since GameTime starts at 0 when the application is initialized.
	auto epochTime = std::chrono::system_clock::now();
	float epochMs = epochTime.time_since_epoch() / std::chrono::milliseconds(1);

	InputDelta delta = { deltaPos, deltaTime, epochMs};

	if (m_inputDeltaHistory.size() >= c_maxHistorySize)
		m_inputDeltaHistory.erase(m_inputDeltaHistory.begin());

	m_inputDeltaHistory.push_back(delta);
}

void CVGI::CameraController::TrimHistory(UINT64 clientTimestamp)
{
	// Remove all deltas that are older than the client timestamp.
	UINT32 elementsToRemove = 0;
	while (!m_inputDeltaHistory.empty() && m_inputDeltaHistory.front().timestamp < clientTimestamp)
	{
		elementsToRemove++;
	}

	m_inputDeltaHistory.erase(m_inputDeltaHistory.begin(), m_inputDeltaHistory.begin() + elementsToRemove);

	// If the oldest delta partially overlaps with the client timestamp, than we scale it.
	if (!m_inputDeltaHistory.empty())
	{
		InputDelta delta = m_inputDeltaHistory.front();
		float overlap = (static_cast<float>(delta.timestamp) - static_cast<float>(clientTimestamp)) / (delta.deltaTime * 1000.0f);
		
		delta.PositionDelta.x *= overlap;
		delta.PositionDelta.y *= overlap;
		delta.PositionDelta.z *= overlap;
		delta.deltaTime *= overlap;

		m_inputDeltaHistory.front() = delta;
	}
}

