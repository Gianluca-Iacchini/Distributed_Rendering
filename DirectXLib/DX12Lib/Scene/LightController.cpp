#include "DX12Lib/pch.h"
#include "LightController.h"

#include "DX12Lib/Commons/NetworkManager.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

void LightController::Init(DX12Lib::CommandContext& context)
{
}

void LightController::Update(DX12Lib::CommandContext& context)
{
	float speed = 0.25f;
	float deltaTime = GameTime::GetDeltaTime();

	if (!m_controlOverNetwork)
		Move(speed, deltaTime);
	else
		PredictInput(speed, deltaTime);
}

void DX12Lib::LightController::ControlOverNetwork(bool control)
{
	m_controlOverNetwork = control;
}

void LightController::SetRemoteInput(NodeInputState nodeState, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 clientTimestamp)
{

	m_inputState.UpdateState(nodeState);

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

void LightController::SetRemoteInput(std::uint8_t nodeBitMask, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 timestamp)
{
	NodeInputState newState;
	newState.NodeForward = (nodeBitMask & 0x01) ? RemoteInput::ON : RemoteInput::OFF;
	newState.NodeBackward = (nodeBitMask & 0x02) ? RemoteInput::ON : RemoteInput::OFF;
	newState.NodeLeft = (nodeBitMask & 0x04) ? RemoteInput::ON : RemoteInput::OFF;
	newState.NodeRight = (nodeBitMask & 0x08) ? RemoteInput::ON : RemoteInput::OFF;
	newState.NodeUp = (nodeBitMask & 0x10) ? RemoteInput::ON : RemoteInput::OFF;
	newState.NodeDown = (nodeBitMask & 0x20) ? RemoteInput::ON : RemoteInput::OFF;

	SetRemoteInput(newState, clientAbsPos, clientAbsRot, timestamp);
}

void LightController::Move(float speed, float deltaTime)
{
	auto kbState = Graphics::s_kbTracker->GetLastState();
	auto mouseState = Graphics::s_mouse->GetState();


	if (kbState.Up)
		this->Node->Rotate(Node->GetUp(), deltaTime * speed);
	else if (kbState.Down)
		this->Node->Rotate(Node->GetUp(), -deltaTime * speed);
	if (kbState.Left)
		this->Node->Rotate(Node->GetRight(), -deltaTime * speed);
	else if (kbState.Right)
		this->Node->Rotate(Node->GetRight(), deltaTime * speed);


}

void LightController::PredictInput(float speed, float deltaTime)
{
	DirectX::XMFLOAT3 currentPos = this->Node->GetPosition();
	

	if (m_inputState.NodeForward == RemoteInput::ON)
		this->Node->Rotate(Node->GetUp(), deltaTime * speed);
	if (m_inputState.NodeBackward == RemoteInput::ON)
		this->Node->Rotate(Node->GetUp(), -deltaTime * speed);
	if (m_inputState.NodeLeft == RemoteInput::ON)
		this->Node->Rotate(Node->GetRight(), -deltaTime * speed);
	if (m_inputState.NodeRight == RemoteInput::ON)
		this->Node->Rotate(Node->GetRight(), deltaTime * speed);

	DirectX::XMFLOAT3 newPos = this->Node->GetPosition();

	DirectX::XMFLOAT3 deltaPos = { newPos.x - currentPos.x, newPos.y - currentPos.y, newPos.z - currentPos.z };

	// Using epoch time instead of GameTime, since GameTime starts at 0 when the application is initialized.
	InputDelta delta = { deltaPos, deltaTime, NetworkHost::GetEpochTime() };

	if (m_inputDeltaHistory.size() >= c_maxHistorySize)
		m_inputDeltaHistory.erase(m_inputDeltaHistory.begin());

	m_inputDeltaHistory.push_back(delta);
}

void LightController::TrimHistory(UINT64 clientTimestamp)
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
