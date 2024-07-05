#include "DX12Lib/pch.h"
#include "CameraController.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace DirectX;

void CVGI::CameraController::Init(DX12Lib::CommandContext& context)
{
	Graphics::s_mouse->SetMode(Mouse::MODE_RELATIVE);
}

void CameraController::Update(DX12Lib::CommandContext& context)
{


	float speed = 3.0f;

	float deltaTime = GameTime::GetDeltaTime();

	auto kbState = Graphics::s_kbTracker->GetLastState();
	auto mouseState = Graphics::s_mouse->GetState();

	if (Graphics::s_kbTracker->IsKeyPressed(Keyboard::Escape))
		PostQuitMessage(0);

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

