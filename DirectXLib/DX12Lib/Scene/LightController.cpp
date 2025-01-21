#include "DX12Lib/pch.h"
#include "LightController.h"

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


	Move(speed, deltaTime);

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
