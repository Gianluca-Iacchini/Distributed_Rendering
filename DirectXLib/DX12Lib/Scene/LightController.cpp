#include "C:/Users/iacco/Desktop/DistributedRendering/build_vs2022/DirectXLib/CMakeFiles/DirectXLib.dir/Debug/cmake_pch.hxx"
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

	float deltaTime = 1.0f / 60.0f;

	auto kbState = Graphics::s_kbTracker->GetLastState();

	if (kbState.Left)
		this->Node->Rotate(Node->GetRight(), -deltaTime * speed);
	else if (kbState.Right)
		this->Node->Rotate(Node->GetRight(), deltaTime * speed);

	if (kbState.Up)
		this->Node->Rotate(Node->GetUp(), deltaTime * speed);
	else if (kbState.Down)
		this->Node->Rotate(Node->GetUp(), -deltaTime * speed);
}