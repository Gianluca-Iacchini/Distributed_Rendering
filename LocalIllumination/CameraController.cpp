#include "CameraController.h"
#include "LIScene.h"
#include "DX12Lib/pch.h"
#include <string_view>

using namespace LI;
using namespace DX12Lib;
using namespace DirectX;

#define STREAMING 0

void CameraController::Update(DX12Lib::CommandContext& context)
{
	LIScene* scene = dynamic_cast<LIScene*>(this->Node->GetScene());
	
	auto input = std::vector<char>();// scene->GetNetworkData();

	std::string inputData = std::string(input.data(), input.size());

	float mouseX = 0.0f, mouseY = 0.0f;

	ParseInputString(inputData, &mouseX, &mouseY, &m_cameraForward, &m_cameraStrafe, &m_cameraLift);



	auto& transform = this->Node->Transform;
	float deltaTime = GameTime::GetDeltaTime();



	auto kbState = Graphics::s_kbTracker->GetLastState();

	if (Graphics::s_kbTracker->IsKeyPressed(Keyboard::Escape))
		PostQuitMessage(0);


#if STREAMING
	float deltaX = mouseX * Graphics::Renderer::s_clientWidth * deltaTime;
	float deltaY = mouseY * Graphics::Renderer::s_clientHeight * deltaTime;

	this->Node->Translate(Node->GetForward(), m_cameraForward * deltaTime);
	this->Node->Translate(Node->GetRight(), m_cameraStrafe * deltaTime);
	this->Node->Translate({ 0.0f, 1.0f, 0.0f }, m_cameraLift * deltaTime);

	this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, deltaX);
	this->Node->Rotate(Node->GetRight(), deltaY);
#else

	auto state = Graphics::s_mouse->GetState();

	if (state.positionMode == Mouse::MODE_RELATIVE)
	{
		if (state.x != 0 || state.y != 0)
		{
			auto rotation = this->Node->GetRotationEulerAngles();
			this->Node->Rotate(Node->GetRight(), state.y * deltaTime);
			this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, state.x * deltaTime);
		}
	}

	if (kbState.W)
		this->Node->Translate(Node->GetForward(), deltaTime);
	if (kbState.S)
		this->Node->Translate(Node->GetForward(), -deltaTime);
	if (kbState.A)
		this->Node->Translate(Node->GetRight(), -deltaTime);
	if (kbState.D)
		this->Node->Translate(Node->GetRight(), deltaTime);
	if (kbState.E)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, deltaTime);
	if (kbState.Q)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, -deltaTime);
#endif
}

void CameraController::ParseInputString(std::string& input, float* mouseX, float* mouseY, int* cameraForward, int* cameraStrafe, int* cameraLift)
{
	size_t pos = 0, end = 0;
	while ((end = input.find('\n', pos)) != std::string::npos) {
		std::string_view line = std::string_view(input).substr(pos, end - pos);
		pos = end + 1;

		if (line.substr(0, 2) == "M ") {
			size_t xPos = line.find("x:");
			if (xPos != std::string::npos) {
				size_t xEnd = line.find(' ', xPos);
				*mouseX = std::stof(std::string(line.substr(xPos + 2, xEnd - xPos - 2)));
			}
			size_t yPos = line.find("y:");
			if (yPos != std::string::npos) {
				size_t yEnd = line.find(' ', yPos);
				*mouseY = std::stof(std::string(line.substr(yPos + 2, yEnd - yPos - 2)));
			}
		}
		else if (line.substr(0, 3) == "CF ") {
			*cameraForward = std::stoi(std::string(line.substr(3)));
		}
		else if (line.substr(0, 3) == "CS ") {
			*cameraStrafe = std::stoi(std::string(line.substr(3)));
		}
		else if (line.substr(0, 3) == "CL ") {
			*cameraLift = std::stoi(std::string(line.substr(3)));
		}
	}

}