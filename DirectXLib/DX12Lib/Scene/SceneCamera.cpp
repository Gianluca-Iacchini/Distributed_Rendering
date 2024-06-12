#include "DX12Lib/pch.h"
#include "SceneCamera.h"
#include "Mouse.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace DirectX;

void DX12Lib::SceneCamera::Init(CommandContext& context)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, ((float)Renderer::s_clientWidth / Renderer::s_clientHeight), 1.0f, 1000.0f);
}

void DX12Lib::SceneCamera::Update(CommandContext& context)
{
	auto input = this->Node->Scene.GetNetworkData();

	std::string inputData = std::string(input.data(), input.size());

	float mouseX = 0.0f, mouseY = 0.0f;

	ParseInputString(inputData, &mouseX, &mouseY, &m_cameraForward, &m_cameraStrafe, &m_cameraLift);



	auto& transform = this->Node->Transform;
	float deltaTime = GameTime::GetDeltaTime();



	auto kbState = s_kbTracker->GetLastState();

	if (s_kbTracker->IsKeyPressed(Keyboard::Escape))
		PostQuitMessage(0);


#if STREAMING
	float deltaX = mouseX * Renderer::s_clientWidth * deltaTime;
	float deltaY = mouseY * Renderer::s_clientHeight * deltaTime;

	this->Node->Translate(Node->GetForward(), m_cameraForward * deltaTime);
	this->Node->Translate(Node->GetRight(), m_cameraStrafe * deltaTime);
	this->Node->Translate({ 0.0f, 1.0f, 0.0f }, m_cameraLift * deltaTime);

	this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, deltaX);
	this->Node->Rotate(Node->GetRight(), deltaY);
#else

	auto state = s_mouse->GetState();
	if (state.positionMode == Mouse::MODE_RELATIVE)
	{
		auto rotation = this->Node->GetRotationEulerAngles();
		this->Node->Rotate(Node->GetRight(), state.y * deltaTime);
		this->Node->Rotate({ 0.0f, 1.0f, 0.0f }, state.x * deltaTime);

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
		this->Node->Translate({0.0f, 1.0f, 0.0f }, deltaTime);
	if (kbState.Q)
		this->Node->Translate({ 0.0f, 1.0f, 0.0f }, -deltaTime);
#endif

	if (Node->IsTransformDirty())
		Camera::UpdateViewMatrix(transform.GetWorldPosition(), transform.GetUp(), transform.GetForward(), transform.GetRight());
}

void DX12Lib::SceneCamera::Render(CommandContext& context)
{
	Renderer::AddMainCamera(this);
}

void DX12Lib::SceneCamera::OnResize(CommandContext& context, int newWidth, int newHeight)
{
	Camera::SetLens(0.25f * DirectX::XM_PI, ((float)newWidth / newHeight), 1.0f, 1000.0f);
}

void DX12Lib::SceneCamera::UseCamera(CommandContext& context)
{
	DirectX::XMMATRIX view = this->GetView();
	DirectX::XMMATRIX projection = this->GetProjection();
	DirectX::XMMATRIX viewProjection = view * projection;
	DirectX::XMMATRIX invView = DirectX::XMMatrixInverse(nullptr, view);
	DirectX::XMMATRIX invProjection = DirectX::XMMatrixInverse(nullptr, projection);
	DirectX::XMMATRIX invViewProjection = DirectX::XMMatrixInverse(nullptr, viewProjection);

	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.view, DirectX::XMMatrixTranspose(view));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invView, DirectX::XMMatrixTranspose(invView));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.projection, DirectX::XMMatrixTranspose(projection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invProjection, DirectX::XMMatrixTranspose(invProjection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.viewProjection, DirectX::XMMatrixTranspose(viewProjection));
	DirectX::XMStoreFloat4x4(&m_constantBufferCamera.invViewProjection, DirectX::XMMatrixTranspose(invViewProjection));
	m_constantBufferCamera.eyePosition = this->Node->GetPosition();
	m_constantBufferCamera.farPlane = this->m_farZ;
	m_constantBufferCamera.nearPlane = this->m_nearZ;

	auto cbCamera = Renderer::s_graphicsMemory->AllocateConstant(m_constantBufferCamera);

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
		(UINT)Renderer::RootSignatureSlot::CameraCBV,
		cbCamera.GpuAddress()
	);
}

void DX12Lib::SceneCamera::ParseInputString(std::string& input, float* mouseX, float* mouseY, int* cameraForward, int* cameraStrafe, int* cameraLift)
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
