#include "DX12Lib/pch.h"
#include "DX12Lib/Encoder/FFmpegStreamer.h"
#include "LIScene.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/CameraController.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/LightController.h"

using namespace LI;
using namespace DX12Lib;
using namespace Graphics;

LI::LIScene::LIScene(bool shouldStream) : DX12Lib::Scene()
{

}

void LI::LIScene::Init(DX12Lib::GraphicsContext& context)
{
	Scene::Init(context);

	auto lightNode = this->AddNode();
	lightNode->SetPosition(-1, 38, 0);
	auto light = lightNode->AddComponent<DX12Lib::LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.45f, 0.45f, 0.45f });
	if (light->CastsShadows())
	{
		light->GetShadowCamera()->SetShadowBufferDimensions(2048, 2048);
	}


	lightNode->Rotate(lightNode->GetRight(), DirectX::XMConvertToRadians(90));

	lightNode->AddComponent<LightController>();

	m_cameraController = m_camera->Node->AddComponent<DX12Lib::CameraController>();

	m_mainLight = light;

}

void LI::LIScene::Update(DX12Lib::GraphicsContext& context)
{
	Scene::Update(context);
}

void LI::LIScene::Render(DX12Lib::GraphicsContext& context)
{

	Scene::Render(context);

}

void LI::LIScene::OnResize(DX12Lib::GraphicsContext& context, int width, int height)
{
	Scene::OnResize(context, width, height);
}

void LI::LIScene::OnClose(DX12Lib::GraphicsContext& context)
{
	Scene::OnClose(context);
}
