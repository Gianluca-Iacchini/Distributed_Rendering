#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "LIScene.h"
#include "DX12Lib/Commons/Logger.h"

using namespace LI;
using namespace DX12Lib;

void LI::LIScene::Init(DX12Lib::CommandContext& context)
{
	Scene::Init(context);

	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 100, 0);
	auto light = lightNode->AddComponent<LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.6f, 0.6f, 0.6f });
	lightNode->Rotate(lightNode->GetRight(), 1.2f);

}

void LI::LIScene::Update(DX12Lib::CommandContext& context)
{
	Scene::Update(context);
}

void LI::LIScene::Render(DX12Lib::CommandContext& context)
{
	Scene::Render(context);
}

void LI::LIScene::OnResize(DX12Lib::CommandContext& context, int width, int height)
{
	Scene::OnResize(context, width, height);
}
