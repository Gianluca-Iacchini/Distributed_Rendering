#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "LIScene.h"

using namespace LI;
using namespace DX12Lib;

void LI::LIScene::Init(DX12Lib::CommandContext& context)
{
	auto cameraChild = this->AddNode();
	m_camera = cameraChild->AddComponent<SceneCamera>();

	cameraChild->SetPosition(0, 3, 0);

	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 100, 0);
	auto light = lightNode->AddComponent<LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.6f, 0.6f, 0.6f });
	lightNode->Rotate(lightNode->GetRight(), 1.2f);
}
