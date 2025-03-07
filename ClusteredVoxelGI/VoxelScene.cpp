#include "DX12Lib/pch.h"
#include "VoxelScene.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "DX12Lib/Scene/CameraController.h"
#include "DX12Lib/Scene/RemoteNodeController.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightController.h"
#include "DX12Lib/Commons/ShadowMap.h"


using namespace DX12Lib;
using namespace CVGI;
using namespace Graphics;



void VoxelScene::Init(DX12Lib::GraphicsContext& context)
{

	auto lightNode = this->AddNode();
	lightNode->SetPosition(-1, 38, 0);
	auto light = lightNode->AddComponent<DX12Lib::LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.45f, 0.45f, 0.45f });
	light->GetShadowCamera()->SetShadowBufferDimensions(2048, 2048);

	lightNode->Rotate(lightNode->GetRight(), DirectX::XMConvertToRadians(90));
	
	lightNode->AddComponent<LightController>();
	lightNode->AddComponent<RemoteNodeController>();

	m_mainLight = light;

	m_camera->Node->AddComponent<CameraController>();
	m_camera->Node->AddComponent<RemoteNodeController>();

	auto* voxelCameraNode = this->AddNode();
	m_voxelCamera = voxelCameraNode->AddComponent<VoxelCamera>(VoxelTextureDimensions);

	std::string sourcePath = SOURCE_DIR;
	sourcePath += std::string("\\..\\Models\\PBR\\sponza2.gltf");

	bool loaded = this->AddFromFile(sourcePath.c_str());

	assert(loaded && "Model not loaded");


	Scene::Init(context);
}

void VoxelScene::Update(DX12Lib::GraphicsContext& context)
{
	Scene::Update(context);
}

void VoxelScene::Render(DX12Lib::GraphicsContext& context)
{
	Scene::Render(context);
}

void VoxelScene::OnResize(DX12Lib::GraphicsContext& context, int newWidth, int newHeight)
{
	Scene::OnResize(context, newWidth, newHeight);


}

void VoxelScene::OnClose(DX12Lib::GraphicsContext& context)
{
	Scene::OnClose(context);
}

void CVGI::VoxelScene::OnModelChildAdded(DX12Lib::SceneNode& modelNode, DX12Lib::MeshRenderer& meshRenderer, DX12Lib::ModelRenderer& modelRenderer)
{
	Scene::OnModelChildAdded(modelNode, meshRenderer, modelRenderer);
}

 