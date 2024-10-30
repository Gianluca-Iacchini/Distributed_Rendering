#include "DX12Lib/pch.h"
#include "VoxelScene.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "CameraController.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "VoxelMaterial.h"


using namespace DX12Lib;
using namespace CVGI;
using namespace Graphics;



void VoxelScene::Init(DX12Lib::CommandContext& context)
{
	m_voxelTexture.Create3D(VoxelTextureDimensions.x, VoxelTextureDimensions.y, VoxelTextureDimensions.z, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 150, 0);
	auto light = lightNode->AddComponent<DX12Lib::LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.45f, 0.45f, 0.45f });
	lightNode->Rotate(lightNode->GetRight(), DirectX::XMConvertToRadians(90));

	m_camera->Node->AddComponent<CameraController>();

	auto* voxelCameraNode = this->AddNode();
	m_voxelCamera = voxelCameraNode->AddComponent<VoxelCamera>(VoxelTextureDimensions);

	std::string sourcePath = SOURCE_DIR;
	sourcePath += std::string("\\..\\LocalIllumination\\Models\\PBR\\sponza2.gltf");

	bool loaded = this->AddFromFile(sourcePath.c_str());

	assert(loaded && "Model not loaded");


	Scene::Init(context);
}

void VoxelScene::Update(DX12Lib::CommandContext& context)
{
	Scene::Update(context);
}

void VoxelScene::Render(DX12Lib::CommandContext& context)
{
	Scene::Render(context);
}

void VoxelScene::OnResize(DX12Lib::CommandContext& context, int newWidth, int newHeight)
{
	Scene::OnResize(context, newWidth, newHeight);


}

void VoxelScene::OnClose(DX12Lib::CommandContext& context)
{
	Scene::OnClose(context);
}

void CVGI::VoxelScene::OnModelChildAdded(DX12Lib::SceneNode& modelNode, DX12Lib::MeshRenderer& meshRenderer, DX12Lib::ModelRenderer& modelRenderer)
{
	Scene::OnModelChildAdded(modelNode, meshRenderer, modelRenderer);
}

 