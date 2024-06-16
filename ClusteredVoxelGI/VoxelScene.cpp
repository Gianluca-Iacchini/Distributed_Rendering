#include "VoxelScene.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "CameraController.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Models/Material.h"

using namespace CVGI;

void VoxelScene::Init(DX12Lib::CommandContext& context)
{
	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 100, 0);
	auto light = lightNode->AddComponent<DX12Lib::LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.6f, 0.6f, 0.6f });
	lightNode->Rotate(lightNode->GetRight(), 1.2f);

	m_camera->Node->AddComponent<CameraController>();

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
	//std::shared_ptr<DX12Lib::Material> defMat = Graphics::Renderer::s_materialManager->GetMaterial(Graphics::Renderer::s_materialManager->PBR_DEFAULT);
	//meshRenderer.SetMaterial(defMat);
}
