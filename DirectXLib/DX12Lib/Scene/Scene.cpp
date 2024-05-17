#include "DX12Lib/pch.h"
#include "Scene.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "DX12Lib/Models/ModelRenderer.h"

using namespace DX12Lib;
using namespace Assimp;
using namespace Graphics;

Scene::Scene()
{
	m_rootNode = std::make_unique<SceneNode>();
	m_rootNode->Name = L"RootNode";
}

Scene::~Scene()
{

}

bool DX12Lib::Scene::AddFromFile(const std::wstring& filename)
{
	DXLIB_CORE_INFO(L"Loading model from file {0}", filename);

	Importer importer;

	std::string filenameStr = Utils::ToString(filename);

	importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80.0f);
	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);

	unsigned int preprocessFlags = aiProcess_ConvertToLeftHanded | aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_Triangulate;
	/*| aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes | aiProcess_GenBoundingBoxes;*/

	const aiScene* scene = importer.ReadFile(filenameStr, preprocessFlags);

	if (scene == nullptr)
	{
		DXLIB_CORE_WARN(L"Failed to load scene");
		return false;
	}

	std::wstring directoryPath = Utils::GetFileDirectory(filename);
	Utils::SetWorkingDirectory(directoryPath);

	std::shared_ptr<Model> model = std::make_shared<Model>();
	model->LoadFromFile(scene);


	auto modelNode = m_rootNode->AddChild();
	
	m_numNodes += 1;
	modelNode->Name = L"FirstChild";


	auto modelRenderer = modelNode->AddComponent<ModelRenderer>();
	modelRenderer->Model = model;

	Traverse(modelRenderer, scene->mRootNode, modelNode);

	Utils::SetWorkingDirectory(Utils::StartingWorkingDirectoryPath);

	DXLIB_CORE_INFO(L"Model loaded successfully");

	return true;
}



bool DX12Lib::Scene::AddFromFile(const wchar_t* filename)
{
    return this->AddFromFile(std::wstring(filename));
}

bool DX12Lib::Scene::AddFromFile(const char* filename)
{
	return this->AddFromFile(Utils::ToWstring(filename));
}

void Scene::Update(CommandContext* context)
{
	m_rootNode->Update(context);
}

void Scene::Render(CommandContext* context)
{
	m_rootNode->Render(context);

	Renderer::RenderLayers(context);
}

void DX12Lib::Scene::Draw(ID3D12GraphicsCommandList* cmdList)
{
	
}

void DX12Lib::Scene::Traverse(ModelRenderer* modelRenderer, aiNode* node, SceneNode* parentNode)
{
	assert(modelRenderer != nullptr);
	assert(node != nullptr);
	assert(parentNode != nullptr);

	SceneNode* firstChildren = nullptr;

	// Create one SceneNode for each mesh in the node. These mesh nodes are create as siblings of the original node
	// Since we now have multiple nodes instead of one, we need to elect one of them (firstChildren) as the parent of the
	// original node children

	Model* model = modelRenderer->Model.get();

	for (UINT i = 0; i < node->mNumMeshes; i++)
	{
		aiVector3D position, scaling;
		aiQuaternion rotation;

		auto child = parentNode->AddChild();
		child->Name = Utils::ToWstring(node->mName.C_Str());
		node->mTransformation.Decompose(scaling, rotation, position);
		child->SetRelativePosition({ position.x, position.y, position.z });
		child->SetRelativeRotation({ rotation.x, rotation.y, rotation.z, rotation.w });
		child->SetRelativeScale({ scaling.x, scaling.y, scaling.z });


		int meshIndex = node->mMeshes[i];
		auto mesh = model->GetMeshAt(meshIndex);
		SharedMaterial meshMat = model->GetMaterialAt(mesh->m_materialIndex);
		
		auto meshRenderer = child->AddComponent<MeshRenderer>(modelRenderer);
		meshRenderer->SetMesh(mesh);
		meshRenderer->SetMaterial(meshMat);
		
		// If there is at least one mesh in the node, set the firstChildren to the first mesh
		if (firstChildren == nullptr)
			firstChildren = child;
	}

	// If there are no meshes in the node but the node is not a leaft then create an empty node
	// to store the children 
	if (firstChildren == nullptr && node->mNumChildren > 0)
		firstChildren = parentNode->AddChild();

	for (UINT i = 0; i < node->mNumChildren; i++)
	{
		Traverse(modelRenderer, node->mChildren[i], firstChildren);
	}

}
