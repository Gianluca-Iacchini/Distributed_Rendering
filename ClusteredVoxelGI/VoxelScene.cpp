#include "DX12Lib/pch.h"
#include "VoxelScene.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "CameraController.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "VoxelMaterial.h"
#include "VoxelCamera.h"

using namespace DX12Lib;
using namespace CVGI;
using namespace Graphics;


void CVGI::VoxelScene::OnAppStart(DX12Lib::CommandContext& context)
{
	DX12Lib::ColorBuffer colorBuffer;

	colorBuffer.Create3D(256, 256, 256, 1, DXGI_FORMAT_R8G8B8A8_UNORM);

	Renderer::s_PSOs[L"PSO_VOXEL"] = BuildVoxelPSO();

	std::shared_ptr<VoxelMaterial> mat = std::make_shared<VoxelMaterial>();
	mat->SetName(L"VoxelMaterial");

	Renderer::s_materialManager->AddMaterial(mat);
}

void VoxelScene::Init(DX12Lib::CommandContext& context)
{
	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 100, 0);
	auto light = lightNode->AddComponent<DX12Lib::LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.6f, 0.6f, 0.6f });
	lightNode->Rotate(lightNode->GetRight(), 1.2f);


	auto* cameraNode = m_camera->Node;

	cameraNode->RemoveComponent(m_camera);

	m_camera = cameraNode->AddComponent<VoxelCamera>();
	cameraNode->AddComponent<CameraController>();


	std::string sourcePath = SOURCE_DIR;
	sourcePath += std::string("\\..\\LocalIllumination\\Models\\PBR\\sponza2.gltf");

	bool loaded = this->AddFromFile(sourcePath.c_str());

	assert(loaded && "Model not loaded");

	//m_camera->SetProjectionType(true);
	//m_camera->SetOrthogonalBounds(100, 100, 0.1f, 100);




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
	//Scene::OnModelChildAdded(modelNode, meshRenderer, modelRenderer);
	std::shared_ptr<DX12Lib::Material> voxelMat = Graphics::Renderer::s_materialManager->GetMaterial(L"VoxelMaterial");
	meshRenderer.SetMaterial(voxelMat);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelScene::BuildVoxelRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> baseRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)VoxelRootParameter::Count, 1);
	baseRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*baseRootSignature)[(UINT)VoxelRootParameter::CommonCBV].InitAsConstantBuffer(0);
	(*baseRootSignature)[(UINT)VoxelRootParameter::ObjectCBV].InitAsConstantBuffer(1);
	(*baseRootSignature)[(UINT)VoxelRootParameter::VoxelDataCBV].InitAsConstantBuffer(2);
	(*baseRootSignature)[(UINT)VoxelRootParameter::LightSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*baseRootSignature)[(UINT)VoxelRootParameter::MaterialSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_PIXEL, 1);
	(*baseRootSignature)[(UINT)VoxelRootParameter::VoxelTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
	(*baseRootSignature)[(UINT)VoxelRootParameter::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, NUM_PHONG_TEXTURES);
	baseRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return baseRootSignature;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::VoxelScene::BuildVoxelPSO()
{
	auto rootSignature = BuildVoxelRootSignature();

	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring vertexShaderPath = shaderPath + L"\\Voxel_VS.hlsl";
	std::wstring geometryShaderPath = shaderPath + L"\\Voxel_GS.hlsl";
	std::wstring pixelShaderPath = shaderPath + L"\\Voxel_PS.hlsl";

	std::shared_ptr<Shader> vertexShader = std::make_shared<Shader>(vertexShaderPath, "VS", "vs_5_1");
	std::shared_ptr<Shader> geometryShader = std::make_shared<Shader>(geometryShaderPath, "GS", "gs_5_1");
	std::shared_ptr<Shader> pixelShader = std::make_shared<Shader>(pixelShaderPath, "PS", "ps_5_1");

	vertexShader->Compile();
	geometryShader->Compile();
	pixelShader->Compile();

	std::shared_ptr<PipelineState> voxelPSO = std::make_shared<PipelineState>();
	voxelPSO->InitializeDefaultStates();
	voxelPSO->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
		DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
	voxelPSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	voxelPSO->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
	voxelPSO->SetShader(vertexShader, ShaderType::Vertex);
	//voxelPSO->SetShader(geometryShader, ShaderType::Geometry);
	voxelPSO->SetShader(pixelShader, ShaderType::Pixel);
	voxelPSO->SetRootSignature(rootSignature);
	voxelPSO->Finalize();

	return voxelPSO;
}


 