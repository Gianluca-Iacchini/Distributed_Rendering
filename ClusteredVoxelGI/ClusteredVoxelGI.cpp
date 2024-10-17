#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/Commons/ShadowMap.h"



using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

using namespace CVGI;



void ClusteredVoxelGIApp::Initialize(GraphicsContext& commandContext)
{

	if (!IsDirectXRaytracingSupported() || Graphics::s_device->GetDXRComPtr() == nullptr)
	{
		throw std::exception("DirectX Raytracing is not supported by your GPU.");
	}

	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);

	VoxelScene* voxelScene = static_cast<VoxelScene*>(this->m_Scene.get());

	DirectX::XMUINT3 voxelTexDimensions = DirectX::XMUINT3(
		VoxelTextureDimension.x,
		VoxelTextureDimension.y,
		VoxelTextureDimension.z);

	m_data = std::make_shared<TechniqueData>();
	m_data->SetVoxelGridSize(voxelTexDimensions);

	m_voxelizeScene = std::make_unique<VoxelizeScene>(m_data);
	m_displayVoxelScene = std::make_unique<DisplayVoxelScene>(m_data);
	m_prefixSumVoxels = std::make_unique<PrefixSumVoxels>(m_data);
	m_clusterVoxels = std::make_unique<ClusterVoxels>(m_data);
	m_mergeClusters = std::make_unique<MergeClusters>(voxelTexDimensions);
	m_clusterVisibility = std::make_unique<ClusterVisibility>(m_data);
	m_faceCountTechnique = std::make_unique<FaceCountTechnique>(m_data);
	m_buildAABBsTechnique = std::make_unique<BuildAABBsTechnique>(m_data);
	m_lightVoxel = std::make_unique<LightVoxel>(m_data);
	m_lightTransportTechnique = std::make_unique<LightTransportTechnique>(m_data);



	auto voxelScenePSO = m_voxelizeScene->BuildPipelineState(); 
	auto voxelDisplayPSO = m_displayVoxelScene->BuildPipelineState();


	auto compactBufferPSO = m_prefixSumVoxels->BuildPipelineState();

	auto clusterizeVoxelPso = m_clusterVoxels->BuildPipelineState(); 

	auto clusterReduceRootSignature = m_mergeClusters->BuildMergeClustersRootSignature(); 
	auto clusterReducePso = m_mergeClusters->BuildMergeClustersPipelineState(clusterReduceRootSignature); 


	auto faceCountPso = m_faceCountTechnique->BuildPipelineState();

	auto aabbGenerationPso = m_buildAABBsTechnique->BuildPipelineState();

	auto raytracePso = m_clusterVisibility->BuildPipelineState();

	auto lightVoxelPso = m_lightVoxel->BuildPipelineState();

	auto lightTransportPso = m_lightTransportTechnique->BuildPipelineState();

	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;
	Renderer::s_PSOs[compactBufferPSO->Name] = compactBufferPSO;
	Renderer::s_PSOs[clusterizeVoxelPso->Name] = clusterizeVoxelPso;
	Renderer::s_PSOs[clusterReducePso->Name] = clusterReducePso;
	Renderer::s_PSOs[faceCountPso->Name] = faceCountPso;
	Renderer::s_PSOs[aabbGenerationPso->Name] = aabbGenerationPso;
	Renderer::s_PSOs[raytracePso->Name] = raytracePso;
	Renderer::s_PSOs[lightVoxelPso->Name] = lightVoxelPso;
	Renderer::s_PSOs[lightTransportPso->Name] = lightTransportPso;


	m_voxelizeScene->InitializeBuffers();

	D3DApp::Initialize(commandContext);

	Renderer::SetUpRenderFrame(commandContext);

	auto* rootNode = m_Scene->GetRootNode();
	UINT childCount = rootNode->GetChildCount();

	DX12Lib::AABB sceneBounds;

	for (UINT i = 0; i < childCount; i++)
	{
		auto* child = rootNode->GetChildAt(i);

		auto* renderer = child->GetComponent<ModelRenderer>();

		if (renderer != nullptr)
		{
			sceneBounds = renderer->Model->GetBounds();
		}
	}

	float minComponent = std::min(sceneBounds.Min.x, std::min(sceneBounds.Min.y, sceneBounds.Min.z));
	float maxComponent = std::max(sceneBounds.Max.x, std::max(sceneBounds.Max.y, sceneBounds.Max.z));

	float extent = maxComponent - minComponent;

	sceneBounds.Min.x = minComponent;
	sceneBounds.Min.y = minComponent;
	sceneBounds.Min.z = minComponent;

	sceneBounds.Max.x = maxComponent;
	sceneBounds.Max.y = maxComponent;
	sceneBounds.Max.z = maxComponent;

	voxelCellSize.x = extent / voxelTexDimensions.x;
	voxelCellSize.y = extent / voxelTexDimensions.y;
	voxelCellSize.z = extent / voxelTexDimensions.z;

	m_data->SetVoxelCellSize(voxelCellSize);
	m_data->SetSceneAABB(sceneBounds);
	m_data->SetCamera(m_Scene->GetMainCamera());

	m_data->BuildMatrices();

	DXLIB_CORE_INFO("Scene bounds found at: Min: {0} {1} {2}; Max: {3} {4} {5}",
				sceneBounds.Min.x, sceneBounds.Min.y, sceneBounds.Min.z,
				sceneBounds.Max.x, sceneBounds.Max.y, sceneBounds.Max.z);

	voxelScene->Render(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();

	m_voxelizeScene->SetVoxelCamera(voxelCamera);
	m_voxelizeScene->PerformTechnique(commandContext);

	commandContext.Flush(true);

	ComputeContext& computeContext = ComputeContext::Begin();

	m_prefixSumVoxels->InitializeBuffers(computeContext);
	m_prefixSumVoxels->PerformTechnique(computeContext);


	// Voxelize scene temporary buffers can be deleted after the voxelization and prefix sum passes.
	m_voxelizeScene->DeleteTemporaryBuffers();
	// Prefix sum temporary buffers are only needed for the prefix sum pass.
	m_prefixSumVoxels->DeleteTemporaryBuffers();

	m_clusterVoxels->InitializeBuffers();
	m_clusterVoxels->PerformTechnique(computeContext);


	//m_mergeClusters->InitializeBuffers(commandContext, *m_clusterVoxels);
	//m_mergeClusters->StartMerging(computeContext, *m_prefixSumVoxels->GetBufferManager());
	
	m_faceCountTechnique->InitializeBuffers();
	m_faceCountTechnique->PerformTechnique(computeContext);

	m_buildAABBsTechnique->InitializeBuffers();
	m_buildAABBsTechnique->PerformTechnique(computeContext);

	m_clusterVisibility->InitializeBuffers();
	m_data->SetTlas(m_clusterVisibility->BuildAccelerationStructures(computeContext));

	m_clusterVisibility->PerformTechnique(computeContext);

	m_lightTransportTechnique->InitializeBuffers();

	m_lightVoxel->InitializeBuffers();

	bool foundLightComponent = false;
	for (UINT i = 0; i < childCount; i++)
	{
		auto* child = rootNode->GetChildAt(i);

		auto lightComponent = child->GetComponent<LightComponent>();

		if (lightComponent != nullptr)
		{
			m_lightVoxel->SetLightComponent(lightComponent);
			foundLightComponent = true;
			break;
		}
	}

	assert(foundLightComponent && "Failed to find light component with shadows enabled.");


	computeContext.Finish(true);

	m_displayVoxelScene->SetCamera(m_Scene->GetMainCamera());
	m_displayVoxelScene->SetVertexData(commandContext);

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Update(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::Update(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Draw(DX12Lib::GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	RayTracingContext& context = RayTracingContext::Begin();
	m_lightVoxel->PerformTechnique(context);

	m_lightTransportTechnique->PerformTechnique(context);

	context.Finish();

	m_displayVoxelScene->PerformTechnique(commandContext);

	Renderer::PostDrawCleanup(commandContext);
}

bool CVGI::ClusteredVoxelGIApp::IsDirectXRaytracingSupported() const
{
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 featureSupport = {};

	if (FAILED(Graphics::s_device->Get()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &featureSupport, sizeof(featureSupport))))
		return false;

	return featureSupport.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}




	




int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		CVGI::ClusteredVoxelGIApp app(hInstance, new CVGI::VoxelScene());

		if (!app.InitializeApp())
			return 0;

		return app.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}