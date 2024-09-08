#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"



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

	m_voxelizeScene = std::make_unique<VoxelizeScene>(voxelTexDimensions, voxelCellSize);
	m_prefixSumVoxels = std::make_unique<PrefixSumVoxels>(voxelTexDimensions);
	m_clusterVoxels = std::make_unique<ClusterVoxels>(voxelTexDimensions);
	m_mergeClusters = std::make_unique<MergeClusters>(voxelTexDimensions);
	m_clusterVisibility = std::make_unique<ClusterVisibility>(voxelTexDimensions);



	auto voxelSceneRootSig = m_voxelizeScene->BuildVoxelizeSceneRootSignature();
	auto voxelScenePSO = m_voxelizeScene->BuildVoxelizeScenePso(voxelSceneRootSig); 

	auto voxelDisplayRootSig = m_voxelizeScene->BuildDisplayVoxelRootSignature(); 
	auto voxelDisplayPSO = m_voxelizeScene->BuildDisplayVoxelPso(voxelDisplayRootSig);

	auto compactBufferRootSig = m_prefixSumVoxels->BuildPrefixSumRootSignature(); 
	auto compactBufferPSO = m_prefixSumVoxels->BuildPrefixSumPipelineState(compactBufferRootSig); 

	auto clusterizeRootSignature = m_clusterVoxels->BuildClusterizeRootSignature(); 
	auto clusterizeVoxelPso = m_clusterVoxels->BuildClusterizePipelineState(clusterizeRootSignature); 

	auto clusterReduceRootSignature = m_mergeClusters->BuildMergeClustersRootSignature(); 
	auto clusterReducePso = m_mergeClusters->BuildMergeClustersPipelineState(clusterReduceRootSignature); 

	auto faceCountRootSignature = m_clusterVisibility->BuildFaceCountRootSignature();
	auto faceCountPso = m_clusterVisibility->BuildFaceCountPipelineState(faceCountRootSignature);

	auto aabbGenerationRootSignature = m_clusterVisibility->BuildAABBGenerationRootSignature();
	auto aabbGenerationPso = m_clusterVisibility->BuildAABBGenerationPipelineState(aabbGenerationRootSignature);

	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;
	Renderer::s_PSOs[compactBufferPSO->Name] = compactBufferPSO;
	Renderer::s_PSOs[clusterizeVoxelPso->Name] = clusterizeVoxelPso;
	Renderer::s_PSOs[clusterReducePso->Name] = clusterReducePso;
	Renderer::s_PSOs[faceCountPso->Name] = faceCountPso;
	Renderer::s_PSOs[aabbGenerationPso->Name] = aabbGenerationPso;




	m_voxelizeScene->InitializeBuffers();

	D3DApp::Initialize(commandContext);

	Renderer::SetUpRenderFrame(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();
	voxelScene->Render(commandContext);
	if (voxelCamera != nullptr)
	{
		m_voxelizeScene->VoxelizePass(commandContext, voxelCamera);
	}


	m_voxelizeScene->UpdateBuffers(commandContext);

	if (voxelCamera != nullptr)
	{
		m_voxelizeScene->VoxelizePass(commandContext, voxelCamera);
	}

	UINT32 voxelCount = *m_voxelizeScene->GetBufferManager()->ReadFromBuffer<UINT32*>(commandContext, (UINT)VoxelizeScene::VoxelBufferType::VoxelCounter);
	

	ComputeContext& computeContext = ComputeContext::Begin();

	m_prefixSumVoxels->InitializeBuffers(commandContext);
	m_prefixSumVoxels->StartPrefixSum(computeContext, m_voxelizeScene->GetBufferManager());

	// Voxelize scene temporary buffers can be deleted after the voxelization and prefix sum passes.
	m_voxelizeScene->DeleteTemporaryBuffers();
	// Prefix sum temporary buffers are only needed for the prefix sum pass.
	m_prefixSumVoxels->DeleteTemporaryBuffers();

	m_clusterVoxels->InitializeBuffers(voxelCount);
	m_clusterVoxels->StartClustering(computeContext, *m_voxelizeScene->GetBufferManager(), *m_prefixSumVoxels->GetBufferManager());

	m_mergeClusters->InitializeBuffers(commandContext, *m_clusterVoxels);
	m_mergeClusters->StartMerging(computeContext, *m_prefixSumVoxels->GetBufferManager());
	
	m_clusterVisibility->InitializeBuffers(voxelCount, m_mergeClusters->GetClusterCount());
	m_clusterVisibility->StartVisibility(computeContext, *m_prefixSumVoxels->GetBufferManager());

	m_clusterVisibility->StartAABBGeneration(computeContext, *m_prefixSumVoxels->GetBufferManager(), *m_mergeClusters->GetBufferManager());

	m_clusterVisibility->BuildAccelerationStructures(computeContext);

	computeContext.Finish(true);

	m_voxelizeScene->SetVertexData(commandContext, m_clusterVisibility->GetFaceCount());

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Update(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::Update(commandContext);

	m_cbVoxelCommons.totalTime = GameTime::GetTotalTime();
	m_cbVoxelCommons.deltaTime = GameTime::GetDeltaTime();

	m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);
}

void CVGI::ClusteredVoxelGIApp::Draw(DX12Lib::GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	m_voxelizeScene->DisplayVoxelPass(commandContext, m_Scene->GetMainCamera(), m_prefixSumVoxels->GetBufferManager(), m_mergeClusters->GetBufferManager(), m_clusterVisibility->GetFaceBufferManager());

	//VoxelDisplayPass(commandContext);

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