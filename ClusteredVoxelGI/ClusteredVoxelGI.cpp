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

	auto clusterVisibility = m_clusterVisibility->BuildFaceCountRootSignature();
	auto clusterVisibilityPso = m_clusterVisibility->BuildFaceCountPipelineState(clusterVisibility);

	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;
	Renderer::s_PSOs[compactBufferPSO->Name] = compactBufferPSO;
	Renderer::s_PSOs[clusterizeVoxelPso->Name] = clusterizeVoxelPso;
	Renderer::s_PSOs[clusterReducePso->Name] = clusterReducePso;
	Renderer::s_PSOs[clusterVisibilityPso->Name] = clusterVisibilityPso;



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

	UINT32 voxelCount = *m_voxelizeScene->GetBufferManager()->ReadFromBuffer<UINT32*>(commandContext, (UINT)BufferType::VoxelCounter);
	



	m_prefixSumVoxels->InitializeBuffers(commandContext);
	m_prefixSumVoxels->StartPrefixSum(m_voxelizeScene->GetBufferManager());

	// Voxelize scene temporary buffers can be deleted after the voxelization and prefix sum passes.
	m_voxelizeScene->DeleteTemporaryBuffers();
	// Prefix sum temporary buffers are only needed for the prefix sum pass.
	m_prefixSumVoxels->DeleteTemporaryBuffers();

	m_clusterVoxels->InitializeBuffers(voxelCount);
	m_clusterVoxels->StartClustering(*m_voxelizeScene->GetBufferManager(), *m_prefixSumVoxels->GetBufferManager());

	m_mergeClusters->InitializeBuffers(commandContext, *m_clusterVoxels);
	m_mergeClusters->StartMerging(*m_prefixSumVoxels->GetBufferManager());

	m_clusterVisibility->InitializeBuffers(voxelCount);
	m_clusterVisibility->StartVisibility(*m_prefixSumVoxels->GetBufferManager());

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

	m_voxelizeScene->DisplayVoxelPass(commandContext, m_Scene->GetMainCamera(), m_prefixSumVoxels->GetBufferManager(), m_mergeClusters->GetBufferManager(), m_clusterVisibility->GetBufferManager());

	//VoxelDisplayPass(commandContext);

	Renderer::PostDrawCleanup(commandContext);
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