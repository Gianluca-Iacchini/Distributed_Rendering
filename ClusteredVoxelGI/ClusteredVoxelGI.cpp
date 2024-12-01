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

	m_rtgiFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	m_rasterFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	m_blockFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	m_shadowFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);

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
	m_computeNeighboursTechnique = std::make_unique<ComputeNeighboursTechnique>(m_data);
	m_clusterVisibility = std::make_unique<ClusterVisibility>(m_data);
	m_buildAABBsTechnique = std::make_unique<BuildAABBsTechnique>(m_data);
	m_facePenaltyTechnique = std::make_unique<FacePenaltyTechnique>(m_data);
	m_sceneDepthTechnique = std::make_unique<SceneDepthTechnique>(m_data);
	m_lightVoxel = std::make_unique<LightVoxel>(m_data);
	m_lightTransportTechnique = std::make_unique<LightTransportTechnique>(m_data);
	m_gaussianFilterTechnique = std::make_unique<GaussianFilterTechnique>(m_data);
	m_lerpRadianceTechnique = std::make_unique<LerpRadianceTechnique>(m_data);



	auto voxelScenePSO = m_voxelizeScene->BuildPipelineState(); 
	auto voxelDisplayPSO = m_displayVoxelScene->BuildPipelineState();


	auto compactBufferPSO = m_prefixSumVoxels->BuildPipelineState();

	auto clusterizeVoxelPso = m_clusterVoxels->BuildPipelineState(); 

	auto clusterReduceRootSignature = m_mergeClusters->BuildMergeClustersRootSignature(); 
	
	auto clusterReducePso = m_mergeClusters->BuildMergeClustersPipelineState(clusterReduceRootSignature); 
	
	auto computeNeighboursPso = m_computeNeighboursTechnique->BuildPipelineState();

	auto aabbGenerationPso = m_buildAABBsTechnique->BuildPipelineState();

	auto facePenaltyPso = m_facePenaltyTechnique->BuildPipelineState();

	auto raytracePso = m_clusterVisibility->BuildPipelineState();

	auto lightVoxelPso = m_lightVoxel->BuildPipelineState();

	auto lightTransportPso = m_lightTransportTechnique->BuildPipelineState();

	auto gaussianFilterPso = m_gaussianFilterTechnique->BuildPipelineState();

	auto lerpRadiancePso = m_lerpRadianceTechnique->BuildPipelineState();

	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;
	Renderer::s_PSOs[compactBufferPSO->Name] = compactBufferPSO;
	Renderer::s_PSOs[clusterizeVoxelPso->Name] = clusterizeVoxelPso;
	Renderer::s_PSOs[clusterReducePso->Name] = clusterReducePso;
	Renderer::s_PSOs[computeNeighboursPso->Name] = computeNeighboursPso;
	Renderer::s_PSOs[aabbGenerationPso->Name] = aabbGenerationPso;
	Renderer::s_PSOs[facePenaltyPso->Name] = facePenaltyPso;
	Renderer::s_PSOs[raytracePso->Name] = raytracePso;
	Renderer::s_PSOs[lightVoxelPso->Name] = lightVoxelPso;
	Renderer::s_PSOs[lightTransportPso->Name] = lightTransportPso;
	Renderer::s_PSOs[gaussianFilterPso->Name] = gaussianFilterPso;
	Renderer::s_PSOs[lerpRadiancePso->Name] = lerpRadiancePso;


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

	DirectX::XMFLOAT3 originalMin = sceneBounds.Min;
	DirectX::XMFLOAT3 originalMax = sceneBounds.Max;

	float minComponent = std::min(sceneBounds.Min.x, std::min(sceneBounds.Min.y, sceneBounds.Min.z));
	float maxComponent = std::max(sceneBounds.Max.x, std::max(sceneBounds.Max.y, sceneBounds.Max.z));

	float extent = maxComponent - minComponent;

	sceneBounds.Min.x = minComponent;
	sceneBounds.Min.y = minComponent;
	sceneBounds.Min.z = minComponent;

	sceneBounds.Max.x = maxComponent;
	sceneBounds.Max.y = maxComponent;
	sceneBounds.Max.z = maxComponent;

	voxelCellSize.x = (sceneBounds.Max.x - sceneBounds.Min.x) / voxelTexDimensions.x;
	voxelCellSize.y = (sceneBounds.Max.y - sceneBounds.Min.y) / voxelTexDimensions.y;
	voxelCellSize.z = (sceneBounds.Max.z - sceneBounds.Min.z) / voxelTexDimensions.z;

	m_data->SetVoxelCellSize(voxelCellSize);
	m_data->SetSceneAABB(sceneBounds);
	m_data->SetCamera(m_Scene->GetMainCamera());

	m_data->BuildMatrices();

	DXLIB_CORE_INFO("Scene bounds found at: Min: {0} {1} {2}; Max: {3} {4} {5}",
				sceneBounds.Min.x, sceneBounds.Min.y, sceneBounds.Min.z,
				sceneBounds.Max.x, sceneBounds.Max.y, sceneBounds.Max.z);

	voxelScene->Render(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();
	voxelCamera->SetOrthogonalHalfExtents(DirectX::XMFLOAT3(extent / 2.0f, extent / 2.0f, extent / 2.0f));

	m_voxelizeScene->SetVoxelCamera(voxelCamera);
	m_voxelizeScene->PerformTechnique(commandContext);

	bool foundLightComponent = false;
	for (UINT i = 0; i < childCount; i++)
	{
		auto* child = rootNode->GetChildAt(i);

		auto lightComponent = child->GetComponent<LightComponent>();

		if (lightComponent != nullptr)
		{
			m_lightVoxel->SetLightComponent(lightComponent);
			m_data->SetLightComponent(lightComponent);
			foundLightComponent = true;
			break;
		}
	}

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
	
	m_computeNeighboursTechnique->InitializeBuffers();
	m_computeNeighboursTechnique->PerformTechnique(computeContext);

	m_data->FaceCount = m_data->GetVoxelCount() * 6;

	m_buildAABBsTechnique->InitializeBuffers();
	m_buildAABBsTechnique->PerformTechnique(computeContext);

	m_clusterVisibility->InitializeBuffers();
	m_data->SetTlas(m_clusterVisibility->BuildAccelerationStructures(computeContext));

	m_clusterVisibility->PerformTechnique(computeContext);

	m_facePenaltyTechnique->InitializeBuffers();
	m_facePenaltyTechnique->PerformTechnique(computeContext);

	m_sceneDepthTechnique->InitializeBuffers();

	m_lightVoxel->InitializeBuffers();

	m_lightTransportTechnique->InitializeBuffers();


	m_gaussianFilterTechnique->InitializeBuffers();
	m_gaussianFilterTechnique->SetIndirectCommandSignature(m_lightTransportTechnique->GetIndirectCommandSignature());
	m_gaussianFilterTechnique->InitializeGaussianConstants(computeContext);

	m_lerpRadianceTechnique->InitializeBuffers();

	assert(foundLightComponent && "Failed to find light component with shadows enabled.");


	computeContext.Finish(true);

	m_displayVoxelScene->SetCamera(m_Scene->GetMainCamera());
	m_displayVoxelScene->SetVertexData(commandContext);

	Renderer::SetRTGIData(m_data, originalMin, originalMax);
	Renderer::UseRTGI(true);

	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);
	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_blockFence);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_shadowFence);
	m_lastBlockFenceVal = m_blockFence->CurrentFenceValue;

	//Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_swapFence);


	m_rtgiFence->Get()->SetName(L"RTGI Fence");
	m_rasterFence->Get()->SetName(L"Acc Fence");
	m_blockFence->Get()->SetName(L"Block Fence");
	m_shadowFence->Get()->SetName(L"ShadowFence Fence");

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Update(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::Update(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Draw(DX12Lib::GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	auto kbState = Graphics::s_keyboard->GetState();

	bool didLightChange = m_data->GetLightComponent()->Node->IsTransformDirty();
	bool didCameraMove = m_data->GetCamera()->IsDirty();

	m_Scene->Render(commandContext);

	if (!LightDispatched && (didLightChange || didCameraMove))
	{
		DX12Lib::ComputeContext& context = ComputeContext::Begin();
		m_sceneDepthTechnique->PerformTechnique(commandContext);
		commandContext.Flush(true);

		if (didLightChange)
		{
			m_lightVoxel->PerformTechnique(context);
			m_lightTransportTechnique->ResetRadianceBuffers(true);
		}
		else
		{
			m_gaussianFilterTechnique->CopyBufferData(context);
		}

		m_lightTransportTechnique->PerformTechnique(context);

		context.Finish();

		LightDispatched = true;

		m_resetTime = didLightChange;
		m_resetCamera = didCameraMove;

		m_rtgiFence->CurrentFenceValue++;
		Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);
	}

	if (LightDispatched && m_rtgiFence->IsFenceComplete(m_rtgiFence->CurrentFenceValue))
	{
		DX12Lib::ComputeContext& context = ComputeContext::Begin();
		if (IndirectBlockCount < 48)
		{
			UINT32 fenceCompletedValue = m_blockFence->GetGPUFenceValue();

			UINT32 dispatchesInFlight = m_blockFence->CurrentFenceValue - fenceCompletedValue;
			UINT32 dispatchesToLaunch = maxDispatchesPerFrame - dispatchesInFlight;

			for (UINT32 i = 0; i < dispatchesToLaunch; i++)
			{
				if (IndirectBlockCount < 16)
				{
					m_lightTransportTechnique->LaunchIndirectLightBlock(context, IndirectBlockCount);
					IndirectBlockCount += 1;
				}
				else if (IndirectBlockCount < 32)
				{
					m_gaussianFilterTechnique->SetGaussianBlock(IndirectBlockCount - 16);
					m_gaussianFilterTechnique->PerformTechnique(context);
					IndirectBlockCount += 1;
				}
				else if (IndirectBlockCount < 48)
				{
					m_gaussianFilterTechnique->SetGaussianBlock(IndirectBlockCount - 32);
					m_gaussianFilterTechnique->PerformTechnique2(context);
					IndirectBlockCount += 1;
				}

				context.Flush();

				m_blockFence->CurrentFenceValue++;
				Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_blockFence);
			}
		}
		else if (IndirectBlockCount == 48 && m_blockFence->IsFenceComplete(m_blockFence->CurrentFenceValue))
		{
			m_gaussianFilterTechnique->SwapBuffers();
			IndirectBlockCount = 0;
			LightDispatched = false;
			m_gaussianFilterTechnique->TransferRadianceData(context);
			if (m_resetTime)
			{
				m_lastTotalTime = RTGIUpdateDelta;
				RTGIUpdateDelta = 0.0f;
				m_resetTime = false;
			}
			changeLerp = true;
		}

		context.Finish();
	}

	m_rasterFence->WaitForCurrentFence();
	{
		if (lerpDelta > m_lastTotalTime && changeLerp)
		{
			lerpDelta = 0.0f;
			m_lerpRadianceTechnique->SetPhase(1);
			changeLerp = false;
		}

		DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();

		float totTime = m_lastTotalTime;
		float deltTime = lerpDelta;

		if (m_resetCamera)
		{
			totTime = 1.0f;
			deltTime = 1.0f;
			m_resetCamera = false;
		}

		m_lerpRadianceTechnique->SetMaxTime(totTime);
		m_lerpRadianceTechnique->SetAccumulatedTime(deltTime);
		m_lerpRadianceTechnique->PerformTechnique(context);
		context.Finish(true);
	}

	lerpDelta += GameTime::GetDeltaTime();

	Renderer::ShadowPass(commandContext);
	Renderer::MainRenderPass(commandContext);
	Renderer::DeferredPass(commandContext);
	Renderer::PostProcessPass(commandContext);

	//m_displayVoxelScene->PerformTechnique(commandContext);

	commandContext.Flush();
	m_rasterFence->CurrentFenceValue++;
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);

	m_sceneDepthTechnique->UpdateCameraMatrices();

	RTGIUpdateDelta += GameTime::GetDeltaTime();



	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::OnClose(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::OnClose(commandContext);
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