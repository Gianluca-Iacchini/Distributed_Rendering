#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "CameraController.h"
#include "DX12Lib/Scene/LightController.h"
#include "DX12Lib/Commons/DX12Window.h"
#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"



using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

using namespace CVGI;
using namespace VOX;


void ClusteredVoxelGIApp::Initialize(GraphicsContext& commandContext)
{

	if (!IsDirectXRaytracingSupported() || Graphics::s_device->GetDXRComPtr() == nullptr)
	{
		throw std::exception("DirectX Raytracing is not supported by your GPU.");
	}


	D3DApp::Initialize(commandContext);

	m_rtgiFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	m_rasterFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	
	m_timingQueryHandle = Graphics::s_queryHeap->Alloc(16);
	m_timingReadBackBuffer.Create(16, sizeof(UINT64));


	m_data = std::make_shared<VOX::TechniqueData>();


	m_voxelizeScene = std::make_unique<VoxelizeScene>(m_data);
	m_displayVoxelScene = std::make_unique<DisplayVoxelScene>(m_data);
	m_prefixSumVoxels = std::make_unique<PrefixSumVoxels>(m_data);
	m_clusterVoxels = std::make_unique<ClusterVoxels>(m_data);
	m_computeNeighboursTechnique = std::make_unique<ComputeNeighboursTechnique>(m_data);
	m_clusterVisibility = std::make_unique<ClusterVisibility>(m_data);
	m_buildAABBsTechnique = std::make_unique<BuildAABBsTechnique>(m_data);
	m_sceneDepthTechnique = std::make_unique<SceneDepthTechnique>(m_data);
	m_lightVoxel = std::make_unique<LightVoxel>(m_data);
	m_lightTransportTechnique = std::make_unique<LightTransportTechnique>(m_data);
	m_gaussianFilterTechnique = std::make_unique<GaussianFilterTechnique>(m_data);


	m_voxelizeScene->BuildPipelineState();
	m_displayVoxelScene->BuildPipelineState();
	m_prefixSumVoxels->BuildPipelineState();
	m_clusterVoxels->BuildPipelineState();
	m_computeNeighboursTechnique->BuildPipelineState();
	m_clusterVisibility->BuildPipelineState();
	m_buildAABBsTechnique->BuildPipelineState();
	m_lightVoxel->BuildPipelineState();
	m_lightTransportTechnique->BuildPipelineState();
	m_gaussianFilterTechnique->BuildPipelineState();
}

void CVGI::ClusteredVoxelGIApp::Update(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::Update(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Draw(DX12Lib::GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	if (!ShowIMGUIWindow(commandContext))
	{
		Renderer::UIPass(commandContext, true);
		Renderer::PostDrawCleanup(commandContext);
		return;
	}




	auto kbState = Graphics::s_keyboard->GetState();

	bool didLightChange = 
		m_data->GetLightComponent()->Node->IsTransformDirty() ||
		m_data->GetLightComponent()->DidLightPropertyChange() ||
		m_lightTransportTechnique->DidRadianceStrengthChange();

	bool didCameraMove = m_data->GetCamera()->IsDirty();

	m_cameraMovedSinceLastUpdate |= didCameraMove;
	m_lightChangedSinceLastUpdate |= didLightChange;

	bool shouldUpdateLight = m_lightChangedSinceLastUpdate && RTGIUpdateDelta > 0.15f;

	m_Scene->Render(commandContext);


	bool lerpLightUpdate = false;


	// Don't launch another light dispatch if the previous one hasn't finished.
	if (m_rtgiFence->IsFenceComplete(m_rtgiFence->CurrentFenceValue))
	{
		// We only want to dispatch the light if the camera moved or the light changed.
		if (!LightDispatched && (shouldUpdateLight || m_cameraMovedSinceLastUpdate))
		{
			// We wait for the rasterization pass to finish since we use depth maps to compute the lights.
			if (m_rasterFence->IsFenceComplete(m_rasterFenceValue))
			{

				DX12Lib::ComputeContext& context = ComputeContext::Begin();

				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
				if (shouldUpdateLight)
				{
					m_lightVoxel->PerformTechnique(context);

					m_lightTransportTechnique->ResetRadianceBuffers(true);
					RTGIUpdateDelta = 0.0f;

					m_wasRadianceReset = 1;
				}
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 1);

				// Compute visible faces
				m_lightTransportTechnique->PerformTechnique(context);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 2);
				// Compute GI for each face
				m_lightTransportTechnique->LaunchIndirectLightBlock(context, 1);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 3);
				// Gaussian double filter pass for each visible face
				m_gaussianFilterTechnique->PerformTechnique(context);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 4);
				m_gaussianFilterTechnique->PerformTechnique2(context);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 5);
				context.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 6);

				// Copy data to readback buffer if we're sending it to the client.
				if (m_receiveState == ReceiveState::CAMERA_DATA)
				{
					m_lightTransportTechnique->TransferRadianceData(context);
				}
					

				m_rtgiFence->CurrentFenceValue = context.Finish();
				Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);

				LightDispatched = true;
				m_lightDispachCount += 1;
			}
		}
		else if (LightDispatched)
		{
			// Send radiance data to client
			if (m_receiveState == ReceiveState::CAMERA_DATA)
			{
				UINT32 visibleFacesCount = m_lightTransportTechnique->GetVisibleFacesCount();

				if (visibleFacesCount > 0)
				{
					PacketGuard packet = m_networkServer.CreatePacket();
					packet->ClearPacket();
					packet->AppendToBuffer("RDXBUF");
					packet->AppendToBuffer(visibleFacesCount);
					packet->AppendToBuffer(m_wasRadianceReset);
					packet->AppendToBuffer(m_lightTransportTechnique->GetVisibleFacesIndices(visibleFacesCount), visibleFacesCount * sizeof(UINT32));
					packet->AppendToBuffer(m_lightTransportTechnique->GetVisibleFacesRadiance(visibleFacesCount), visibleFacesCount * sizeof(UINT32));
					m_networkServer.SendData(packet);

					m_wasRadianceReset = 0;
				}
			}

			UINT64* timingData = reinterpret_cast<UINT64*>(m_timingReadBackBuffer.ReadBack(sizeof(UINT64) * 6));
			m_litVoxelTime = ((timingData[1] - timingData[0]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
			m_visibleVoxelTime = ((timingData[2] - timingData[1]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
			m_computeRadianceTime = ((timingData[3] - timingData[2]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
			m_firstGaussianFilterTime = ((timingData[4] - timingData[3]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
			m_secondGaussianFilterTime = ((timingData[5] - timingData[4]) * 1000.0f) / Graphics::GetComputeGPUFrequency();

			m_accTotalTime += m_litVoxelTime + m_visibleVoxelTime + m_computeRadianceTime + m_firstGaussianFilterTime + m_secondGaussianFilterTime;

			// Signal lerp shader to update the radiance data.
			m_isRadianceReady = true;
			LightDispatched = false;

			lerpLightUpdate = m_lightChangedSinceLastUpdate;

			m_cameraMovedSinceLastUpdate = false;
			m_lightChangedSinceLastUpdate = false;
		}
	}


	if (m_isRadianceReady)
	{
		Renderer::ResetLerpTime();
		m_lerpDeltaTime = 0.0f;
	}

	// If only the camera moved, we don't want to lerp but change the radiance immediately to avoid radiance popping.
	if (didCameraMove && !didLightChange)
	{
		Renderer::SetLerpMaxTime(1.0f);
		Renderer::SetDeltaLerpTime(1.0f);
	}
	// In all other cases we want to lerp. Radiance popping may still be noticeable but it's less jarring when the light changes.
	else
	{
		Renderer::SetLerpMaxTime(0.2f);
		Renderer::SetDeltaLerpTime(m_lerpDeltaTime);
	}


	// Render Layers
	{
		// Update depth cameras used to compute GI.
		// Ideally we would only update the camera if the light changed or camera moved, but this is a cheap operation so we can get away with doing it every frame.
		if (m_isRadianceReady)
		{
			m_sceneDepthTechnique->UpdateCameraMatrices();
			m_sceneDepthTechnique->PerformTechnique(commandContext);
		}

		Renderer::ShadowPass(commandContext);
		Renderer::MainRenderPass(commandContext);
		
		Renderer::LerpRadiancePass(commandContext);

		Renderer::DeferredPass(commandContext);
		Renderer::PostProcessPass(commandContext);
		Renderer::UIPass(commandContext);
	}


	if (m_isRadianceReady)
	{
		m_rasterFence->CurrentFenceValue = commandContext.Flush();
		Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);
		m_rasterFenceValue = m_rasterFence->CurrentFenceValue;
	}



	// Display voxels
	//m_displayVoxelScene->PerformTechnique(commandContext);

	m_lerpDeltaTime += GameTime::GetDeltaTime();
	RTGIUpdateDelta += GameTime::GetDeltaTime();

	m_isRadianceReady = false;

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::OnClose(DX12Lib::GraphicsContext& commandContext)
{
	m_isRunning = false;
	
	m_networkServer.Disconnect();
	DX12Lib::NetworkHost::DeinitializeEnet();
	
	D3DApp::OnClose(commandContext);
}

bool CVGI::ClusteredVoxelGIApp::ShowIMGUIWindow(DX12Lib::GraphicsContext& context)
{
	static IMGUIWindowStatus voxelInitStatus = IMGUIWindowStatus::INITIALIZING;

	float appX = static_cast<float>(Renderer::s_clientWidth);
	float appY = static_cast<float>(Renderer::s_clientHeight);

	if (voxelInitStatus == IMGUIWindowStatus::VOXEL_SELECTION_SCREEN)
	{
		if (ShowIMGUIVoxelOptionWindow(appX, appY))
		{
			voxelInitStatus = IMGUIWindowStatus::LOADING_SCREEN;
		}

		Renderer::UIPass(context, true);

		return false;
	}
	else if (voxelInitStatus == IMGUIWindowStatus::LOADING_SCREEN)
	{
		ShowIMGUILoadingWindow(appX, appY);

		voxelInitStatus = IMGUIWindowStatus::INITIALIZING;

		return false;
	}
	else if (voxelInitStatus == IMGUIWindowStatus::INITIALIZING)
	{

		InitializeVoxelData(context);
		context.Flush(true);
		voxelInitStatus = IMGUIWindowStatus::VOXEL_DEBUG_SCREEN;

		return false;
	}
	else if (voxelInitStatus == IMGUIWindowStatus::VOXEL_DEBUG_SCREEN)
	{
		ShowIMGUIVoxelDebugWindow(appX, appY);
	}

	return true;
}

bool CVGI::ClusteredVoxelGIApp::ShowIMGUIVoxelOptionWindow(float appX, float appY)
{
	float spacing = 10.0f;
	float buttonWidth = appX * 0.2f;
	float buttonHeight = appY * 0.08f;

	ImVec2 buttonPosition = ImVec2((appX - buttonWidth) * 0.5f, appY * 0.1f);

	ImGui::SetNextWindowSize(ImVec2(appX, appY));
	ImGui::SetNextWindowPos(ImVec2(appX * 0.5f, appY * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f)); // Centered position


	ImGui::Begin("Voxel Options", nullptr, ImGuiWindowFlags_NoTitleBar);

	const char* label = "Choose Voxel Scene Size";
	ImVec2 labelSize = ImGui::CalcTextSize(label);

	ImGui::SetCursorPos(ImVec2((appX - labelSize.x) * 0.5f, appY * 0.1f));
	ImGui::Text("%s", label);
	
	buttonPosition.y += labelSize.y + spacing;

	const uint32_t voxelSizes[4] = { 64, 128, 256, 512 };


	std::string buttonName;

	for (int i = 0; i < 4; i++)
	{
		ImGui::SetCursorPos(buttonPosition);
		
		buttonName = "Voxel Size: " + std::to_string(voxelSizes[i]) + " x " + std::to_string(voxelSizes[i]) + " x " + std::to_string(voxelSizes[i]);
		bool pressed = ImGui::Button(buttonName.c_str(), ImVec2(buttonWidth, buttonHeight));

		if (pressed)
		{
			VoxelTextureDimension = DirectX::XMUINT3(voxelSizes[i], voxelSizes[i], voxelSizes[i]);
			ImGui::End();
			return true;
		}

		buttonPosition.y += buttonHeight + spacing;
	}

	ImGui::End();

	return false;
}

void CVGI::ClusteredVoxelGIApp::ShowIMGUILoadingWindow(float appX, float appY)
{
	ImGui::SetNextWindowSize(ImVec2(appX, appY));
	ImGui::SetNextWindowPos(ImVec2(appX * 0.5f, appY * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f)); // Centered position

	ImGui::Begin("Voxel Options", nullptr, ImGuiWindowFlags_NoTitleBar);
	const char* label = "Loading...";
	ImVec2 labelSize = ImGui::CalcTextSize(label);

	ImGui::SetCursorPos(ImVec2((appX - labelSize.x) * 0.5f, appY * 0.1f));
	ImGui::Text("%s", label);
	ImGui::End();
}

void CVGI::ClusteredVoxelGIApp::ShowIMGUIVoxelDebugWindow(float appX, float appY)
{
	ImVec2 windowSize = ImVec2(appX * 0.25f, appY);
	ImVec2 windowPos = ImVec2(appX * 0.875f, appY * 0.5f);

	float spacing = 10.0f;
	float halfSpacing = spacing / 2.0f;



	ImGui::SetNextWindowSize(windowSize);
	ImGui::SetNextWindowPos(windowPos, ImGuiCond_Once, ImVec2(0.5f, 0.5f)); // Centered position

	ImGui::Begin("CVGI Scene Options");
	
	ImGui::SeparatorText("Clustered Voxel RTXGI");

	int fps = 0;
	float mspf = 0.0f;
	GetFrameStats(fps, mspf);

	ImGui::Text("FPS: %d\tMSPF: %.2f", fps, mspf);

	ImGui::SeparatorText("Controls");

	float maxX = ImGui::CalcTextSize("- Hold Right Mouse Button:\t").x;

	ImGui::Text("- W, A, S, D:");
	ImGui::SameLine(maxX);
	ImGui::Text("Move Camera");

	ImGui::Text("- E, Q:");
	ImGui::SameLine(maxX);
	ImGui::Text("Move Camera Up/Down");

	ImGui::Text("- Hold Right Mouse Button:");
	ImGui::SameLine(maxX);
	ImGui::Text("Rotate Camera");

	ImGui::Text("- Arrow Keys:");
	ImGui::SameLine(maxX);
	ImGui::Text("Move Light");

	ImGui::Text("- ESC:");
	ImGui::SameLine(maxX);
	ImGui::Text("Quit");

	if (ImGui::CollapsingHeader("Voxelization Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		maxX = ImGui::CalcTextSize("Compute Neighbours Time:\t").x;

		ImGui::Text("Voxel Grid Size: (%d x %d x %d)", VoxelTextureDimension.x, VoxelTextureDimension.y, VoxelTextureDimension.z);
		ImGui::Text("Voxel Count: %d\tCluster Count: %d", m_data->GetVoxelCount(), m_data->GetClusterCount());

		ImGui::SeparatorText("Real-time GI timings");
		ImGui::Text("Lit voxels:");
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_litVoxelTime);

		ImGui::Text("Visible voxels:");
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_visibleVoxelTime);

		ImGui::Text("Compute radiance:");
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_computeRadianceTime);

		ImGui::Text("First gaussian filter:");
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_firstGaussianFilterTime);

		ImGui::Text("Second gaussian filter:");
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_secondGaussianFilterTime);

		float totalTime = m_litVoxelTime + m_visibleVoxelTime + m_computeRadianceTime + m_firstGaussianFilterTime + m_secondGaussianFilterTime;
		ImGui::Text("Total time (last frame):"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", totalTime);
		
		ImGui::Text("Average total time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_accTotalTime / (m_lightDispachCount));

		ImGui::SeparatorText("Initialization timings");

		ImGui::Text("Voxelization Time:"); 
		ImGui::SameLine(maxX); 
		ImGui::Text("%.2f ms", m_voxelBuildTime);
		
		ImGui::Text("Prefix Sum Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_prefixSumTime);
		
		ImGui::Text("Clusterize Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_clusterizeTime);

		ImGui::Text("Compute Neighbours Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_computeNeighboursTime);
		
		ImGui::Text("Build AABBs Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_buildAABBsTime);

		ImGui::Text("Building TLAS time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_buildingAccelerationStructuresTime);

		ImGui::Text("Cluster Visibility Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_clusterVisibilityTime);

		ImGui::Text("Initial Radiance Time:"); 
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_initialRadianceTime);
	}

	if (ImGui::CollapsingHeader("Light options"))
	{
		auto* light = m_voxelScene->GetMainLight();

		DirectX::XMFLOAT3 lightColor = light->GetLightColor();

		float color[3] = { lightColor.x, lightColor.y, lightColor.z };

		ImGui::CalcItemWidth();

		if (ImGui::ColorPicker3("Light Color", color, 
			 ImGuiColorEditFlags_Float | ImGuiColorEditFlags_DisplayRGB))
		{
			lightColor = DirectX::XMFLOAT3(color[0], color[1], color[2]);
			light->SetLightColor(lightColor);
		}

		maxX = ImGui::CalcTextSize("Close voxels bounce strength:\t").x;

		float intensity = light->GetLightIntensity();

		ImGui::SetCursorPosY(ImGui::GetCursorPosY() + spacing);
		ImGui::Text("Light Intensity:\t", intensity);
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##LightIntensity", &intensity, 0.0f, 10.0f))
		{
			light->SetLightIntensity(intensity);
		}
		
		float farStrength = m_lightTransportTechnique->GetFarVoxelRadianceStrength();
		float closeStrength = m_lightTransportTechnique->GetCloseVoxelRadianceStrength();

		ImGui::Text("Far voxels bounce strength:", farStrength);
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##FarStrength", &farStrength, 0.0f, 7.0f))
		{
			m_lightTransportTechnique->SetFarVoxelRadianceStrength(farStrength);
		}

		ImGui::Text("Close voxels bounce strength:", closeStrength);
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##CloseStrength", &closeStrength, 0.0f, 7.0f))
		{
			m_lightTransportTechnique->SetCloseVoxelRadianceStrength(closeStrength);
		}
	}


	ImGui::End();
}

void CVGI::ClusteredVoxelGIApp::InitializeVoxelData(DX12Lib::GraphicsContext& commandContext)
{
	m_data->SetVoxelGridSize(VoxelTextureDimension);

	m_voxelizeScene->InitializeBuffers();

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

	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);


	voxelCellSize.x = (sceneBounds.Max.x - sceneBounds.Min.x) / ((float)VoxelTextureDimension.x);
	voxelCellSize.y = (sceneBounds.Max.y - sceneBounds.Min.y) / ((float)VoxelTextureDimension.y);
	voxelCellSize.z = (sceneBounds.Max.z - sceneBounds.Min.z) / ((float)VoxelTextureDimension.z);

	m_data->SetVoxelCellSize(voxelCellSize);
	m_data->SetSceneAABB(sceneBounds);
	m_data->SetCamera(m_Scene->GetMainCamera());

	m_data->BuildMatrices();

	DXLIB_CORE_INFO("Scene bounds found at: Min: {0} {1} {2}; Max: {3} {4} {5}",
		sceneBounds.Min.x, sceneBounds.Min.y, sceneBounds.Min.z,
		sceneBounds.Max.x, sceneBounds.Max.y, sceneBounds.Max.z);

	VoxelScene* voxelScene = static_cast<VoxelScene*>(this->m_Scene.get());
	voxelScene->Render(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();
	voxelCamera->SetOrthogonalHalfExtents(DirectX::XMFLOAT3(extent / 2.0f, extent / 2.0f, extent / 2.0f));

	m_voxelizeScene->SetVoxelCamera(voxelCamera);

	commandContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
	m_voxelizeScene->PerformTechnique(commandContext);
	commandContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 1);
	commandContext.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 2);

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



	UINT64* timingData = reinterpret_cast<UINT64*>(m_timingReadBackBuffer.ReadBack(sizeof(UINT64) * 2));

	m_voxelBuildTime = ((timingData[1] - timingData[0]) * 1000.0f) / Graphics::GetGraphicsGPUFrequency();

	DXLIB_CORE_INFO("Voxelization time: {0} ms", m_voxelBuildTime);
	



	ComputeContext& computeContext = ComputeContext::Begin();

	m_prefixSumVoxels->InitializeBuffers(computeContext);

	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
	m_prefixSumVoxels->PerformTechnique(computeContext);
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 1);

	// Voxelize scene temporary buffers can be deleted after the voxelization and prefix sum passes.
	m_voxelizeScene->DeleteTemporaryBuffers();
	// Prefix sum temporary buffers are only needed for the prefix sum pass.
	m_prefixSumVoxels->DeleteTemporaryBuffers();



	m_clusterVoxels->InitializeBuffers();

	m_clusterVoxels->PerformTechnique(computeContext);
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 2);

	m_data->SetClusterCount(m_clusterVoxels->GetNumberOfClusters());

	m_computeNeighboursTechnique->InitializeBuffers();
	m_computeNeighboursTechnique->PerformTechnique(computeContext);
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 3);


	m_buildAABBsTechnique->InitializeBuffers();
	m_buildAABBsTechnique->PerformTechnique(computeContext);
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 4);

	m_clusterVisibility->InitializeBuffers();
	m_data->SetTlas(m_clusterVisibility->BuildAccelerationStructures(computeContext));
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 5);

	m_clusterVisibility->PerformTechnique(computeContext);
	computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 6);
	computeContext.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 7);


	m_sceneDepthTechnique->InitializeBuffers();

	m_lightVoxel->InitializeBuffers();

	m_lightTransportTechnique->InitializeBuffers();


	m_gaussianFilterTechnique->InitializeBuffers();
	m_gaussianFilterTechnique->SetIndirectCommandSignature(m_lightTransportTechnique->GetIndirectCommandSignature());
	m_gaussianFilterTechnique->InitializeGaussianConstants(computeContext);

	assert(foundLightComponent && "Failed to find light component with shadows enabled.");

	computeContext.Flush(true);

	timingData = reinterpret_cast<UINT64*>(m_timingReadBackBuffer.ReadBack(sizeof(UINT64) * 7));

	m_prefixSumTime = ((timingData[1] - timingData[0]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
	m_clusterizeTime = ((timingData[2] - timingData[1]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
	m_computeNeighboursTime = ((timingData[3] - timingData[2]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
	m_buildAABBsTime = ((timingData[4] - timingData[3]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
	m_buildingAccelerationStructuresTime = ((timingData[5] - timingData[4]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
	m_clusterVisibilityTime = ((timingData[6] - timingData[5]) * 1000.0f) / Graphics::GetComputeGPUFrequency();

	m_sceneDepthTechnique->UpdateCameraMatrices();
	m_sceneDepthTechnique->PerformTechnique(commandContext);

	commandContext.Flush(true);

	commandContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
	m_lightVoxel->PerformTechnique(computeContext);
	m_lightTransportTechnique->ComputeStartingRadiance(computeContext);
	m_lightTransportTechnique->LaunchIndirectLightBlock(computeContext, 1);
	m_gaussianFilterTechnique->PerformTechnique(computeContext);
	m_gaussianFilterTechnique->PerformTechnique2(computeContext);
	commandContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 1);

	commandContext.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 2);

	computeContext.Finish(true);

	timingData = reinterpret_cast<UINT64*>(m_timingReadBackBuffer.ReadBack(sizeof(UINT64) * 2));
	m_initialRadianceTime = ((timingData[1] - timingData[0]) * 1000.0f) / Graphics::GetComputeGPUFrequency();

	DXLIB_CORE_INFO("Prefix sum time: {0} ms", m_prefixSumTime);
	DXLIB_CORE_INFO("Clusterize time: {0} ms", m_clusterizeTime);
	DXLIB_CORE_INFO("Compute neighbours time: {0} ms", m_computeNeighboursTime);
	DXLIB_CORE_INFO("Build AABBs time: {0} ms", m_buildAABBsTime);
	DXLIB_CORE_INFO("Build Acceleration structure time: {0} ms", m_buildingAccelerationStructuresTime);
	DXLIB_CORE_INFO("Cluster visibility time: {0} ms", m_clusterVisibilityTime);
	DXLIB_CORE_INFO("Initial radiance time: {0} ms", m_initialRadianceTime);

	m_lightTransportTechnique->ResizeIndexBuffers();

	m_displayVoxelScene->SetCamera(m_Scene->GetMainCamera());
	m_displayVoxelScene->SetVertexData(commandContext);

	auto& rendererRTGIHandle = Renderer::GetRTGIHandleSRV();


	D3D12_CPU_DESCRIPTOR_HANDLE srvHandles[6];

	srvHandles[0] = m_voxelizeScene->GetBufferManager()->GetBuffer(0).GetSRV();

	for (UINT i = 1; i < 5; i++)
	{
		srvHandles[i] = m_prefixSumVoxels->GetBufferManager()->GetBuffer(i - 1).GetSRV();
	}

	srvHandles[5] = m_data->GetBufferManager(GaussianFilterTechnique::ReadName).GetBuffer(0).GetSRV();

	auto descriptorSize = Renderer::s_textureHeap->GetDescriptorSize();
	for (UINT i = 0; i < 6; i++)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(1, rendererRTGIHandle + descriptorSize * i, srvHandles[i], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Renderer::SetRTGIData(m_data->GetVoxelCommons());
	Renderer::UseRTGI(true);

	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);


	m_rtgiFence->Get()->SetName(L"RTGI Fence");
	m_rasterFence->Get()->SetName(L"Acc Fence");

	m_voxelScene = voxelScene;

	DX12Lib::NetworkHost::InitializeEnet();
	m_networkServer.OnPeerConnected = std::bind(&ClusteredVoxelGIApp::OnClientConnected, this, std::placeholders::_1);
	m_networkServer.OnPacketReceived = std::bind(&ClusteredVoxelGIApp::OnPacketReceived, this, std::placeholders::_1);
	m_networkServer.StartServer(1234);

}

void CVGI::ClusteredVoxelGIApp::OnPacketReceived(const DX12Lib::NetworkPacket* packet)
{
	if (m_receiveState == ReceiveState::CAMERA_DATA)
	{
		if (NetworkHost::CheckPacketHeader(packet, "CAMINP"))
		{
			ConsumeNodeInput(packet, true);
		}
		else if (NetworkHost::CheckPacketHeader(packet, "LGTINP"))
		{
			ConsumeNodeInput(packet, false);
		}
	}
	// Awaiting ACK for buffer data. If received we change state to listen for camera data.
	else if (m_receiveState == ReceiveState::NECESSARY_BUFFERS)
	{
		if (NetworkHost::CheckPacketHeader(packet, "BUFFER"))
		{
			DXLIB_CORE_INFO("Client received buffers");
			m_receiveState = ReceiveState::CAMERA_DATA;
		}
	}
	// Voxel size data was sent on connection. Awaiting for an ACK from client to send buffer data.
	else if (m_receiveState == ReceiveState::INITIALIZATION)
	{
		if (NetworkHost::CheckPacketHeader(packet, "INIT"))
		{
			DXLIB_CORE_INFO("Received INIT packet");
			m_receiveState = ReceiveState::NECESSARY_BUFFERS;

			auto& voxelOccupiedBuffer = m_voxelizeScene->GetOccupiedVoxelBuffer();

			PacketGuard occupiedBufferPkt = m_networkServer.CreatePacket();
			occupiedBufferPkt->ClearPacket();
			occupiedBufferPkt->AppendToBuffer("OCCVOX");
			occupiedBufferPkt->AppendToBuffer(voxelOccupiedBuffer);
			m_networkServer.SendData(occupiedBufferPkt);

			DXLIB_CORE_INFO("Send VOXOCC buffer with a size of: {0}", voxelOccupiedBuffer.size());

			auto& indRnkBuffer = m_prefixSumVoxels->GetIndirectionRankBuffer();

			PacketGuard indirectRankBufferPkt = m_networkServer.CreatePacket();
			indirectRankBufferPkt->ClearPacket();
			indirectRankBufferPkt->AppendToBuffer("INDRNK");
			indirectRankBufferPkt->AppendToBuffer(indRnkBuffer);
			m_networkServer.SendData(indirectRankBufferPkt);

			DXLIB_CORE_INFO("Send INDRNK buffer with a size of: {0}", indRnkBuffer.size());


			auto& indIdxBUffer = m_prefixSumVoxels->GetIndirectionIndexBuffer();


			PacketGuard indirectIndexBufferPkt = m_networkServer.CreatePacket();
			indirectIndexBufferPkt->ClearPacket();
			indirectIndexBufferPkt->AppendToBuffer("INDIDX");
			indirectIndexBufferPkt->AppendToBuffer(indIdxBUffer);
			m_networkServer.SendData(indirectIndexBufferPkt);

			DXLIB_CORE_INFO("Send INDIDX buffer with a size of: {0}", indIdxBUffer.size());


			auto& cmpIdxBuffer = m_prefixSumVoxels->GetCompactedVoxelIndexBuffer();


			PacketGuard compactedIndicesPkt = m_networkServer.CreatePacket();
			compactedIndicesPkt->ClearPacket();
			compactedIndicesPkt->AppendToBuffer("CMPIDX");
			compactedIndicesPkt->AppendToBuffer(cmpIdxBuffer);
			m_networkServer.SendData(compactedIndicesPkt);

			DXLIB_CORE_INFO("Send CMPIDX buffer with a size of: {0}", cmpIdxBuffer.size());

			auto& cmpHshBuffer = m_prefixSumVoxels->GetCompactedHashedBuffer();

			PacketGuard compactedHashesPkt = m_networkServer.CreatePacket();
			compactedHashesPkt->ClearPacket();
			compactedHashesPkt->AppendToBuffer("CMPHSH");
			compactedHashesPkt->AppendToBuffer(cmpHshBuffer);
			m_networkServer.SendData(compactedHashesPkt);

			DXLIB_CORE_INFO("Send CMPHSH buffer with a size of: {0}", cmpHshBuffer.size());

		}
	}
}

void CVGI::ClusteredVoxelGIApp::OnClientConnected(const ENetPeer* peer)
{
	PacketGuard packet = m_networkServer.CreatePacket();
	packet->ClearPacket();
	packet->AppendToBuffer("VOX");
	packet->AppendToBuffer(VoxelTextureDimension.x);
	packet->AppendToBuffer(VoxelTextureDimension.y);
	packet->AppendToBuffer(VoxelTextureDimension.z);
	packet->AppendToBuffer(m_data->GetVoxelCount());
	packet->AppendToBuffer(m_data->GetClusterCount());

	m_networkServer.SendData(packet);

	m_receiveState = ReceiveState::INITIALIZATION;
}

void CVGI::ClusteredVoxelGIApp::ConsumeNodeInput(const DX12Lib::NetworkPacket* packet, bool isCamera)
{
	auto& dataVector = packet->GetDataVector();

	UINT64 timeStamp = 0;
	DirectX::XMFLOAT3 clientAbsPos;
	DirectX::XMFLOAT4 clientAbsRot;
	std::uint8_t clientInputBitmask = 0;

	// HEADER + NULL character
	size_t previousSize = 7;


	memcpy(&timeStamp, dataVector.data() + previousSize, sizeof(UINT64));
	previousSize += sizeof(UINT64);

	memcpy(&clientAbsPos, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT3));
	previousSize += sizeof(DirectX::XMFLOAT3);

	memcpy(&clientAbsRot, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT4));
	previousSize += sizeof(DirectX::XMFLOAT4);

	memcpy(&clientInputBitmask, dataVector.data() + previousSize, sizeof(std::uint8_t));
	previousSize += sizeof(std::uint8_t);

	if (isCamera)
	{
		auto* camera = m_data->GetCamera();

		if (camera != nullptr)
		{
			CameraController* controller = camera->Node->GetComponent<CameraController>();

			if (controller != nullptr)
			{
				controller->IsRemote = true;
				controller->SetRemoteInput(clientInputBitmask, clientAbsPos, clientAbsRot, timeStamp);
			}
		}
	}

	else
	{
		auto* light = m_data->GetLightComponent();

		if (light != nullptr)
		{
			LightController* controller = light->Node->GetComponent<LightController>();

			if (controller != nullptr)
			{
				controller->ControlOverNetwork(true);
				controller->SetRemoteInput(clientInputBitmask, clientAbsPos, clientAbsRot, timeStamp);
			}
		}
	}
}

bool CVGI::ClusteredVoxelGIApp::IsDirectXRaytracingSupported() const
{
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 featureSupport = {};

	if (FAILED(Graphics::s_device->Get()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &featureSupport, sizeof(featureSupport))))
		return false;

	return featureSupport.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}

void CVGI::ClusteredVoxelGIApp::GetFrameStats(int& fps, float& mspf)
{
	static int lastFps = 0.0f;
	static float lastMSPF = 0.0f;

	static int frameCount = 0;
	static float lastTime = 0.0f;

	frameCount++;

	float frameTime = GameTime::GetTotalTime();

	if (frameTime - lastTime >= 1.0)
	{
		lastFps = frameCount;
		lastMSPF = 1000.0f / (float)lastFps;

		frameCount = 0;

		lastTime = frameTime;
	}

	fps = lastFps;
	mspf = lastMSPF;
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