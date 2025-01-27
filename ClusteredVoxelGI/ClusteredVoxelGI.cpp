#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/RemoteNodeController.h"
#include "DX12Lib/Scene/LightController.h"
#include "DX12Lib/Scene/CameraController.h"
#include "DX12Lib/Commons/DX12Window.h"
#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"
#include "UIHelpers.h"



using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

using namespace CVGI;
using namespace VOX;
using namespace Commons;


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
		m_lightTransportTechnique->DidRadianceStrengthChange() ||
		m_gaussianFilterTechnique->GaussianOptionModified();

	bool didCameraMove = m_data->GetCamera()->IsDirty();

	m_cameraMovedSinceLastUpdate |= didCameraMove;
	m_lightChangedSinceLastUpdate |= didLightChange;

	bool shouldUpdateLight = m_lightChangedSinceLastUpdate && (m_RTGIUpdateDelta > m_RTGIMaxTime);

	m_Scene->Render(commandContext);


	bool lerpLightUpdate = false;


	// Don't launch another light dispatch if the previous one hasn't finished.
	if (m_rtgiFence->IsFenceComplete(m_rtgiFence->CurrentFenceValue))
	{
		// We only want to dispatch the light if the camera moved or the light changed.
		if (!LightDispatched && (shouldUpdateLight || m_cameraMovedSinceLastUpdate))
		{

			// We wait for the rasterization pass to finish since we use depth maps to compute the lights.
			if (m_rasterFence->IsFenceComplete(m_rasterFence->CurrentFenceValue))
			{
				DX12Lib::ComputeContext& context = ComputeContext::Begin();

				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
				if (shouldUpdateLight)
				{
					m_lightVoxel->PerformTechnique(context);

					m_lightTransportTechnique->ResetRadianceBuffers(true);
					m_RTGIUpdateDelta = 0.0f;

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
				
				if (m_renderRasterScene)
					m_gaussianFilterTechnique->PerformTechnique(context);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 4);
				
				if (m_renderRasterScene)
					m_gaussianFilterTechnique->PerformTechnique2(context);
				context.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 5);
				context.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 6);

				// Copy data to readback buffer if we're sending it to the client.
				if (m_isClientReadyForRadiance)
				{
					m_lightTransportTechnique->TransferRadianceData(context);
				}


				m_rtgiFence->CurrentFenceValue = context.Finish();
				Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);

				LightDispatched = true;
			}
		}
		else if (LightDispatched)
		{
			// Send radiance data to client
			if (m_isClientReadyForRadiance)
			{
				UINT32 visibleFacesCount = m_lightTransportTechnique->GetVisibleFacesCount();

				if (!m_firstRadianceSent)
				{
					visibleFacesCount = m_data->FaceCount;
					m_firstRadianceSent = true;
				}

				if (visibleFacesCount > 0)
				{
					PacketGuard packet = m_networkServer.CreatePacket();
					packet->SetPacketType(NetworkPacket::PacketType::PACKET_UNRELIABLE);
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

			m_lightDispatchCount += 1;

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
		Renderer::SetLerpMaxTime(m_lerpMaxTime);
		Renderer::SetDeltaLerpTime(m_lerpDeltaTime);
	}

	

	// Update depth cameras used to compute GI.
	// Ideally we would only update the camera if the light changed or camera moved, but this is a cheap operation so we can get away with doing it every frame.
	if (m_isRadianceReady)
	{
		m_sceneDepthTechnique->UpdateCameraMatrices();
		m_sceneDepthTechnique->PerformTechnique(commandContext);
		m_rasterFence->CurrentFenceValue = commandContext.Flush();
		Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);
	}

	// Display voxels

	if (m_renderRasterScene)
	{
		if (m_displayVoxelScene->GetDisplayMode() == 0)
		{
			Renderer::ShadowPass(commandContext);
			Renderer::MainRenderPass(commandContext);

			Renderer::LerpRadiancePass(commandContext);

			Renderer::DeferredPass(commandContext);
			Renderer::PostProcessPass(commandContext);
		}
		else
		{
			m_displayVoxelScene->PerformTechnique(commandContext);
		}
	}

	Renderer::UIPass(commandContext, !m_renderRasterScene);
	





	m_lerpDeltaTime += GameTime::GetDeltaTime();
	m_RTGIUpdateDelta += GameTime::GetDeltaTime();

	m_isRadianceReady = false;

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::OnClose(DX12Lib::GraphicsContext& commandContext)
{
	m_isRunning = false;
	
	if (m_networkServer.IsConnected())
	{
		m_networkServer.Disconnect();
	}

	Commons::NetworkHost::DeinitializeEnet();
	
	D3DApp::OnClose(commandContext);
}

bool CVGI::ClusteredVoxelGIApp::ShowIMGUIWindow(DX12Lib::GraphicsContext& context)
{
	static IMGUIWindowStatus voxelInitStatus = IMGUIWindowStatus::VOXEL_SELECTION_SCREEN;

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

	const uint32_t voxelSizes[3] = { 64, 128, 256 };


	std::string buttonName;

	for (int i = 0; i < 3; i++)
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

	int clusterLevel = m_clusterVoxels->GetClusterizationLevl();



	buttonPosition.y += spacing;

	const char* nextLabel = "Clusterization Level";
	labelSize = ImGui::CalcTextSize(nextLabel);

	ImGui::SetCursorPos(ImVec2((appX - labelSize.x) * 0.5f, buttonPosition.y));
	ImGui::Text("%s", nextLabel);

	buttonPosition.y += labelSize.y + spacing;
	ImGui::SetCursorPos(buttonPosition);
	ImGui::SetNextItemWidth(buttonWidth);
	if (ImGui::SliderInt("##Clusterization Level", &clusterLevel, 1, 6))
	{
		m_clusterVoxels->SetClusterizationLevel(clusterLevel);
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

	float memoryUsageMiB = (float)m_rtgiMemoryUsage;
	memoryUsageMiB /= (1024.0f * 1024.0f);

	ImGui::Text("FPS: %d\tMSPF: %.2f", fps, mspf);
	ImGui::Text("RTGI Memory usage: %.2f MiB", memoryUsageMiB);

	UIHelpers::ControlInfoBlock(m_isClientReadyForRadiance);

	ImGui::Separator();

	if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
		ImGui::SetTooltip("Enable/Disable server scene rendering. Radiance computation and networking will still work as expected.\n");
	}
	ImGui::Checkbox("Render scene on server", &m_renderRasterScene);

	float maxX = ImGui::CalcTextSize("\t").x;

	if (ImGui::CollapsingHeader("Voxelization Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		maxX = ImGui::CalcTextSize("Compute Neighbours Time:\t").x;

		ImGui::Text("Voxel Grid Size: (%d x %d x %d)", VoxelTextureDimension.x, VoxelTextureDimension.y, VoxelTextureDimension.z);
		ImGui::Text("Voxel Count: %d\tCluster Count: %d", m_data->GetVoxelCount(), m_data->GetClusterCount());

		ImGui::SeparatorText("Real-time GI timings (latest dispatch)");

		ImGui::Text("Lit voxels:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute voxels visible from the light PoV.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_litVoxelTime);

		ImGui::Text("Visible voxels:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute voxels visible from the main camera.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_visibleVoxelTime);

		ImGui::Text("Compute radiance:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute the radiance for each voxel visible from the main camera.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_computeRadianceTime);

		ImGui::Text("First gaussian filter:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute the first gaussian filter pass.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_firstGaussianFilterTime);

		ImGui::Text("Second gaussian filter:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute the second gaussian filter pass.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_secondGaussianFilterTime);

		float totalTime = m_litVoxelTime + m_visibleVoxelTime + m_computeRadianceTime + m_firstGaussianFilterTime + m_secondGaussianFilterTime;
		ImGui::Text("Total time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Total time needed to complete all the previous steps for the latest dispatch.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", totalTime);
		
		ImGui::Text("Average dispatch time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Average time needed to complete all the previous steps for each dispatch.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_accTotalTime / (m_lightDispatchCount));

		ImGui::SeparatorText("Initialization timings");

		ImGui::Text("Voxelization Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to voxelize the scene for the given voxel resolution.");
		}
		ImGui::SameLine(maxX); 
		ImGui::Text("%.2f ms", m_voxelBuildTime);
		
		ImGui::Text("Prefix Sum Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compact voxel buffers using prefix sum algorithm.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_prefixSumTime);
		
		ImGui::Text("Clusterize Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to group the voxels into clusters.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_clusterizeTime);

		ImGui::Text("Compute Neighbours Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute neighbours of each cluster.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_computeNeighboursTime);
		
		ImGui::Text("Build AABBs Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to build AABBs for each voxel.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_buildAABBsTime);

		ImGui::Text("Building TLAS time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to create the Ray-Tracing acceleration structures.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_buildingAccelerationStructuresTime);

		ImGui::Text("Cluster Visibility Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to find, for each cluster, all other visible clusters using RT.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_clusterVisibilityTime);

		ImGui::Text("Initial Radiance Time:"); 
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute the initial radiance for the whole scene.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_initialRadianceTime);
	}

	ImGui::BeginDisabled(m_isClientReadyForRadiance);
	if (ImGui::CollapsingHeader("Light"))
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
		ImGui::Text("Light Intensity:\t");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Light color multiplier to increase/decrease brightness.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##LightIntensity", &intensity, 0.0f, 10.0f))
		{
			light->SetLightIntensity(intensity);
		}
		
		ImGui::SeparatorText("Indirect Light");

		float farStrength = m_lightTransportTechnique->GetFarVoxelRadianceStrength();
		float closeStrength = m_lightTransportTechnique->GetCloseVoxelRadianceStrength();

		ImGui::Text("Far voxels bounce strength:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Sets the intensity of indirect light gathered from far voxels.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##FarStrength", &farStrength, 0.0f, 8.0f))
		{
			m_lightTransportTechnique->SetFarVoxelRadianceStrength(farStrength);
		}

		ImGui::Text("Close voxels bounce strength:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Sets the intensity of indirect light gathered from nearby voxels.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##CloseStrength", &closeStrength, 0.0f, 8.0f))
		{
			m_lightTransportTechnique->SetCloseVoxelRadianceStrength(closeStrength);
		}

		ImGui::SeparatorText("Update frequency");

		ImGui::Text("Light update frequency:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("How often should radiance be recomputed when lightning changes.\n(Smaller is better but more expensive).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		ImGui::SliderFloat("##UpdateFrequency", &m_RTGIMaxTime, 0.0f, 0.5f);

		ImGui::Text("Lerp update frequency:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time to lerp between previous radiance values and new radiance values.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		ImGui::SliderFloat("##LerpFrequency", &m_lerpMaxTime, 0.0f, 1.0f);

		ImGui::Separator();

		if (ImGui::Button("Reset settings"))
		{
			light->SetLightColor(DirectX::XMFLOAT3(0.45f, 0.45f, 0.45f));
			light->SetLightIntensity(1.0f);

			m_lightTransportTechnique->SetFarVoxelRadianceStrength(1.0f);
			m_lightTransportTechnique->SetCloseVoxelRadianceStrength(1.0f);

			m_RTGIMaxTime = 0.15f;
			m_lerpMaxTime = 0.2f;
		}
	}
	ImGui::EndDisabled();
	if (ImGui::CollapsingHeader("Gaussian filter"))
	{
		maxX = ImGui::CalcTextSize("Use precomputed gaussian values:\t").x;

		int kernelSize = (int)m_gaussianFilterTechnique->GetGaussianKernelSize();

		bool usePrecomputed = m_gaussianFilterTechnique->GetUsePrecomputedGaussian();

		ImGui::Text("Use precomputed gaussian values:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Use precomputed gaussian values instead of dynamically computing them at run-time. \
			\nThis is slightly faster but uses more memory.\n(Disable this to modify other gaussian options.");
		}
		ImGui::SameLine(maxX);
		if (ImGui::Checkbox("##UsePrecomputedGaussianValues", &usePrecomputed))
		{
			m_gaussianFilterTechnique->SetUsePrecomputedGaussian(usePrecomputed);

			if (usePrecomputed)
			{
				m_gaussianFilterTechnique->SetGaussianKernelSize(5);
				m_gaussianFilterTechnique->SetGaussianSigma(25.0f);
				m_gaussianFilterTechnique->SetGaussianPassCount(2);
			}
		}

		maxX = ImGui::CalcTextSize("Gaussian Kernel size : \t").x;

		ImGui::BeginDisabled(usePrecomputed);

		ImGui::Text("Gaussian Kernel size:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Size of the kernel used to smooth radiance values.\n(Higher is better but more expensive).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (UIHelpers::OddIntegerSlider("##GaussianKernelSize", &kernelSize, 3, 7))
		{
			m_gaussianFilterTechnique->SetGaussianKernelSize((UINT)kernelSize);
		}

		ImGui::Text("Sigma value:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Constant used to determine the smoothness of the gaussian filters.\n(Smaller is less smooth).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		float sigma = m_gaussianFilterTechnique->GetGaussianSigma();
		if (ImGui::SliderFloat("##SigmaValue", &sigma, 0.1f, 3.0f))
		{
			m_gaussianFilterTechnique->SetGaussianSigma(sigma);
		}

		const char* passCountItems[] = { "No passes", "One pass", "Two passes"  };
		int passCount = (int)m_gaussianFilterTechnique->GetGaussianPassCount();

		ImGui::Text("Gaussian pass count:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("How many should the gaussian filter pass be executed, taking previous pass result as input for the next pass.\n(More is better but more expensive).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::ListBox("##GaussianPassCount:", &passCount, passCountItems, 3))
		{
			m_gaussianFilterTechnique->SetGaussianPassCount((UINT)passCount);
		}


		ImGui::EndDisabled();
	}
	if (ImGui::CollapsingHeader("Post-processing"))
	{
		maxX = ImGui::CalcTextSize("Max world position threshold:\t").x;

		float spatialSigma = Renderer::GetPostProcessSpatialSigma();
		ImGui::Text("Spatial sigma:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Smooth nearby fragments radiance based on their screen space position.\n(Higher is smoother).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##SpatialSigma", &spatialSigma, 0.5f, 50.0f))
		{
			Renderer::SetPostProcessSpatialSigma(spatialSigma);
		}

		float intensitySigma = Renderer::GetPostProcessIntensitySigma();
		ImGui::Text("Intensity sigma:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Smooth nearby fragments radiance based on how similar their radiance is.\n(Higher is smoother).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##IntensitySigma", &intensitySigma, 0.5f, 50.0f))
		{
			Renderer::SetPostProcessIntensitySigma(intensitySigma);
		}

		float worldThreshold = Renderer::GetPostProcessWorldThreshold();
		ImGui::Text("Max world position threshold:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Smoothing is not performed for fragments whose world position is higher than this threshold.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##WorldThreshold", &worldThreshold, 0.1f, 10.0f))
		{
			Renderer::SetPostProcessWorldThreshold(worldThreshold);
		}

		int kernelSize = Renderer::GetPostProcessKernelSize();
		ImGui::Text("Kernel size:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Size of the gaussian kernel used to perform the smoothing.\n(Higher is better but more expensive).");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (UIHelpers::OddIntegerSlider("##KernelSize", &kernelSize, 1, 7))
		{
			Renderer::SetPostProcessKernelSize(kernelSize);
		}

	}
	if (ImGui::CollapsingHeader("Networking"))
	{
		bool isConnected = m_networkServer.IsConnected();
		if (!isConnected)
		{
			ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Server is not running");
			if (ImGui::Button("Start Server"))
			{
				m_networkServer.StartServer(1234);
			}
		}
		else
		{
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Server is running");
			if (ImGui::Button("Stop Server"))
			{
				m_networkServer.Disconnect();
				m_isClientReadyForRadiance = false;
				m_firstRadianceSent = false;
			}

			if (m_networkServer.HasPeers())
			{
				UIHelpers::ConnectedClient(m_networkServer.GetPeerAddress().c_str(), m_networkServer.GetPing());
			}
		}

		int compressionLevel = m_networkServer.GetDefaultCompressionLevel();
		ImGui::SeparatorText("Compression");
		ImGui::Text("Compression level:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Compression level used to send radiance.\n(Higher is better but more expensive).\n0 means no compression is performed.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderInt("##CompressionLevel", &compressionLevel, 0, 22))
		{
			m_networkServer.SetDefaultCompressionLevel(compressionLevel);
			m_networkServer.ResetCompressionStats();
		}

		ImGui::SeparatorText("Info");

		maxX = ImGui::CalcTextSize("Average compression ratio:\t").x;

		ImGui::Text("Average compression ratio:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Average ratio between the size of the packet before and after compression.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("x%.2f", m_networkServer.GetAverageCompressionRatio());

		ImGui::Text("Average compression time:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Average time that it took for each packet to be compressed.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_networkServer.GetAverageCompressionTime());

	}
	if (ImGui::CollapsingHeader("Debugging"))
	{
		const char* displayModeItems[] = { "Final Scene", "Voxel Color", "Voxel Normals", "Clusters", "Lit Voxels", "Raw Radiance", "Filtered Radiance"};
		int displayMode = (int)m_displayVoxelScene->GetDisplayMode();

		ImGui::Text("Display Mode:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Select debug display mode");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::ListBox("##DisplayMode:", &displayMode, displayModeItems, ARRAYSIZE(displayModeItems)))
		{
			m_displayVoxelScene->SetDisplayMode(displayMode);
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

	auto& bufferManagers = m_data->GetBufferManagers();

	for (auto& bufferManager : bufferManagers)
	{
		m_rtgiMemoryUsage += bufferManager.second->GetTotalMemorySize();
	}

	auto tlasDesc = m_data->GetTlas()->GetDesc();
	D3D12_RESOURCE_ALLOCATION_INFO info = Graphics::s_device->Get()->GetResourceAllocationInfo(0, 1, &tlasDesc);
	m_rtgiMemoryUsage += info.SizeInBytes;

	Renderer::SetRTGIData(m_data->GetVoxelCommons());
	Renderer::UseRTGI(true);

	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);


	m_rtgiFence->Get()->SetName(L"RTGI Fence");
	m_rasterFence->Get()->SetName(L"Acc Fence");

	m_voxelScene = voxelScene;

	Commons::NetworkHost::InitializeEnet();
	m_networkServer.OnPeerConnected = std::bind(&ClusteredVoxelGIApp::OnClientConnected, this, std::placeholders::_1);
	m_networkServer.OnPacketReceived = std::bind(&ClusteredVoxelGIApp::OnPacketReceived, this, std::placeholders::_1);
	m_networkServer.OnPeerDisconnected = std::bind(&ClusteredVoxelGIApp::OnClientDisconnected, this, std::placeholders::_1);

}

void CVGI::ClusteredVoxelGIApp::OnPacketReceived(const Commons::NetworkPacket* packet)
{

	if (NetworkHost::CheckPacketHeader(packet, "CAMINP") || NetworkHost::CheckPacketHeader(packet, "LGTINP"))
	{
		bool isLightData = NetworkHost::CheckPacketHeader(packet, "LGTINP");

		auto& dataVector = packet->GetDataVector();

		UINT64 timeStamp = 0;
		DirectX::XMFLOAT3 clientVelocity;
		DirectX::XMFLOAT3 clientAbsPos;
		DirectX::XMFLOAT4 clientAbsRot;
		std::uint8_t clientInputBitmask = 0;

		// HEADER + NULL character
		size_t previousSize = 7;

		memcpy(&timeStamp, dataVector.data() + previousSize, sizeof(UINT64));
		previousSize += sizeof(UINT64);

		memcpy(&clientVelocity, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT3));
		previousSize += sizeof(DirectX::XMFLOAT3);

		memcpy(&clientAbsPos, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT3));
		previousSize += sizeof(DirectX::XMFLOAT3);

		memcpy(&clientAbsRot, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT4));
		previousSize += sizeof(DirectX::XMFLOAT4);

		memcpy(&clientInputBitmask, dataVector.data() + previousSize, sizeof(std::uint8_t));
		previousSize += sizeof(std::uint8_t);

		RemoteNodeController* controller = nullptr;

		if (isLightData)
		{
			auto* light = m_data->GetLightComponent();

			if (light == nullptr) return;

			float intensity = light->GetLightIntensity();
			DirectX::XMFLOAT3 color = light->GetLightColor();
			float closeVoxelStrength = m_lightTransportTechnique->GetCloseVoxelRadianceStrength();
			float farVoxelStrength = m_lightTransportTechnique->GetFarVoxelRadianceStrength();

			memcpy(&intensity, dataVector.data() + previousSize, sizeof(float));
			previousSize += sizeof(float);

			memcpy(&color, dataVector.data() + previousSize, sizeof(DirectX::XMFLOAT3));
			previousSize += sizeof(DirectX::XMFLOAT3);

			memcpy(&closeVoxelStrength, dataVector.data() + previousSize, sizeof(float));
			previousSize += sizeof(float);

			memcpy(&farVoxelStrength, dataVector.data() + previousSize, sizeof(float));
			previousSize += sizeof(float);

			light->SetLightIntensity(intensity);
			light->SetLightColor(color);
			m_lightTransportTechnique->SetCloseVoxelRadianceStrength(closeVoxelStrength);
			m_lightTransportTechnique->SetFarVoxelRadianceStrength(farVoxelStrength);

			controller = light->Node->GetComponent<RemoteNodeController>();
		}
		else
		{
			controller = m_data->GetCamera()->Node->GetComponent<RemoteNodeController>();
		}

		if (controller == nullptr) return;

		controller->FeedRemoteData(clientVelocity, clientAbsPos, clientAbsRot, timeStamp);
		
	}
	// Awaiting ACK for buffer data. If received we change state to listen for camera data.
	else if (NetworkHost::CheckPacketHeader(packet, "BUFFER"))
	{
		DXLIB_CORE_INFO("Client received buffers");
		m_isClientReadyForRadiance = true;
	}
	else if (NetworkHost::CheckPacketHeader(packet, "INIT"))
	{
		DXLIB_CORE_INFO("Received INIT packet");

		auto& voxelOccupiedBuffer = m_voxelizeScene->GetOccupiedVoxelBuffer();

		PacketGuard occupiedBufferPkt = m_networkServer.CreatePacket();
		occupiedBufferPkt->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		occupiedBufferPkt->ClearPacket();
		occupiedBufferPkt->AppendToBuffer("OCCVOX");
		occupiedBufferPkt->AppendToBuffer(voxelOccupiedBuffer);
		m_networkServer.SendData(occupiedBufferPkt);

		DXLIB_CORE_INFO("Send VOXOCC buffer with a size of: {0}", voxelOccupiedBuffer.size());

		auto& indRnkBuffer = m_prefixSumVoxels->GetIndirectionRankBuffer();

		PacketGuard indirectRankBufferPkt = m_networkServer.CreatePacket();
		indirectRankBufferPkt->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		indirectRankBufferPkt->ClearPacket();
		indirectRankBufferPkt->AppendToBuffer("INDRNK");
		indirectRankBufferPkt->AppendToBuffer(indRnkBuffer);
		m_networkServer.SendData(indirectRankBufferPkt);

		DXLIB_CORE_INFO("Send INDRNK buffer with a size of: {0}", indRnkBuffer.size());


		auto& indIdxBUffer = m_prefixSumVoxels->GetIndirectionIndexBuffer();


		PacketGuard indirectIndexBufferPkt = m_networkServer.CreatePacket();
		indirectIndexBufferPkt->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		indirectIndexBufferPkt->ClearPacket();
		indirectIndexBufferPkt->AppendToBuffer("INDIDX");
		indirectIndexBufferPkt->AppendToBuffer(indIdxBUffer);
		m_networkServer.SendData(indirectIndexBufferPkt);

		DXLIB_CORE_INFO("Send INDIDX buffer with a size of: {0}", indIdxBUffer.size());


		auto& cmpIdxBuffer = m_prefixSumVoxels->GetCompactedVoxelIndexBuffer();


		PacketGuard compactedIndicesPkt = m_networkServer.CreatePacket();
		compactedIndicesPkt->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		compactedIndicesPkt->ClearPacket();
		compactedIndicesPkt->AppendToBuffer("CMPIDX");
		compactedIndicesPkt->AppendToBuffer(cmpIdxBuffer);
		m_networkServer.SendData(compactedIndicesPkt);

		DXLIB_CORE_INFO("Send CMPIDX buffer with a size of: {0}", cmpIdxBuffer.size());

		auto& cmpHshBuffer = m_prefixSumVoxels->GetCompactedHashedBuffer();

		PacketGuard compactedHashesPkt = m_networkServer.CreatePacket();
		compactedHashesPkt->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		compactedHashesPkt->ClearPacket();
		compactedHashesPkt->AppendToBuffer("CMPHSH");
		compactedHashesPkt->AppendToBuffer(cmpHshBuffer);
		m_networkServer.SendData(compactedHashesPkt);

		DXLIB_CORE_INFO("Send CMPHSH buffer with a size of: {0}", cmpHshBuffer.size());
	}
	
}

void CVGI::ClusteredVoxelGIApp::OnClientConnected(const ENetPeer* peer)
{
	PacketGuard packet = m_networkServer.CreatePacket();
	packet->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
	packet->ClearPacket();
	packet->AppendToBuffer("VOX");
	packet->AppendToBuffer(VoxelTextureDimension.x);
	packet->AppendToBuffer(VoxelTextureDimension.y);
	packet->AppendToBuffer(VoxelTextureDimension.z);
	packet->AppendToBuffer(m_data->GetVoxelCount());
	packet->AppendToBuffer(m_data->GetClusterCount());

	m_networkServer.SendData(packet);

	RemoteNodeController* controller = m_data->GetCamera()->Node->GetComponent<RemoteNodeController>();
	CameraController* cameraController = m_data->GetCamera()->Node->GetComponent<CameraController>();

	if (controller != nullptr && cameraController != nullptr)
	{
		controller->SetRemoteControl(true);
		cameraController->IsEnabled = false;
	}

	controller = m_data->GetLightComponent()->Node->GetComponent<RemoteNodeController>();
	LightController* lightController = m_data->GetLightComponent()->Node->GetComponent<LightController>();
	if (controller != nullptr && lightController != nullptr)
	{
		controller->SetRemoteControl(true);
		lightController->IsEnabled = false;
	}
}

void CVGI::ClusteredVoxelGIApp::OnClientDisconnected(const ENetPeer* peer)
{
	m_isClientReadyForRadiance = false;
	m_firstRadianceSent = false;
	
	RemoteNodeController* controller = m_data->GetCamera()->Node->GetComponent<RemoteNodeController>();
	CameraController* cameraController = m_data->GetCamera()->Node->GetComponent<CameraController>();

	if (controller != nullptr && cameraController != nullptr)
	{
		controller->SetRemoteControl(false);
		cameraController->IsEnabled = true;
	}

	controller = m_data->GetLightComponent()->Node->GetComponent<RemoteNodeController>();
	LightController* lightController = m_data->GetLightComponent()->Node->GetComponent<LightController>();
	if (controller != nullptr && lightController != nullptr)
	{
		controller->SetRemoteControl(false);
		lightController->IsEnabled = true;
	}
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