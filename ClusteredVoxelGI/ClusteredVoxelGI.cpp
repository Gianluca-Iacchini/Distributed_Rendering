#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "CameraController.h"
#include "DX12Lib/Scene/LightController.h"



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

	m_rtgiFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);
	m_rasterFence = std::make_unique<Fence>(*Graphics::s_device, 0, 1);

	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);

	VoxelScene* voxelScene = static_cast<VoxelScene*>(this->m_Scene.get());


	m_data = std::make_shared<VOX::TechniqueData>();
	m_data->SetVoxelGridSize(VoxelTextureDimension);

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
	
	m_computeNeighboursTechnique->InitializeBuffers();
	m_computeNeighboursTechnique->PerformTechnique(computeContext);


	m_buildAABBsTechnique->InitializeBuffers();
	m_buildAABBsTechnique->PerformTechnique(computeContext);

	m_clusterVisibility->InitializeBuffers();
	m_data->SetTlas(m_clusterVisibility->BuildAccelerationStructures(computeContext));

	m_clusterVisibility->PerformTechnique(computeContext);

	m_sceneDepthTechnique->InitializeBuffers();

	m_lightVoxel->InitializeBuffers();

	m_lightTransportTechnique->InitializeBuffers();


	m_gaussianFilterTechnique->InitializeBuffers();
	m_gaussianFilterTechnique->SetIndirectCommandSignature(m_lightTransportTechnique->GetIndirectCommandSignature());
	m_gaussianFilterTechnique->InitializeGaussianConstants(computeContext);

	assert(foundLightComponent && "Failed to find light component with shadows enabled.");


	computeContext.Finish(true);

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

	m_cameraMovedSinceLastUpdate |= didCameraMove;
	m_lightChangedSinceLastUpdate |= didLightChange;

	bool shouldUpdateLight = m_lightChangedSinceLastUpdate && RTGIUpdateDelta > 0.15f;

	m_Scene->Render(commandContext);

	bool lerpLightUpdate = false;
	
	static float lastSendTime = GameTime::GetTotalTime();

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

				if (shouldUpdateLight)
				{
					m_lightVoxel->PerformTechnique(context);
					m_lightTransportTechnique->ResetRadianceBuffers(true);
					RTGIUpdateDelta = 0.0f;

					m_wasRadianceReset = 1;
				}

				// Compute visible faces
				m_lightTransportTechnique->PerformTechnique(context);
				// Compute GI for each face
				m_lightTransportTechnique->LaunchIndirectLightBlock(context, 1);
				// Gaussian double filter pass for each visible face
				m_gaussianFilterTechnique->PerformTechnique(context);
				m_gaussianFilterTechnique->PerformTechnique2(context);

				// Copy data to readback buffer if we're sending it to the client.
				if (m_receiveState == ReceiveState::CAMERA_DATA)
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
					packet->AppendToBuffer(m_lightTransportTechnique->GetVisibleFacesRadiance(visibleFacesCount), visibleFacesCount * sizeof(DirectX::XMUINT2));
					m_networkServer.SendData(packet);

					m_wasRadianceReset = 0;
				}
			}

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
	if (didCameraMove && !lerpLightUpdate)
	{
		Renderer::SetLerpMaxTime(1.0f);
		Renderer::SetDeltaLerpTime(1.0f);

	}
	// In all other cases we want to lerp. Radiance popping may still be noticeable but it's less jarring when the light changes.
	else
	{
		Renderer::SetLerpMaxTime(0.15f);
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