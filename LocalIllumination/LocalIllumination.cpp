#include "LocalIllumination.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"

#include "DX12Lib/Encoder/FFmpegStreamer.h"

#include "LIUtils.h"
#include "Technique.h"
#include "./Data/Shaders/Include/GaussianOnly_CS.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"
#include "UIHelpers.h"

#include "DX12Lib/Scene/CameraController.h"
#include "DX12Lib/Scene/LightController.h"
#include "DX12Lib/Scene/RemoteNodeController.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;
using namespace Commons;
using namespace LI;

using UploadBufferFencePair = std::pair<std::shared_ptr<DX12Lib::UploadBuffer>, UINT64>;

LocalIlluminationApp::~LocalIlluminationApp()
{
	this->FlushCommandQueue();
}


void LocalIlluminationApp::OnPacketReceivedClient(const NetworkPacket* packet)
{

	if (NetworkHost::CheckPacketHeader(packet, "RDXBUF"))
	{
		std::size_t pktSize = packet->GetSize() - 7;
		std::size_t vecSize = pktSize / sizeof(DirectX::XMUINT2);

		UploadBufferFencePair pair;

		{
			std::lock_guard<std::mutex> lock(m_vectorMutex);

			if (m_ReadyToWriteBuffers.empty())
				return;

			pair = m_ReadyToWriteBuffers.front();
		}

		// Wait for the first buffer to complete its copy operation before we can write to it.
		m_bufferFence->WaitForFence(pair.second);

		{
			std::lock_guard<std::mutex> lock(m_vectorMutex);
			// Assign again in case the queue was modified while waiting for the fence
			pair = m_ReadyToWriteBuffers.front();
			m_ReadyToWriteBuffers.pop();
		}

		auto& uploadBuffer = pair.first;

		// RADBUF + NULL character
		UINT totalBytes = 7;

		UINT nFaces = 0;
		UINT shouldReset = 0;

		void* mappedData = uploadBuffer->Map();

		memcpy(&nFaces, packet->GetDataVector().data() + 7, sizeof(UINT));
		totalBytes += sizeof(UINT);
		memcpy(&shouldReset, packet->GetDataVector().data() + totalBytes, sizeof(UINT));
		totalBytes += sizeof(UINT);
		memcpy(mappedData, packet->GetDataVector().data() + totalBytes, vecSize * sizeof(DirectX::XMUINT2));


		// The buffer can now be used in the main thread command list.
		{
			NetworkRadianceBufferInfo buffInfo;
			buffInfo.buffer = uploadBuffer;
			buffInfo.nFaces = nFaces;
			buffInfo.ShouldReset = shouldReset;
			std::lock_guard<std::mutex> lock(m_vectorMutex);
			m_ReadyToCopyBuffer.push(buffInfo);
		}
	}
	// To ensure that the server sent the initialization message, the message starts with "VOX" (4 bytes due to null character)
	// Then each float is 4 bytes long.

	else if (NetworkHost::CheckPacketHeader(packet, "VOX"))
	{
		auto& dataVector = packet->GetDataVector();
		DirectX::XMUINT3 voxelizationSize;

		// VOX + NULL character
		size_t previousSize = 4;

		memcpy(&voxelizationSize, dataVector.data() + previousSize, sizeof(DirectX::XMUINT3));

		DXLIB_INFO("Received voxelization data with size: [{0},{1},{2}]", voxelizationSize.x, voxelizationSize.y, voxelizationSize.z);

		previousSize += sizeof(DirectX::XMUINT3);
		UINT32 voxelCount = 0;
		memcpy(&voxelCount, dataVector.data() + previousSize, sizeof(UINT));


		previousSize += sizeof(UINT);
		UINT32 clusterCount = 0;
		memcpy(&clusterCount, dataVector.data() + previousSize, sizeof(UINT));


		DXLIB_INFO("Received voxelization data with voxel count: {0} and cluster count: {1}", voxelCount, clusterCount);

		UINT32 faceCount = voxelCount * 6;
		for (UINT i = 0; i < 3; i++)
		{
			std::shared_ptr<DX12Lib::UploadBuffer> uploadBuffer = std::make_shared<DX12Lib::UploadBuffer>();
			uploadBuffer->Create(faceCount * sizeof(DirectX::XMUINT2));

			m_ReadyToWriteBuffers.push(std::make_pair(uploadBuffer, 0));
		}


		DX12Lib::AABB sceneBounds = GetSceneAABBExtents();
		DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3((sceneBounds.Max.x - sceneBounds.Min.x) / voxelizationSize.x,
			(sceneBounds.Max.y - sceneBounds.Min.y) / voxelizationSize.y,
			(sceneBounds.Max.z - sceneBounds.Min.z) / voxelizationSize.z);

		m_data->SetSceneAABB(sceneBounds);
		m_data->SetVoxelGridSize(voxelizationSize);
		m_data->SetVoxelCellSize(voxelCellSize);
		m_data->SetClusterCount(clusterCount);
		m_data->SetVoxelCount(voxelCount);
		m_data->BuildMatrices();
		Renderer::SetRTGIData(m_data->GetVoxelCommons());

		m_radianceFromNetworkTechnique->InitializeBuffers();
		m_lightTransportTechnique->InitializeBuffers();

		m_gaussianFilterTechnique->InitializeBuffers();
		m_gaussianFilterTechnique->SetIndirectCommandSignature(m_lightTransportTechnique->GetIndirectCommandSignature());


		UINT32 voxelLinearSIze = voxelizationSize.x * voxelizationSize.y * voxelizationSize.z;
		UINT32 voxelBitmapSize = (voxelLinearSIze + 31) / 32;

		m_voxelBufferManager->AddStructuredBuffer(voxelBitmapSize, sizeof(UINT32));
		m_voxelBufferManager->AllocateBuffers();

		m_prefixSumBufferManager->AddStructuredBuffer(voxelizationSize.y * voxelizationSize.z, sizeof(UINT32));
		m_prefixSumBufferManager->AddStructuredBuffer(voxelizationSize.y * voxelizationSize.z, sizeof(UINT32));
		m_prefixSumBufferManager->AddStructuredBuffer(m_data->GetVoxelCount(), sizeof(UINT32));
		m_prefixSumBufferManager->AddStructuredBuffer(m_data->GetVoxelCount(), sizeof(UINT32));
		m_prefixSumBufferManager->AllocateBuffers();


		PacketGuard packet = m_networkClient.CreatePacket();
		packet->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
		packet->ClearPacket();
		packet->AppendToBuffer("INIT");
		m_networkClient.SendData(packet);
	}
	else
	{
		for (int bf = 0; bf < 5; bf++)
		{
			if (NetworkHost::CheckPacketHeader(packet, packetHeaders[bf]))
			{
				std::size_t pktSize = packet->GetSize() - 7;
				std::size_t vecSize = pktSize / sizeof(UINT32);
				DXLIB_INFO("Received packet with header: {0} and vector size: {1}", packetHeaders[bf], vecSize);

				DX12Lib::GPUBuffer& currentBuffer = (bf == 4) ? m_voxelBufferManager->GetBuffer(0) : m_prefixSumBufferManager->GetBuffer(bf);

				// If the previous fence has not been signaled, then we have to wait before we can write data to the
				// Upload buffer, since the GPU might still be using it.

				UploadBufferFencePair pair;

				{
					std::lock_guard<std::mutex> lock(m_vectorMutex);
					pair = m_ReadyToWriteBuffers.front();
				}

				m_bufferFence->WaitForFence(pair.second);

				{
					std::lock_guard<std::mutex> lock(m_vectorMutex);
					// Assign again in case the queue was modified while waiting for the fence
					pair = m_ReadyToWriteBuffers.front();
					m_ReadyToWriteBuffers.pop();
				}

				auto& uploadBuffer = pair.first;

				void* mappedData = uploadBuffer->Map();

				memcpy(mappedData, packet->GetDataVector().data() + 7, vecSize * sizeof(UINT32));

				DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();

				// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
				context.TransitionResource(currentBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
				context.m_commandList->Get()->CopyBufferRegion(currentBuffer.Get(), 0, uploadBuffer->Get(), 0, vecSize * sizeof(UINT32));
				context.TransitionResource(currentBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
				UINT64 fenceVal = context.Finish();

				// Since the copy is done in the same thread, we can just put the buffer in the write queue again
				{
					std::lock_guard<std::mutex> lock(m_vectorMutex);
					m_ReadyToWriteBuffers.push(UploadBufferFencePair(pair.first, fenceVal));
				}

				m_bufferFence->CurrentFenceValue = fenceVal;
				Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);

				// This is not a very good way to check if all buffers have been initialized, as if the server sends the same buffer multiple
				// times it will increment the counter. However, for the purposes of this demo it is enough.
				m_buffersInitialized++;
				if (m_buffersInitialized >= NUM_BASIC_BUFFERS)
				{
					PacketGuard packet = m_networkClient.CreatePacket();
					packet->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
					packet->ClearPacket();
					packet->AppendToBuffer("BUFFER");
					m_networkClient.SendData(packet);

					InitializeBasicBuffers();
					m_buffersInitialized = 0;

					m_isReadyForRadiance = true;
				}

			}
		}
	}
	
}

void LI::LocalIlluminationApp::OnPeerDisconnectedClient(const ENetPeer* peer)
{
	m_isReadyForRadiance = false;
	Renderer::UseRTGI(false);
	m_isInitialized = false;
}

void LI::LocalIlluminationApp::OnPacketReceivedServer(const Commons::NetworkPacket* packet)
{
	if (NetworkHost::CheckPacketHeader(packet, "STRINP"))
	{
		auto dataVector = packet->GetDataVector();

		// HEADER + NULL character
		size_t previousSize = 7;

		UINT64 timeStamp = 0;
		std::uint8_t cameraInputBitmas = 0;
		float mouseDeltaXY[2];
		std::uint8_t lightInputBitmask = 0;
		float clientDeltaTime = 0.0f;



		memcpy(&timeStamp, dataVector.data() + previousSize, sizeof(UINT64));
		previousSize += sizeof(UINT64);

		memcpy(&cameraInputBitmas, dataVector.data() + previousSize, sizeof(std::uint8_t));
		previousSize += sizeof(std::uint8_t);

		memcpy(mouseDeltaXY, dataVector.data() + previousSize, sizeof(float) * 2);
		previousSize += (sizeof(float) * 2);

		memcpy(&lightInputBitmask, dataVector.data() + previousSize, sizeof(std::uint8_t));
		previousSize += sizeof(std::uint8_t);

		memcpy(&clientDeltaTime, dataVector.data() + previousSize, sizeof(float));
		previousSize += sizeof(float);

		DX12Lib::RemoteNodeController* controller = m_LIScene->GetMainCamera()->Node->GetComponent<RemoteNodeController>();

		if (controller != nullptr)
		{
			controller->SetRemoteControl(true);
			m_LIScene->GetCameraController()->IsEnabled = false;
			controller->FeedRemoteData(DirectX::XMFLOAT2(mouseDeltaXY[0], mouseDeltaXY[1]), cameraInputBitmas, timeStamp, clientDeltaTime);
		}

		auto lightNode = m_LIScene->GetMainLight()->Node;

		lightNode->GetComponent<LightController>()->IsEnabled = false;

		DirectX::XMFLOAT2 lightRotationInput = DirectX::XMFLOAT2(0.0f, 0.0f);
			
		lightRotationInput.x += (lightInputBitmask & (1 << 0)) ? 1.0f : 0.0f;
		lightRotationInput.x += (lightInputBitmask & (1 << 1)) ? -1.0f : 0.0f;

		lightRotationInput.y += (lightInputBitmask & (1 << 2)) ? -1.0f : 0.0f;
		lightRotationInput.y += (lightInputBitmask & (1 << 3)) ? 1.0f : 0.0f;


		float dt = (GameTime::GetTimeSinceEpoch() - timeStamp) / 1000000.0f;

		lightRotationInput.x *= dt / clientDeltaTime;
		lightRotationInput.y *= dt / clientDeltaTime;


		lightNode->Rotate(lightNode->GetRight(), lightRotationInput.y);
		lightNode->Rotate(lightNode->GetUp(), lightRotationInput.x);

	}
	else if (NetworkHost::CheckPacketHeader(packet, "STR264"))
	{
		if (!m_isStreaming)
		{
			this->OpenStream(false);
		}
	}
	else if (NetworkHost::CheckPacketHeader(packet, "STR265"))
	{
		if (!m_isStreaming)
		{
			this->OpenStream(true);
		}
	}
}

void LI::LocalIlluminationApp::OnPeerConnectedServer(const ENetPeer* peer)
{
	DXLIB_CORE_INFO("Peer connected: {0}", peer->address.host);
}

DX12Lib::AABB LocalIlluminationApp::GetSceneAABBExtents()
{
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
	float maxComponent = max(sceneBounds.Max.x, max(sceneBounds.Max.y, sceneBounds.Max.z));

	float extent = maxComponent - minComponent;

	sceneBounds.Min.x = minComponent;
	sceneBounds.Min.y = minComponent;
	sceneBounds.Min.z = minComponent;

	sceneBounds.Max.x = maxComponent;
	sceneBounds.Max.y = maxComponent;
	sceneBounds.Max.z = maxComponent;

	return sceneBounds;
}

void LI::LocalIlluminationApp::CopyDataToBasicBuffer(UINT bufferIdx)
{
}

void LI::LocalIlluminationApp::InitializeBasicBuffers()
{

	m_bufferFence->WaitForCurrentFence();
	DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();
	m_gaussianFilterTechnique->InitializeGaussianConstants(context);
	context.Finish(true);


	DescriptorHandle& rendererRTGIHandle = Renderer::GetRTGIHandleSRV();

	D3D12_CPU_DESCRIPTOR_HANDLE srvHandles[6];

	srvHandles[0] = m_voxelBufferManager->GetBuffer(0).GetSRV();

	for (UINT i = 1; i < 5; i++)
	{
		srvHandles[i] = m_prefixSumBufferManager->GetBuffer(i - 1).GetSRV();
	}

	srvHandles[5] = m_data->GetBufferManager(VOX::GaussianFilterTechnique::ReadName).GetBuffer(0).GetSRV();

	auto descriptorSize = Renderer::s_textureHeap->GetDescriptorSize();
	for (UINT i = 0; i < 6; i++)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(1, rendererRTGIHandle + descriptorSize * i, srvHandles[i], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Renderer::UseRTGI(true);

	// Technically we should use an atomic operation here, but since this is only for the UI and it is checked every frame anyway we should be fine.
	m_isInitialized = true;
}

void LI::LocalIlluminationApp::ShowIMGUIWindow()
{
	float appX = static_cast<float>(Renderer::s_clientWidth);
	float appY = static_cast<float>(Renderer::s_clientHeight);

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

	float maxX = ImGui::CalcTextSize("\t").x;
	UIHelpers::ControlInfoBlock(m_isStreaming);

	bool isConnectedAsClient = m_networkClient.IsConnected() && m_networkClient.HasPeers() && m_isReadyForRadiance;

	if (ImGui::CollapsingHeader("Networking", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::SeparatorText("Radiance");

		static float connectionTime = 0.0f;
		static bool isWaitingForConnection = false;
		float connectionTimeout = 8.0f;

		if (ImGui::Button("Start Server"))
		{
			m_networkClient.StartServer(1234);
		}


		if (!isConnectedAsClient)
		{
			ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Not connected");
			ImGui::BeginDisabled(isWaitingForConnection);
			const char* buttonText = isWaitingForConnection ? "Connecting..." : "Connect to server";


			//ImGui::InputText("Server address", m_serverAddress, 16);

			if (ImGui::Button(buttonText))
			{
				//m_networkClient.Connect(m_serverAddress, 1234);
				isWaitingForConnection = true;
				connectionTime = 0.0f;
			}

		
			ImGui::EndDisabled();


			if (isWaitingForConnection)
			{
				connectionTime += ImGui::GetIO().DeltaTime;
			}

			if (connectionTime >= connectionTimeout)
			{
				ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Could not connect to server: Timeout");
				m_networkClient.Disconnect();
				isWaitingForConnection = false;
				m_isReadyForRadiance = false;
				m_isInitialized = false;
				Renderer::UseRTGI(false);
			}
		}
		else
		{
			isWaitingForConnection = m_isInitialized;
			connectionTime = 0.0f;

			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Server is running");
			if (ImGui::Button("Disconnect"))
			{
				m_networkClient.Disconnect();
				m_isReadyForRadiance = false;
				Renderer::UseRTGI(false);
				m_isInitialized = false;
				isWaitingForConnection = false;
			}
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected to Server at: %s, RTT: %u ms", m_networkClient.GetPeerAddress().c_str(), m_networkClient.GetPing());
		}

		ImGui::SeparatorText("Streaming");

		if (m_isStreaming)
		{
			std::string codecName = "No codec selected";
			if (m_ffmpegStreamer)
			{
				AVCodecID codecId = m_ffmpegStreamer->GetCodecID();
				if (codecId == AV_CODEC_ID_HEVC)
				{
					codecName = "HEVC";
				}
				else if (codecId == AV_CODEC_ID_H264)
				{
					codecName = "H264";
				}
			}

			if (m_networkServer.IsConnected() && m_networkServer.HasPeers())
			{
				Commons::UIHelpers::ConnectedClient(m_networkServer.GetPeerAddress().c_str(), m_networkServer.GetPing());
			}
		}
	}

	if (ImGui::CollapsingHeader("Light"))
	{
		auto* light = m_LIScene->GetMainLight();

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

		ImGui::BeginDisabled(!isConnectedAsClient);
		ImGui::Text("Far voxels bounce strength:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Sets the intensity of indirect light gathered from far voxels.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##FarStrength", &m_farVoxelStrength, 0.0f, 7.0f))
		{
			m_indirectSettingChanged = true;
		}

		ImGui::Text("Close voxels bounce strength:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Sets the intensity of indirect light gathered from nearby voxels.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderFloat("##CloseStrength", &m_closeVoxelStrength, 0.0f, 7.0f))
		{
			m_indirectSettingChanged = true;
		}

		ImGui::SeparatorText("Update frequency");

		ImGui::Text("Lerp update frequency:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time to lerp between previous radiance values and new radiance values.");
		}
		ImGui::SameLine(maxX);
		ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
		ImGui::SliderFloat("##LerpFrequency", &m_lerpMaxTime, 0.0f, 1.0f);
		ImGui::EndDisabled();

		ImGui::Separator();

		if (ImGui::Button("Reset settings"))
		{
			light->SetLightColor(DirectX::XMFLOAT3(0.45f, 0.45f, 0.45f));
			light->SetLightIntensity(1.0f);

			m_lerpMaxTime = 0.2f;
		}
	}

	ImGui::BeginDisabled(!isConnectedAsClient);
	if (ImGui::CollapsingHeader("Voxelization Info"))
	{
		maxX = ImGui::CalcTextSize("Compute Neighbours Time:\t").x;

		DirectX::XMUINT3 voxelGridSize = m_data->GetVoxelGridSize();
		ImGui::Text("Voxel Grid Size: (%d x %d x %d)", voxelGridSize.x, voxelGridSize.y, voxelGridSize.z);
		ImGui::Text("Voxel Count: %d\tCluster Count: %d", m_data->GetVoxelCount(), m_data->GetClusterCount());

		ImGui::SeparatorText("Real-time GI timings (latest dispatch)");

		ImGui::Text("Process network time:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to fill the radiance buffer from the network data.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_processNetworkBufferTime);

		ImGui::Text("Visible voxels:");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
			ImGui::SetTooltip("Time needed to compute voxels visible from the main camera.");
		}
		ImGui::SameLine(maxX);
		ImGui::Text("%.2f ms", m_visibleVoxelTime);

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

		float totalTime = m_visibleVoxelTime + m_firstGaussianFilterTime + m_secondGaussianFilterTime;
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
		
	}

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

		const char* passCountItems[] = { "No passes", "One pass", "Two passes" };
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
	ImGui::EndDisabled();

	ImGui::End();
	
}



void LocalIlluminationApp::Initialize(GraphicsContext& context)
{

	std::string sourcePath = std::string(SOURCE_DIR);


	sourcePath += std::string("\\..\\Models\\PBR\\sponza2.gltf");



	bool loaded = this->m_Scene->AddFromFile(sourcePath.c_str());

	assert(loaded && "Model not loaded");

	Commons::NetworkHost::InitializeEnet();


	m_networkClient.OnPacketReceived = std::bind(&LocalIlluminationApp::OnPacketReceivedClient, this, std::placeholders::_1); 
	m_networkClient.OnPeerDisconnected = std::bind(&LocalIlluminationApp::OnPeerDisconnectedClient, this, std::placeholders::_1);

	m_networkServer.OnPacketReceived = std::bind(&LocalIlluminationApp::OnPacketReceivedServer, this, std::placeholders::_1);
	m_networkServer.OnPeerConnected = std::bind(&LocalIlluminationApp::OnPeerConnectedServer, this, std::placeholders::_1);

	m_LIScene = dynamic_cast<LI::LIScene*>(m_Scene.get());

	assert(m_LIScene != nullptr && "Error when initializing LI scene.");

	m_LIScene->Init(context);
	
	m_bufferFence = std::make_unique<DX12Lib::Fence>(*Graphics::s_device, 0, 1);
	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);

	m_rasterFence = std::make_unique<DX12Lib::Fence>(*Graphics::s_device, 0, 1);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);

	m_rtgiFence = std::make_unique<DX12Lib::Fence>(*Graphics::s_device, 0, 1);
	Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rtgiFence);


	m_data = std::make_shared<VOX::TechniqueData>();
	m_data->SetCamera(m_Scene->GetMainCamera());

	m_voxelBufferManager = std::make_shared<VOX::BufferManager>();
	m_prefixSumBufferManager = std::make_shared<VOX::BufferManager>();

	m_data->SetBufferManager(VOXELIZE_SCENE, m_voxelBufferManager);
	m_data->SetBufferManager(PREFIX_SUM, m_prefixSumBufferManager);

	m_sceneDepthTechnique = std::make_shared<VOX::SceneDepthTechnique>(m_data, true);
	m_sceneDepthTechnique->InitializeBuffers();

	auto shaderBytecode = CD3DX12_SHADER_BYTECODE((void*)g_pGaussianOnly_CS, ARRAYSIZE(g_pGaussianOnly_CS));
	m_lightTransportTechnique = std::make_shared<VOX::LightTransportTechnique>(m_data, false);
	m_lightTransportTechnique->BuildPipelineState(shaderBytecode);

	m_radianceFromNetworkTechnique = std::make_shared<LI::RadianceFromNetworkTechnique>(m_data);
	m_radianceFromNetworkTechnique->BuildPipelineState();

	m_gaussianFilterTechnique = std::make_shared<VOX::GaussianFilterTechnique>(m_data);
	m_gaussianFilterTechnique->BuildPipelineState();

	m_timingQueryHandle = Graphics::s_queryHeap->Alloc(5);
	m_timingReadBackBuffer.Create(5, sizeof(UINT64));

	m_networkServer.StartServer(2345);
}

void LocalIlluminationApp::Update(GraphicsContext& context)
{
	DX12Lib::SceneCamera* camera = this->m_Scene->GetMainCamera();


	DX12Lib::LightComponent* light = m_LIScene->GetMainLight();

	bool isCameraDirty = camera != nullptr && camera->IsDirty();
	bool isLightDirty = light != nullptr && (light->Node->IsTransformDirty() || light->DidLightPropertyChange());

	D3DApp::Update(context);

	auto kbState = Graphics::s_keyboard->GetState();
	m_kbTracker.Update(kbState);

	if (m_kbTracker.pressed.Escape)
	{
		PostQuitMessage(0);
	}

	if ((m_isReadyForRadiance))
	{
		std::uint8_t cameraInputBitMask = 0;
		std::uint8_t lightInputBitMask = 0;


		if (kbState.W)
			cameraInputBitMask |= 1 << 0;
		if (kbState.S)
			cameraInputBitMask |= 1 << 1;
		if (kbState.A)
			cameraInputBitMask |= 1 << 2;
		if (kbState.D)
			cameraInputBitMask |= 1 << 3;
		if (kbState.E)
			cameraInputBitMask |= 1 << 4;
		if (kbState.Q)
			cameraInputBitMask |= 1 << 5;

		if (kbState.Up)
			lightInputBitMask |= 1 << 0;
		if (kbState.Down)
			lightInputBitMask |= 1 << 1;
		if (kbState.Left)
			lightInputBitMask |= 1 << 2;
		if (kbState.Right)
			lightInputBitMask |= 1 << 3;

		if (camera != nullptr && ((m_lastCameraBitMask != cameraInputBitMask) || isCameraDirty))
		{

			CameraController* cameraContr = camera->Node->GetComponent<CameraController>();

			PacketGuard packet = m_networkClient.CreatePacket();
			packet->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
			packet->ClearPacket();
			packet->AppendToBuffer("CAMINP");
			packet->AppendToBuffer(GameTime::GetTimeSinceEpoch());
			packet->AppendToBuffer(cameraContr->GetVelocity());
			packet->AppendToBuffer(camera->Node->GetPosition());
			packet->AppendToBuffer(camera->Node->GetRotationQuaternion());
			packet->AppendToBuffer(cameraInputBitMask);
			m_networkClient.SendData(packet);
		}

		if (light != nullptr && ((m_lastLightBitMask != lightInputBitMask) || isLightDirty || m_indirectSettingChanged))
		{
			DirectX::XMFLOAT3 velocity = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

			PacketGuard packet = m_networkClient.CreatePacket();
			packet->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);
			packet->ClearPacket();
			packet->AppendToBuffer("LGTINP");
			packet->AppendToBuffer(GameTime::GetTimeSinceEpoch());
			packet->AppendToBuffer(velocity);
			packet->AppendToBuffer(light->Node->GetPosition());
			packet->AppendToBuffer(light->Node->GetRotationQuaternion());
			packet->AppendToBuffer(lightInputBitMask);
			packet->AppendToBuffer(light->GetLightIntensity());
			packet->AppendToBuffer(light->GetLightColor());
			packet->AppendToBuffer(m_closeVoxelStrength);
			packet->AppendToBuffer(m_farVoxelStrength);

			m_indirectSettingChanged = false;

			m_networkClient.SendData(packet);
		}

		m_lastLightBitMask = lightInputBitMask;
		m_lastCameraBitMask = cameraInputBitMask;

		sendPacketDeltaTime = 0.0f;
	}

	sendPacketDeltaTime += GameTime::GetDeltaTime();
}

void LocalIlluminationApp::Draw(GraphicsContext& context)
{

	Renderer::SetUpRenderFrame(context);

	ShowIMGUIWindow();

	this->m_Scene->Render(context);


	if (m_isReadyForRadiance)
	{
		NetworkRadianceBufferInfo buffInfo;
		buffInfo.buffer = nullptr;
		buffInfo.nFaces = 0;
		buffInfo.ShouldReset = 0;

		bool isRTGIDataAvailable = false;

		{
			std::lock_guard<std::mutex> lock(m_vectorMutex);
			isRTGIDataAvailable = !m_ReadyToCopyBuffer.empty();
		}

		auto kbState = Graphics::s_keyboard->GetState();

		bool didCameraMove = m_data->GetCamera()->IsDirty();
		m_cameraMovedSinceLastUpdate |= didCameraMove;


		if (m_rtgiFence->IsFenceComplete(m_rtgiFence->CurrentFenceValue))
		{
			if (!LightDispatched && (isRTGIDataAvailable || didCameraMove))
			{
				if (m_rasterFence->IsFenceComplete(m_rasterFenceValue))
				{
					if (isRTGIDataAvailable)
					{
						std::lock_guard<std::mutex> lock(m_vectorMutex);
						buffInfo = m_ReadyToCopyBuffer.front();
						m_ReadyToCopyBuffer.pop();
					}

					bool shouldResetBuffers = (buffInfo.ShouldReset != 0);

					DX12Lib::ComputeContext& computeContext = DX12Lib::ComputeContext::Begin();

					m_lightTransportTechnique->ClearRadianceBuffers(computeContext, shouldResetBuffers);

					computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 0);
					if (isRTGIDataAvailable)
					{
						UINT64 fenceVal = m_radianceFromNetworkTechnique->ProcessNetworkData(computeContext, buffInfo.buffer.get(), buffInfo.nFaces, buffInfo.ShouldReset);
						m_bufferFence->CurrentFenceValue = fenceVal;
						Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);
						{
							std::lock_guard<std::mutex> lock(m_vectorMutex);
							m_ReadyToWriteBuffers.push(std::make_pair(buffInfo.buffer, fenceVal));
						}
					}
					computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 1);

					m_lightTransportTechnique->ComputeVisibleFaces(computeContext);
					computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 2);
					m_gaussianFilterTechnique->PerformTechnique(computeContext);
					computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 3);
					m_gaussianFilterTechnique->PerformTechnique2(computeContext);
					computeContext.EndQuery(*Graphics::s_queryHeap, m_timingQueryHandle, 4);
					computeContext.ResolveQueryData(*Graphics::s_queryHeap, m_timingQueryHandle, m_timingReadBackBuffer, 5);

					m_rtgiFence->CurrentFenceValue = computeContext.Finish();
					Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_rtgiFence);

					LightDispatched = true;
				}
			}
			else if (LightDispatched)
			{
				m_radianceReady = true;
				LightDispatched = false;
				m_cameraMovedSinceLastUpdate = false;

				UINT64* timingData = reinterpret_cast<UINT64*>(m_timingReadBackBuffer.ReadBack(sizeof(UINT64) * 5));
				m_processNetworkBufferTime = ((timingData[1] - timingData[0]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
				m_visibleVoxelTime = ((timingData[2] - timingData[1]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
				m_firstGaussianFilterTime = ((timingData[3] - timingData[2]) * 1000.0f) / Graphics::GetComputeGPUFrequency();
				m_secondGaussianFilterTime = ((timingData[4] - timingData[3]) * 1000.0f) / Graphics::GetComputeGPUFrequency();

				m_accTotalTime += m_processNetworkBufferTime + m_visibleVoxelTime + m_firstGaussianFilterTime + m_secondGaussianFilterTime;

				m_lightDispatchCount += 1;
			}
		}

		if (m_radianceReady)
		{
			Renderer::ResetLerpTime();
			m_lerpDeltaTime = 0.0f;
		}

		if (didCameraMove)
		{
			Renderer::SetLerpMaxTime(0.1f);
			Renderer::SetDeltaLerpTime(m_lerpDeltaTime);
		}
		else
		{
			Renderer::SetLerpMaxTime(0.25f);
			Renderer::SetDeltaLerpTime(m_lerpDeltaTime);
		}
	}

	// Render Layers
	{
		// Update depth cameras used to compute GI.
		// Ideally we would only update the camera if the light changed or camera moved, but this is a cheap operation so we can get away with doing it every frame.
		if (m_radianceReady)
		{
			m_sceneDepthTechnique->UpdateCameraMatrices();
			m_sceneDepthTechnique->PerformTechnique(context);
		}


		{
			Renderer::ShadowPass(context);


			Renderer::MainRenderPass(context);

			if (m_isReadyForRadiance)
				Renderer::LerpRadiancePass(context);

			Renderer::DeferredPass(context);
			Renderer::PostProcessPass(context);

			if (m_isStreaming)
			{
				this->StreamScene(context);
			}


			Renderer::UIPass(context, m_isStreaming);

		}

	}

	if (m_radianceReady)
	{
		m_rasterFence->CurrentFenceValue = context.Flush();
		Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);
		m_rasterFenceValue = m_rasterFence->CurrentFenceValue;
	}
	
		



	m_lerpDeltaTime += GameTime::GetDeltaTime();

	m_radianceReady = false;

	Renderer::PostDrawCleanup(context);
}

void LocalIlluminationApp::OnClose(GraphicsContext& context)
{
	m_networkClient.Disconnect();
	Commons::NetworkHost::DeinitializeEnet();
	D3DApp::OnClose(context);
}

void LI::LocalIlluminationApp::StreamScene(DX12Lib::CommandContext& context)
{
	// Accumulator is used to ensure proper frame rate for the encoder

	float totTime = GameTime::GetTotalTime();
	float encodeDeltaTime = totTime - m_lastUpdateTime;
	m_lastUpdateTime = totTime;
	m_accumulatedTime += encodeDeltaTime;

	float encoderFramerate = 1.f / m_ffmpegStreamer->GetEncoder().maxFrames;

	auto& backBuffer = Renderer::GetCurrentBackBuffer();

	if (m_accumulatedTime >= (encoderFramerate))
	{
		m_accumulatedTime -= encoderFramerate;
		m_ffmpegStreamer->Encode(context, backBuffer);
	}
}

void LI::LocalIlluminationApp::OpenStream(bool useHevc)
{
	AVCodecID codecId = useHevc ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;

	m_ffmpegStreamer = std::make_unique<FFmpegStreamer>();
	m_ffmpegStreamer->OpenStream(Renderer::s_clientWidth, Renderer::s_clientHeight, "", codecId);
	m_ffmpegStreamer->StartStreaming();
	m_isStreaming = true;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		LI::LocalIlluminationApp app(hInstance, new LI::LIScene());
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