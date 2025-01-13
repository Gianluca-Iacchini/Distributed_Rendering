#define STREAMING 0
#define NETWORK_RADIANCE 1

#include "LocalIllumination.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "LIUtils.h"
#include "Technique.h"
#include "./Data/Shaders/Include/GaussianOnly_CS.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;
using namespace LI;

using UploadBufferFencePair = std::pair<std::shared_ptr<DX12Lib::UploadBuffer>, UINT64>;

LocalIlluminationApp::~LocalIlluminationApp()
{
	this->FlushCommandQueue();
}


void LocalIlluminationApp::OnPacketReceived(const NetworkPacket* packet)
{
	if (m_receiveState == ReceiveState::RADIANCE)
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
	}
	else if (m_receiveState == ReceiveState::BASIC_BUFFERS)
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

					m_receiveState = ReceiveState::RADIANCE;

					DXLIB_INFO("All buffers initialized");
					PacketGuard packet = m_networkClient.CreatePacket();
					packet->ClearPacket();
					packet->AppendToBuffer("BUFFER");
					m_networkClient.SendData(packet);

					{
						std::lock_guard<std::mutex> lock(m_mainThreadMutex);
						m_isMainThreadReady = true;
					}

					m_mainThreadCV.notify_one();
				}
				
			}
		}
	}
	else if (m_receiveState == ReceiveState::INITIALIZATION)
	{
		// To ensure that the server sent the initialization message, the message starts with "VOX" (4 bytes due to null character)
		// Then each float is 4 bytes long.

		if (NetworkHost::CheckPacketHeader(packet, "VOX"))
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

			m_receiveState = ReceiveState::BASIC_BUFFERS;


			PacketGuard packet = m_networkClient.CreatePacket();
			packet->ClearPacket();
			packet->AppendToBuffer("INIT");
			m_networkClient.SendData(packet);
		}
	}
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



void LocalIlluminationApp::Initialize(GraphicsContext& context)
{

	std::string sourcePath = std::string(SOURCE_DIR);

	if (m_usePBRMaterials)
		sourcePath += std::string("\\Models\\PBR\\sponza2.gltf");
	else
		sourcePath += std::string("\\Models\\sponza_nobanner.obj");


	bool loaded = this->m_Scene->AddFromFile(sourcePath.c_str());

	assert(loaded && "Model not loaded");

	DX12Lib::NetworkHost::InitializeEnet();


	m_networkClient.OnPacketReceived = std::bind(&LocalIlluminationApp::OnPacketReceived, this, std::placeholders::_1); 



	s_mouse->SetMode(Mouse::MODE_RELATIVE);

	if (!m_usePBRMaterials)
	{
		auto rootNode = m_Scene->GetRootNode();

		rootNode->SetScale(0.01f, 0.01f, 0.01f);
	}

	this->m_Scene->Init(context);


#if NETWORK_RADIANCE
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

	m_networkClient.Connect("127.0.0.1", 1234);

	std::unique_lock<std::mutex> lock(m_mainThreadMutex);
	
	if (m_mainThreadCV.wait_for(lock, std::chrono::seconds(5), [this] { return m_isMainThreadReady; }))
	{
		DXLIB_INFO("Initialization complete");

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
	}
	else
	{
		m_bufferFence->WaitForCurrentFence();
		DXLIB_ERROR("Timeout waiting for initialization");
	}

#endif
}

void LocalIlluminationApp::Update(GraphicsContext& context)
{
	D3DApp::Update(context);

	auto kbState = Graphics::s_keyboard->GetState();
	m_kbTracker.Update(kbState);

	auto mouseState = Graphics::s_mouse->GetState();
	bool mouseMoved = false;

	if ((m_receiveState == ReceiveState::RADIANCE) && sendPacketDeltaTime > 0.1f)
	{
		DX12Lib::SceneCamera* camera = this->m_Scene->GetMainCamera();
		
		LI::LIScene* scene = dynamic_cast<LI::LIScene*>(m_Scene.get());

		DX12Lib::LightComponent* light = scene->GetMainLight();

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

		bool isCameraDirty = camera != nullptr && camera->IsDirty();
		bool isLightDirty = light != nullptr && light->Node->IsTransformDirty();

		if (camera != nullptr && ((m_lastCameraBitMask != cameraInputBitMask) || isCameraDirty))
		{
			PacketGuard packet = m_networkClient.CreatePacket();
			packet->ClearPacket();
			packet->AppendToBuffer("CAMINP");
			packet->AppendToBuffer(NetworkHost::GetEpochTime());
			packet->AppendToBuffer(camera->Node->GetPosition());
			packet->AppendToBuffer(camera->Node->GetRotationQuaternion());
			packet->AppendToBuffer(cameraInputBitMask);
			m_networkClient.SendData(packet);
		}

		if (light != nullptr && ((m_lastLightBitMask != lightInputBitMask) || isLightDirty))
		{
			PacketGuard packet = m_networkClient.CreatePacket();
			packet->ClearPacket();
			packet->AppendToBuffer("LGTINP");
			packet->AppendToBuffer(NetworkHost::GetEpochTime());
			packet->AppendToBuffer(light->Node->GetPosition());
			packet->AppendToBuffer(light->Node->GetRotationQuaternion());
			packet->AppendToBuffer(lightInputBitMask);
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

	this->m_Scene->Render(context);


	if (m_receiveState == ReceiveState::RADIANCE)
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

					m_lightTransportTechnique->ComputeVisibleFaces(computeContext);
					m_gaussianFilterTechnique->PerformTechnique(computeContext);
					m_gaussianFilterTechnique->PerformTechnique2(computeContext);

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
			}
		}

		if (m_radianceReady)
		{
			Renderer::ResetLerpTime();
			lerpDeltaTime = 0.0f;
		}

		if (didCameraMove)
		{
			Renderer::SetLerpMaxTime(0.1f);
			Renderer::SetDeltaLerpTime(lerpDeltaTime);
		}
		else
		{
			Renderer::SetLerpMaxTime(0.25f);
			Renderer::SetDeltaLerpTime(lerpDeltaTime);
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

		Renderer::RenderLayers(context);
	}

	if (m_radianceReady)
	{
		m_rasterFence->CurrentFenceValue = context.Flush();
		Graphics::s_commandQueueManager->GetGraphicsQueue().Signal(*m_rasterFence);
		m_rasterFenceValue = m_rasterFence->CurrentFenceValue;
	}
		
	LI::LIScene* scene = dynamic_cast<LI::LIScene*>(this->m_Scene.get());
		
	if (scene != nullptr)
		scene->StreamScene(context);

	lerpDeltaTime += GameTime::GetDeltaTime();

	m_radianceReady = false;

	Renderer::PostDrawCleanup(context);
}

void LocalIlluminationApp::OnClose(GraphicsContext& context)
{
	m_networkClient.Disconnect();
	DX12Lib::NetworkHost::DeinitializeEnet();
	D3DApp::OnClose(context);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		LI::LocalIlluminationApp app(hInstance, new LI::LIScene(STREAMING));
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