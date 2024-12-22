#define STREAMING 0

#include "LocalIllumination.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "LIUtils.h"
#include "Technique.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;
using namespace LI;

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

			std::vector<DirectX::XMUINT2> radBuffer(vecSize);

			memcpy(radBuffer.data(), packet->GetDataVector().data() + 7, vecSize * sizeof(DirectX::XMUINT2));

			m_bufferFence->WaitForCurrentFence();
			{
				void* mappedData = m_uploadBuffer.GetMappedData();

				auto& writeRdxBuffer = m_radianceRingBuffers[m_writeRadIx];

				memcpy(mappedData, packet->GetDataVector().data() + 7, vecSize * sizeof(DirectX::XMUINT2));

				DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();

				context.TransitionResource(writeRdxBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
				context.m_commandList->Get()->CopyResource(writeRdxBuffer.Get(), m_uploadBuffer.Get());

				UINT64 fenceVal = context.Finish();
					 
				m_bufferFence->CurrentFenceValue = fenceVal;
				Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);

				{
					std::lock_guard<std::mutex> lock(m_vectorMutex);
					m_fenceForBufferIdx.push(std::make_pair(m_writeRadIx, fenceVal));
				}

				m_writeRadIx = (m_writeRadIx + 1) % 3;
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

				m_basicBuffers[bf].Create(vecSize, sizeof(UINT32));

				// If the previous fence has not been signaled, then we have to wait before we can write data to the
				// Upload buffer, since the GPU might still be using it.
				m_bufferFence->WaitForCurrentFence();
				{
					void* mappedData = m_uploadBuffer.GetMappedData();

					memcpy(mappedData, packet->GetDataVector().data() + 7, vecSize * sizeof(UINT32));

					DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();

					// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
					context.TransitionResource(m_basicBuffers[bf], D3D12_RESOURCE_STATE_COPY_DEST, true);
					context.m_commandList->Get()->CopyBufferRegion(m_basicBuffers[bf].Get(), 0, m_uploadBuffer.Get(), 0, vecSize * sizeof(UINT32));

					UINT64 fenceVal = context.Finish();

					m_bufferFence->CurrentFenceValue = fenceVal;
					Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);

					DescriptorHandle& rtgiHandle = Renderer::GetRTGIHandleSRV();
					Graphics::s_device->Get()->CopyDescriptorsSimple(1, rtgiHandle + Renderer::s_textureHeap->GetDescriptorSize() * bf, m_basicBuffers[bf].GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

					// This is not a very good way to check if all buffers have been initialized, as if the server sends the same buffer multiple
					// times it will increment the counter. However, for the purposes of this demo it is enough.
					m_buffersInitialized++;
					if (m_buffersInitialized >= NUM_BASIC_BUFFERS)
					{
						m_receiveState = ReceiveState::RADIANCE;

						m_radianceRingBuffers[0].Create(m_faceCount, sizeof(DirectX::XMUINT2));
						m_radianceRingBuffers[1].Create(m_faceCount, sizeof(DirectX::XMUINT2));
						m_radianceRingBuffers[2].Create(m_faceCount, sizeof(DirectX::XMUINT2));

						Renderer::SetRTGIData(m_cbVoxelCommons);

						DXLIB_INFO("All buffers initialized");
						PacketGuard packet = m_networkClient.CreatePacket();
						packet->ClearPacket();
						packet->AppendToBuffer("BUFFER");
						m_networkClient.SendData(packet);
					}
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
			memcpy(&m_voxelCount, dataVector.data() + previousSize, sizeof(UINT));
			m_faceCount = m_voxelCount * 6;

			previousSize += sizeof(UINT);
			memcpy(&m_clusterCount, dataVector.data() + previousSize, sizeof(UINT));


			DXLIB_INFO("Received voxelization data with voxel count: {0} and cluster count: {1}", m_voxelCount, m_clusterCount);

			m_uploadBuffer.Create(m_faceCount * sizeof(DirectX::XMUINT2));
			m_radianceRingBuffers[0].Create(m_faceCount, sizeof(DirectX::XMUINT2));
			m_radianceRingBuffers[1].Create(m_faceCount, sizeof(DirectX::XMUINT2));
			m_radianceRingBuffers[2].Create(m_faceCount, sizeof(DirectX::XMUINT2));




			m_uploadBuffer.Map();

			DX12Lib::AABB sceneBounds = GetSceneAABBExtents();
			DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3((sceneBounds.Max.x - sceneBounds.Min.x) / voxelizationSize.x,
				(sceneBounds.Max.y - sceneBounds.Min.y) / voxelizationSize.y,
				(sceneBounds.Max.z - sceneBounds.Min.z) / voxelizationSize.z);

			if (m_data == nullptr)
			{
				m_data = std::make_shared<VOX::TechniqueData>();
			}

			m_data->SetSceneAABB(sceneBounds);
			m_data->SetVoxelGridSize(voxelizationSize);
			m_data->SetVoxelCellSize(voxelCellSize);
			m_data->SetClusterCount(m_clusterCount);
			m_data->SetVoxelCount(m_voxelCount);
			m_data->BuildMatrices();

			m_cbVoxelCommons = m_data->GetVoxelCommons();
			m_cbVoxelCommons.VoxelCount = m_voxelCount;
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

	m_bufferFence = std::make_unique<DX12Lib::Fence>(*Graphics::s_device, 0, 1);
	Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_bufferFence);

	m_networkClient.OnPacketReceived = std::bind(&LocalIlluminationApp::OnPacketReceived, this, std::placeholders::_1); 

	m_networkClient.Connect("127.0.0.1", 1234);

	s_mouse->SetMode(Mouse::MODE_RELATIVE);

	if (!m_usePBRMaterials)
	{
		auto rootNode = m_Scene->GetRootNode();

		rootNode->SetScale(0.01f, 0.01f, 0.01f);
	}

	this->m_Scene->Init(context);

	auto* camera = this->m_Scene->GetMainCamera();

	float aspect = camera->GetAspect();
	float fovY = camera->GetFovY();
	float nearZ = camera->GetNearZ();
	float farZ = camera->GetFarZ();

	fovY = 2.0f * atan(tan(fovY / 2.0f) * 1.3f);

	m_depthCamera.SetShadowBufferDimensions(1920, 1080);
	m_depthCamera.SetLens(camera->GetFovY(), aspect, nearZ, farZ);
	m_depthCamera.UpdateShadowMatrix(*camera->Node);
}

void LocalIlluminationApp::Update(GraphicsContext& context)
{
	D3DApp::Update(context);

	auto kbState = Graphics::s_keyboard->GetState();
	m_kbTracker.Update(kbState);

	auto mouseState = Graphics::s_mouse->GetState();
	bool mouseMoved = false;

	if (m_receiveState == ReceiveState::RADIANCE)
	{
		DX12Lib::SceneCamera* camera = this->m_Scene->GetMainCamera();

		std::uint8_t inputBitMask = 0;

		if (kbState.W)
			inputBitMask |= 1 << 0;
		if (kbState.S)
			inputBitMask |= 1 << 1;
		if (kbState.A)
			inputBitMask |= 1 << 2;
		if (kbState.D)
			inputBitMask |= 1 << 3;
		if (kbState.E)
			inputBitMask |= 1 << 4;
		if (kbState.Q)
			inputBitMask |= 1 << 5;



		if (camera != nullptr && ((m_lastInputBitMask != inputBitMask) || camera->IsDirty()))
		{
			PacketGuard packet = m_networkClient.CreatePacket();
			packet->ClearPacket();
			packet->AppendToBuffer("CAMINP");
			packet->AppendToBuffer(NetworkHost::GetEpochTime());
			packet->AppendToBuffer(camera->Node->GetPosition());
			packet->AppendToBuffer(camera->Node->GetRotationQuaternion());
			packet->AppendToBuffer(inputBitMask);
			m_networkClient.SendData(packet);
		}

		m_lastInputBitMask = inputBitMask;
	}
}

void LocalIlluminationApp::Draw(GraphicsContext& context)
{

	Renderer::SetUpRenderFrame(context);

	std::pair<UINT, UINT64> fencePair = std::pair<UINT, UINT64>(0, 0);

	{
		std::lock_guard<std::mutex> lock(m_vectorMutex);
		if (!m_fenceForBufferIdx.empty())
		{
			auto fPair = m_fenceForBufferIdx.front();

			if (m_bufferFence->IsFenceComplete(fencePair.second))
			{
				m_fenceForBufferIdx.pop();
				fencePair = fPair;
			}
		}
	}

	if (fencePair.second != 0)
	{
		Renderer::ResetLerpTime();
		DescriptorHandle& rtgiHandle = Renderer::GetRTGIHandleSRV();
		auto& writeRdxBuffer = m_radianceRingBuffers[fencePair.first];
		Graphics::s_device->Get()->CopyDescriptorsSimple(1, rtgiHandle + Renderer::s_textureHeap->GetDescriptorSize() * 5, writeRdxBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	this->m_Scene->Render(context);

	auto* camera = this->m_Scene->GetMainCamera();

	m_depthCamera.UpdateShadowMatrix(*camera->Node);

	//m_cameraCB.Position = m_data->GetCamera()->Node->GetPosition();
	//m_cameraCB.Direction = m_data->GetCamera()->Node->GetForward();
	//m_cameraCB.shadowTransform = m_depthCamera.GetShadowTransform();
	//m_cameraCB.invShadowTransform = m_depthCamera.GetInvShadowTransform();

	Renderer::ShadowPassForCamera(context, &m_depthCamera);
	Renderer::RenderLayers(context);
		
	LI::LIScene* scene = dynamic_cast<LI::LIScene*>(this->m_Scene.get());
		
	if (scene != nullptr)
		scene->StreamScene(context);

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