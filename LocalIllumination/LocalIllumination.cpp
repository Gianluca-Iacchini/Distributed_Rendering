#define STREAMING 0

#define NUM_BASIC_BUFFERS 5

#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"
#include "DX12Lib/Models/ModelRenderer.h"
#include "DX12Lib/Commons/NetworkManager.h"
#include "LIUtils.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;



class LocalIlluminationApp : public D3DApp
{
private:
	bool m_usePBRMaterials = true;
	DX12Lib::NetworkHost m_networkClient;
	DirectX::Keyboard::KeyboardStateTracker m_kbTracker;

	UINT m_buffersInitialized = 0;
	UINT m_voxelCount = 0;
	UINT m_faceCount = 0;
	UINT m_clusterCount = 0;
	
	DX12Lib::UploadBuffer m_uploadBuffer;
	DX12Lib::StructuredBuffer m_basicBuffers[5];
	DX12Lib::StructuredBuffer m_radianceRingBuffers[3];

	std::unique_ptr<DX12Lib::Fence> m_basicBufferFence;

	const char* packetHeaders[NUM_BASIC_BUFFERS] = { "OCCVOX", "INDRNK", "INDIDX", "CMPIDX", "CMPHSH"};

	enum class ReceiveState
	{
		INITIALIZATION,
		BASIC_BUFFERS,
		RADIANCE,
	} m_receiveState = ReceiveState::INITIALIZATION;

public:
	LocalIlluminationApp(HINSTANCE hInstance, Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
	LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
	LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
	~LocalIlluminationApp() { 

		FlushCommandQueue();
	};

	void OnPacketReceived(const NetworkPacket* packet)
	{
		if (m_receiveState == ReceiveState::RADIANCE)
		{

		}
		else if (m_receiveState == ReceiveState::BASIC_BUFFERS)
		{
			for (int i = 0; i < 5; i++)
			{
				if (NetworkHost::CheckPacketHeader(packet, packetHeaders[i]))
				{
					std::size_t pktSize = packet->GetSize() - 7;
					std::size_t vecSize = pktSize / sizeof(UINT32);
					DXLIB_INFO("Received packet with header: {0} and vector size: {1}", packetHeaders[i], vecSize);

					m_basicBuffers[i].Create(vecSize, sizeof(UINT32));

					// If the previous fence has not been signaled, then we have to wait before we can write data to the
					// Upload buffer, since the GPU might still be using it.
					m_basicBufferFence->WaitForCurrentFence();
					{
						void* mappedData = m_uploadBuffer.GetMappedData();

						// The vertex buffer is just an array of indices frin 0 to faceCount-1.
						// Not great but fine for a debug display.
						for (UINT32 i = 0; i < vecSize; i++)
						{
							// 7 = Size of the packet header
							((UINT32*)mappedData)[i] = i + 7;
						}

						DX12Lib::ComputeContext& context = DX12Lib::ComputeContext::Begin();

						// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
						context.TransitionResource(m_basicBuffers[i], D3D12_RESOURCE_STATE_COPY_DEST, true);
						context.m_commandList->Get()->CopyResource(m_basicBuffers[i].Get(), m_uploadBuffer.Get());

						UINT64 fenceVal = context.Finish();

						m_basicBufferFence->CurrentFenceValue = fenceVal;
						Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_basicBufferFence);

						// This is not a very good way to check if all buffers have been initialized, as if the server sends the same buffer multiple
						// times it will increment the counter. However, for the purposes of this demo it is enough.
						m_buffersInitialized++;
						if (m_buffersInitialized >= NUM_BASIC_BUFFERS)
						{
							DXLIB_INFO("All buffers initialized");
							PacketGuard packet = m_networkClient.CreatePacket();
							packet->ClearPacket();
							packet->AppendToBuffer("BUFFER");
							m_networkClient.SendData(packet);

							m_receiveState = ReceiveState::RADIANCE;
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

				m_uploadBuffer.Create(m_faceCount * sizeof(UINT32));
				m_radianceRingBuffers[0].Create(m_faceCount, sizeof(UINT32));
				m_radianceRingBuffers[1].Create(m_faceCount, sizeof(UINT32));
				m_radianceRingBuffers[2].Create(m_faceCount, sizeof(UINT32));


				m_uploadBuffer.Map();

				LI::LIUtils::BuildVoxelCommons(GetSceneAABBExtents(), voxelizationSize);
				m_receiveState = ReceiveState::BASIC_BUFFERS;


				PacketGuard packet = m_networkClient.CreatePacket();
				packet->ClearPacket();
				packet->AppendToBuffer("INIT");
				m_networkClient.SendData(packet);
			}
		}
	}

	DX12Lib::AABB GetSceneAABBExtents()
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
		float maxComponent = std::max(sceneBounds.Max.x, std::max(sceneBounds.Max.y, sceneBounds.Max.z));

		float extent = maxComponent - minComponent;

		sceneBounds.Min.x = minComponent;
		sceneBounds.Min.y = minComponent;
		sceneBounds.Min.z = minComponent;

		sceneBounds.Max.x = maxComponent;
		sceneBounds.Max.y = maxComponent;
		sceneBounds.Max.z = maxComponent;

		return sceneBounds;
	}



	virtual void Initialize(GraphicsContext& context) override
	{

		std::string sourcePath = std::string(SOURCE_DIR);

		if (m_usePBRMaterials)
			sourcePath += std::string("\\Models\\PBR\\sponza2.gltf");
		else
			sourcePath += std::string("\\Models\\sponza_nobanner.obj");


		bool loaded = this->m_Scene->AddFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");

		DX12Lib::NetworkHost::InitializeEnet();

		m_basicBufferFence = std::make_unique<DX12Lib::Fence>(*Graphics::s_device, 0, 1);
		Graphics::s_commandQueueManager->GetComputeQueue().Signal(*m_basicBufferFence);

		m_networkClient.OnPacketReceived = std::bind(&LocalIlluminationApp::OnPacketReceived, this, std::placeholders::_1); 

		m_networkClient.Connect("127.0.0.1", 1234);

		s_mouse->SetMode(Mouse::MODE_RELATIVE);

		if (!m_usePBRMaterials)
		{
			auto rootNode = m_Scene->GetRootNode();

			rootNode->SetScale(0.01f, 0.01f, 0.01f);
		}

		this->m_Scene->Init(context);


	}

	void sendData()
	{
		DXLIB_CORE_INFO("Sending packets");
		PacketGuard packet = m_networkClient.CreatePacket();
		packet->ClearPacket();
		std::uint8_t message[5] = { 0, 1, 2, 3, 4 };
		packet->AppendToBuffer(message, 5);
		m_networkClient.SendData(packet);
	}

	virtual void Update(GraphicsContext& context) override
	{
		D3DApp::Update(context);

		auto kbState = Graphics::s_keyboard->GetState();
		m_kbTracker.Update(kbState);

		if (m_kbTracker.pressed.B)
		{
			sendData();
		}

	}

	virtual void Draw(GraphicsContext& context) override
	{

		Renderer::SetUpRenderFrame(context);


		this->m_Scene->Render(context);



		Renderer::RenderLayers(context);
		
		LI::LIScene* scene = dynamic_cast<LI::LIScene*>(this->m_Scene.get());
		
		if (scene != nullptr)
			scene->StreamScene(context);

		Renderer::PostDrawCleanup(context);
	}

	virtual void OnClose(GraphicsContext& context) override
	{
		m_networkClient.Disconnect();
		DX12Lib::NetworkHost::DeinitializeEnet();
		D3DApp::OnClose(context);
	}
};


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		LocalIlluminationApp app(hInstance, new LI::LIScene(STREAMING));
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