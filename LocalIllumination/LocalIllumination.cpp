#define STREAMING 0

#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"
#include "DX12Lib/Models/ModelRenderer.h"
#include "DX12Lib/Commons/NetworkManager.h"
#include "LIUtils.h"


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
	ConstantBufferVoxelCommons m_cbVoxelCommons;
	bool m_receivedVoxelizationData = false;
public:
	LocalIlluminationApp(HINSTANCE hInstance, Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
	LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
	LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
	~LocalIlluminationApp() { 

		FlushCommandQueue();
	};

	void OnPacketReceived(const NetworkPacket* packet)
	{
		if (m_receivedVoxelizationData)
		{
			if (NetworkHost::CheckPacketHeader(packet, "VOXOCC"))
			{
				auto& dataVector = packet->GetDataVector();
				std::vector<UINT32> voxelizationSize;
				voxelizationSize.resize((packet->GetSize() - 7) / sizeof(UINT32));

				memcpy(voxelizationSize.data(), dataVector.data() + 7, (packet->GetSize() - 7));

				DXLIB_INFO("Received voxelization occupancy data, buffer size: {0}, first five elements are [{1},{2},{3},{4},{5}]", voxelizationSize.size(), voxelizationSize[0], voxelizationSize[1], voxelizationSize[2], voxelizationSize[3], voxelizationSize[4]);
			}
		}
		else
		{
			// To ensure that the server sent the initialization message, the message starts with "VOX" (4 bytes due to null character)
			// Then each float is 4 bytes long.

			if (NetworkHost::CheckPacketHeader(packet, "VOX"))
			{
				auto& dataVector = packet->GetDataVector();
				DirectX::XMUINT3 voxelizationSize;

				memcpy(&voxelizationSize, dataVector.data() + 4, sizeof(DirectX::XMUINT3));

				DXLIB_INFO("Received voxelization data with size: [{0},{1},{2}]", voxelizationSize.x, voxelizationSize.y, voxelizationSize.z);

				LI::LIUtils::BuildVoxelCommons(GetSceneAABBExtents(), voxelizationSize);
				m_receivedVoxelizationData = true;


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