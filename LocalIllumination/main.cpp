#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "Keyboard.h"

#include "Dx12Lib/DXWrapper/PipelineState.h"
#include "FrameResource.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "GraphicsMemory.h"
#include "GeometricPrimitive.h"
#include "DX12Lib/Commons/Camera.h"
#include "Mouse.h"
#include "ResourceUploadBatch.h"
#include "DX12Lib/Scene/Scene.h"
#include "DX12Lib/DXWrapper/Swapchain.h"

#include "DX12Lib/Encoder/FFmpegStreamer.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

#define USE_PBR 1


class AppTest : public D3DApp
{
	UINT64 frameFences[3] = { 0, 0, 0 };


	CostantBufferCommons m_costantBufferCommons;
	ConstantBufferObject m_costantBufferObject;

	std::unique_ptr<DirectX::GeometricPrimitive> m_shape;

	std::unique_ptr<Scene> m_scene;

	float cameraSpeed = 100.0f;

	float m_theta = 1.25f * XM_PI;
	float m_phi = XM_PIDIV4;
	XMFLOAT2 m_modifier = XMFLOAT2(0.45f, 0.45f);

	std::shared_ptr<RootSignature> m_rootSignature;
	std::shared_ptr<PipelineState> m_pipelineState;

	UINT frameCount = 0;
	float timeSinceRenderStart = 0;


	std::vector<std::vector<std::uint8_t>> totPackets;

	FFmpegStreamer ffmpegStreamer;

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 

		FlushCommandQueue();
	};

	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);



#if USE_PBR
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\PBR\\sponza2.gltf");
#else
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");
#endif
		m_scene = std::make_unique<Scene>(this->m_Time);

		bool loaded = m_scene->AddFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");

		m_scene->Init(*context);


		context->Finish(true);

		s_mouse->SetMode(Mouse::MODE_RELATIVE);

#if STREAMING
		ffmpegStreamer.OpenStream(Renderer::s_clientWidth, Renderer::s_clientHeight);
		ffmpegStreamer.StartStreaming();
#endif

		return true;
	}

	void UpdateCommonConstants(const GameTime& gt)
	{

		m_costantBufferCommons.renderTargetSize = XMFLOAT2((float)Renderer::s_clientWidth, (float)Renderer::s_clientHeight);
		m_costantBufferCommons.invRenderTargetSize = XMFLOAT2(1.0f / Renderer::s_clientWidth, 1.0f / Renderer::s_clientHeight);
		m_costantBufferCommons.totalTime = gt.TotalTime();
		m_costantBufferCommons.deltaTime = gt.DeltaTime();
	}

	virtual void Update(const GameTime& gt) override
	{
		Renderer::WaitForSwapchainBuffers();

		// Update sun orientation
		m_theta +=  m_modifier.x * gt.DeltaTime();
		m_phi += m_modifier.y * gt.DeltaTime();


		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		auto kbState = s_keyboard->GetState();
		s_kbTracker->Update(kbState);

		UpdateCommonConstants(gt);

		auto data = ffmpegStreamer.ConsumeData();
		m_scene->SetNetworkData(std::get<char*>(data), std::get<size_t>(data));
		m_scene->Update(*context);

		context->Finish(true);
	}

	virtual void Draw(const GameTime& gt) override
	{

		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		auto commonRes = Renderer::s_graphicsMemory->AllocateConstant(m_costantBufferCommons);

		Renderer::SetUpRenderFrame(context);
		
		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::RootSignatureSlot::CommonCBV, commonRes.GpuAddress()
		);

		m_scene->Render(*context);

		Renderer::RenderLayers(context);
		
		auto& backBuffer = Renderer::GetCurrentBackBuffer();

#ifdef STREAMING
		static float accumulatedTime = 0;
		static float lastUpdateTime = 0;
		static UINT encodedFPS = 0;

		
		// Accumulator is used to ensure proper frame rate for the encoder

		float totTime = m_Time.TotalTime();
		float encodeDeltaTime = totTime - lastUpdateTime;
		lastUpdateTime = totTime;
		accumulatedTime += encodeDeltaTime;

		float encoderFramerate = 1.f / ffmpegStreamer.GetEncoder().maxFrames;

		if (accumulatedTime >= (encoderFramerate))
		{
			accumulatedTime -= encoderFramerate;
			ffmpegStreamer.Encode(*context, backBuffer);
		}

#endif
		context->TransitionResource(backBuffer, D3D12_RESOURCE_STATE_PRESENT, true);

		auto fenceVal = context->Finish(true);

		Renderer::Present(fenceVal);

		timeSinceRenderStart += m_Time.DeltaTime();
	}

	virtual void OnResize(CommandContext& context) override
	{
		D3DApp::OnResize(context);

		if (m_scene != nullptr)
			m_scene->OnResize(context);
	}

	virtual void OnClose() override
	{
#if STREAMING
		ffmpegStreamer.CloseStream();
#endif
	}
};


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		AppTest app(hInstance);
		if (!app.Initialize())
			return 0;

		return app.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}