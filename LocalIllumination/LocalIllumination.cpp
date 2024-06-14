#define STREAMING 0

#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"

#include "DX12Lib/Encoder/FFmpegStreamer.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

#define USE_PBR 1


class LocalIlluminationApp : public D3DApp
{
	float timeSinceRenderStart = 0;


#if STREAMING
	FFmpegStreamer ffmpegStreamer;
#endif

public:
	LocalIlluminationApp(HINSTANCE hInstance, Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
	LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
	LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
	~LocalIlluminationApp() { 

		FlushCommandQueue();
	};

	virtual void Initialize(CommandContext& context) override
	{

#if USE_PBR
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\PBR\\sponza2.gltf");
#else
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");
#endif

		bool loaded = this->m_Scene->AddFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");

		this->m_Scene->Init(context);


		s_mouse->SetMode(Mouse::MODE_RELATIVE);

#if STREAMING
		ffmpegStreamer.OpenStream(Renderer::s_clientWidth, Renderer::s_clientHeight);
		ffmpegStreamer.StartStreaming();
#endif
	}


	virtual void Update(CommandContext& context) override
	{
		D3DApp::Update(context);

#if STREAMING
		auto data = ffmpegStreamer.ConsumeData();
		this->m_Scene->SetNetworkData(std::get<char*>(data), std::get<size_t>(data));
#endif


	}

	virtual void Draw(CommandContext& context) override
	{



		Renderer::SetUpRenderFrame(context);


		this->m_Scene->Render(context);



		Renderer::RenderLayers(context);
		


#if STREAMING
		static float accumulatedTime = 0;
		static float lastUpdateTime = 0;
		static UINT encodedFPS = 0;

		
		// Accumulator is used to ensure proper frame rate for the encoder

		float totTime = GameTime::GetTotalTime();
		float encodeDeltaTime = totTime - lastUpdateTime;
		lastUpdateTime = totTime;
		accumulatedTime += encodeDeltaTime;

		float encoderFramerate = 1.f / ffmpegStreamer.GetEncoder().maxFrames;

		auto& backBuffer = Renderer::GetCurrentBackBuffer();

		if (accumulatedTime >= (encoderFramerate))
		{
			accumulatedTime -= encoderFramerate;
			ffmpegStreamer.Encode(context, backBuffer);
		}

		timeSinceRenderStart += GameTime::GetTotalTime();
#endif
	}

	virtual void OnClose(CommandContext& context) override
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
		LocalIlluminationApp app(hInstance, new LI::LIScene());
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