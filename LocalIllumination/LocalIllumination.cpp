#define STREAMING 1

#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

#define USE_PBR 1


class LocalIlluminationApp : public D3DApp
{

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
	}


	virtual void Update(CommandContext& context) override
	{
		D3DApp::Update(context);

	}

	virtual void Draw(CommandContext& context) override
	{



		Renderer::SetUpRenderFrame(context);


		this->m_Scene->Render(context);



		Renderer::RenderLayers(context);
		
	}

	virtual void OnClose(CommandContext& context) override
	{
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