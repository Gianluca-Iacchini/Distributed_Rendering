#include "DX12Lib/Helpers.h"
#include "DX12Lib/D3DApp.h"
#include <iostream>
#include "DX12Lib/CIDXGIFactory.h"
#include "Keyboard.h"

using namespace DirectX;
using namespace Microsoft::WRL;

class AppTest : public D3DApp
{
	Keyboard keyboard;
	DirectX::Keyboard::KeyboardStateTracker tracker;

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 
		if (m_d3dDevice != nullptr)
			mCommandQueue->Flush();
		
		FreeConsole();
	};

	virtual bool Initialize() override
	{
		if (AllocConsole())
		{
			FILE* fp;

			freopen_s(&fp, "CONOUT$", "w", stdout);

			freopen_s(&fp, "CONIN$", "r", stdin);

			freopen_s(&fp, "CONOUT$", "w", stderr);

			std::cout.clear();
			std::clog.clear();
			std::cerr.clear();
			
			std::ios::sync_with_stdio();
		}

		std::cout << "Hello World!" << std::endl;

		return D3DApp::Initialize();
	}

private:

	virtual void OnResize() override
	{
		D3DApp::OnResize();
	}

	virtual void Update(const GameTime& gt) override
	{
		auto kbState = keyboard.GetState();
		tracker.Update(kbState);

		if (tracker.IsKeyPressed(DirectX::Keyboard::Keys::Escape))
		{
			PostQuitMessage(0);
		} 


	}

	virtual void Draw(const GameTime& gt) override
	{
		// Draw a square

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