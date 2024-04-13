#include "DX12Lib/Helpers.h"
#include "DX12Lib/D3DApp.h"
#include <iostream>
#include "DX12Lib/CIDXGIFactory.h"
#include "Keyboard.h"
#include "DX12Lib/Shader.h"
#include "Dx12Lib/PipelineState.h"

using namespace DirectX;
using namespace Microsoft::WRL;

struct Vertex
{
	Vertex(XMFLOAT3 p, XMFLOAT4 c) : Pos(p), Color(c) {}
	XMFLOAT3 Pos;
	XMFLOAT4 Color;
};

class AppTest : public D3DApp
{
	Vertex triangleVertices[3] =
	{
		{ XMFLOAT3(0.0f, 0.5f, 0.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) },
		{ XMFLOAT3(0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) },
		{ XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) }
	};

	std::unordered_map<std::string, std::shared_ptr<Shader>> mp_shaders;
	std::unique_ptr<PipelineState> m_pipelineState;

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
		if (!D3DApp::Initialize())
			return false;

		std::cout << "Hello World!" << std::endl;

		std::wstring srcDir = ToWstring(SOURCE_DIR);


		std::wstring shaderPath = srcDir + L"/Shaders/Basic.hlsl";

		std::wcout << shaderPath << std::endl;

		mp_shaders["BasicVS"] = std::make_shared<Shader>(shaderPath, "VS", "vs_5_1");
		mp_shaders["BasicVS"]->InputLayout =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		mp_shaders["BasicPS"] = std::make_shared<Shader>(shaderPath, "PS", "ps_5_1");

		mp_shaders["BasicVS"]->Compile();
		mp_shaders["BasicPS"]->Compile();

		m_pipelineState = std::make_unique<PipelineState>(mBackBufferFormat, mDepthStencilFormat);
		m_pipelineState->SetShader(mp_shaders["BasicVS"], ShaderType::Vertex);
		m_pipelineState->SetShader(mp_shaders["BasicPS"], ShaderType::Pixel);
		//m_pipelineState->Finalize(*m_d3dDevice);

		return true;
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
		int a = 0;
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