#include "DX12Lib/pch.h"

#include "D3DApp.h"
#include <WindowsX.h>
#include <iostream>
#include "DX12Lib/Commons/DX12Window.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "DX12Lib/Scene/Scene.h"


using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

D3DApp* D3DApp::m_App = nullptr;

D3DApp* D3DApp::GetApp()
{
	return m_App;
}

D3DApp::D3DApp(HINSTANCE hInstance, Scene* scene)
	: m_hAppInst(hInstance)
{
	// Only one D3DApp can be constructed.
	assert(m_App == nullptr);
	m_App = this;

	if (scene == nullptr)
		m_Scene = std::make_unique<Scene>();
	else
	{
		m_Scene = std::unique_ptr<Scene>(scene);
	}

}

D3DApp::~D3DApp()
{
	Graphics::Shutdown();

	FreeConsole();
}

HINSTANCE D3DApp::AppInst() const
{
	return m_hAppInst;
}

HWND D3DApp::MainWnd() const
{
	return m_dx12Window->GetWindowHandle();
}

float D3DApp::AspectRatio() const
{
	return static_cast<float>(Renderer::s_clientWidth) / Renderer::s_clientHeight;
}

bool D3DApp::Get4xMsaaSate() const
{
	return m_4xMsaaState;
}

void D3DApp::Set4xMsaaState(bool value)
{
	if (m_4xMsaaState != value)
	{
		m_4xMsaaState = value;

		// Recreate the swapchain and buffers with new multisample settings.
		Renderer::InitializeSwapchain(m_dx12Window.get());
		ResizeCallback(Renderer::s_clientWidth, Renderer::s_clientHeight);
	}
}

#pragma message ("TODO: Move render out of if-else")
int D3DApp::Run()
{
	MSG msg = { 0 };

	GameTime::s_Instance->Reset();

	bool isDone = false;

	while (!isDone)
	{
		// If there are Window messages then process them.
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);

			if (msg.message == WM_QUIT)
				isDone = true;
		}
		// Otherwise, do animation/game stuff.

		float startFrame = GameTime::GetTotalTime();

		GameTime::s_Instance->Tick();

		CalculateFrameStats();
		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
		Update(*context);
		context->Finish();

		context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
		Draw(*context);
		
		// Present the frame.
		auto& backBuffer = Renderer::GetCurrentBackBuffer();
		context->TransitionResource(backBuffer, D3D12_RESOURCE_STATE_PRESENT, true);
		Renderer::Present(context->Finish());
	}

	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
	OnClose(*context);
	context->Finish(true);

	return static_cast<int>(msg.wParam);
}

bool D3DApp::InitializeApp()
{
	if (!InitConsole())
		return false;

	if (!InitMainWindow())
		return false;

	if (!InitDirect3D())
		return false;

	GameTime::CreateInstance();

	// Do the initial resize code.
	ResizeCallback(Renderer::s_clientWidth, Renderer::s_clientHeight);

	s_commandQueueManager->GetGraphicsQueue().Flush();

	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
	this->Initialize(*context);
	context->Finish(true);

	return true;
}

void DX12Lib::D3DApp::OnResize(CommandContext& commandContext, int newWidth, int newHeight)
{
	// Flush before changing any resources.
	FlushCommandQueue();

	Renderer::ResizeSwapchain(&commandContext, newWidth, newHeight);

	if (m_Scene)
		m_Scene->OnResize(commandContext, newWidth, newHeight);
}

void DX12Lib::D3DApp::SetScene(Scene* scene)
{
	m_Scene = std::unique_ptr<Scene>(scene);
}

void DX12Lib::D3DApp::Initialize(CommandContext& commandContext)
{
	m_Scene->Init(commandContext);
}

void DX12Lib::D3DApp::Update(CommandContext& commandContext)
{
	Renderer::WaitForSwapchainBuffers();

	auto kbState = s_keyboard->GetState();
	s_kbTracker->Update(kbState);

	m_Scene->Update(commandContext);
}

void DX12Lib::D3DApp::Draw(CommandContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	m_Scene->Render(commandContext);

	Renderer::RenderLayers(commandContext);

}


void DX12Lib::D3DApp::OnClose(CommandContext& context)
{
	m_Scene->OnClose(context);
}

void D3DApp::ResizeCallback(int newWidth, int newHeight)
{
	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	OnResize(*context, newWidth, newHeight);

	context->Finish(true);
}



bool D3DApp::InitMainWindow()
{
	m_dx12Window = std::make_unique<DX12Window>(m_hAppInst, Renderer::s_clientWidth, Renderer::s_clientHeight, mMainWndCaption);

	if (!m_dx12Window->Create())
		return false;

	m_dx12Window->Show();

	Graphics::s_mouse = std::make_unique<DirectX::Mouse>();
	s_mouse->SetWindow(m_dx12Window->GetWindowHandle());

	return true;
}

bool D3DApp::InitConsole()
{
	if (AllocConsole()) {
		// Redirect unbuffered STDOUT to the console
		FILE* fp;
		freopen_s(&fp, "CONOUT$", "w", stdout);

		// Redirect unbuffered STDIN to the console
		freopen_s(&fp, "CONIN$", "r", stdin);

		// Redirect unbuffered STDERR to the console
		freopen_s(&fp, "CONOUT$", "w", stderr);

		// Use std::cout, std::cin, and std::cerr as usual
		std::cout.clear();
		std::cin.clear();
		std::cerr.clear();

		// Sync C++ and C standard streams
		std::ios::sync_with_stdio();

		// Initialize spdlog
		Logger::Init();

		return true;
	}

	return false;
}

bool D3DApp::InitDirect3D()
{
	if (!Graphics::InitializeApp())
		return false;

	HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	if (FAILED(hr))
	{
		MessageBox(0, L"Failed to initialize COM", 0, 0);
		return false;
	}

	Renderer::InitializeSwapchain(m_dx12Window.get());



	return true;
}

void D3DApp::FlushCommandQueue()
{	
	s_commandQueueManager->GetGraphicsQueue().Flush();
}

void D3DApp::CalculateFrameStats()
{
	static int frameCount = 0;
	static double lastTime = 0.0f;

	frameCount++;

	float frameTime = GameTime::GetTotalTime();

	if (frameTime - lastTime >= 1.0)
	{
		float fps = (float)frameCount;
		float mspf = 1000.0f / fps;

		std::wstring fpsStr = std::to_wstring(fps);
		std::wstring mspfStr = std::to_wstring(mspf);

		std::wstring windowText = mMainWndCaption + L"		fps: " + fpsStr + L"	mspf: " + mspfStr;

		SetWindowTextW(m_dx12Window->GetWindowHandle(), windowText.c_str());

		frameCount = 0;

		lastTime = frameTime;
	}
}

