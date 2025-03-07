#include "DX12Lib/pch.h"

#include "D3DApp.h"
#include <WindowsX.h>
#include <iostream>
#include "DX12Lib/Commons/DX12Window.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "DX12Lib/Scene/Scene.h"
#include "UIHelpers.h"


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

		GraphicsContext& updateContext = GraphicsContext::Begin();
		Update(updateContext);
		updateContext.Finish();

		GraphicsContext& drawContext = GraphicsContext::Begin();
		Draw(drawContext);
		
		// Present the frame.
		auto& backBuffer = Renderer::GetCurrentBackBuffer();
		drawContext.TransitionResource(backBuffer, D3D12_RESOURCE_STATE_PRESENT, true);
		Renderer::Present(drawContext.Finish());
	}

	GraphicsContext& context = GraphicsContext::Begin();
	// Make sure render loop is done before closing.
	context.Flush(true);
	OnClose(context);
	context.Finish(true);
	Commons::UIHelpers::ShutdownIMGUI();

	
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

	GraphicsContext& context = GraphicsContext::Begin();
	this->Initialize(context);
	context.Finish(true);

	return true;
}

void DX12Lib::D3DApp::OnResize(GraphicsContext& commandContext, int newWidth, int newHeight)
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

void DX12Lib::D3DApp::Initialize(GraphicsContext& commandContext)
{
	m_Scene->OnAppStart(commandContext);
	commandContext.Flush(true);
	m_Scene->Init(commandContext);
}

void DX12Lib::D3DApp::Update(GraphicsContext& commandContext)
{
	Renderer::WaitForSwapchainBuffers();

	auto kbState = s_keyboard->GetState();
	s_kbTracker->Update(kbState);

	auto mouseState = s_mouse->GetState();
	s_mouseTracker->Update(mouseState);

	m_Scene->Update(commandContext);
}

void DX12Lib::D3DApp::Draw(GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	m_Scene->Render(commandContext);

	Renderer::RenderLayers(commandContext);

	Renderer::PostDrawCleanup(commandContext);
}


void DX12Lib::D3DApp::OnClose(GraphicsContext& context)
{
	m_Scene->OnClose(context);
}

void D3DApp::ResizeCallback(int newWidth, int newHeight)
{
	GraphicsContext& context = GraphicsContext::Begin();

	OnResize(context, newWidth, newHeight);

	context.Finish(true);
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
	Commons::UIHelpers::InitializeIMGUI(m_dx12Window->GetWindowHandle());


	return true;
}

void D3DApp::FlushCommandQueue()
{	
	s_commandQueueManager->GetGraphicsQueue().Flush();
}

void D3DApp::GetFrameStats(int& fps, float& mspf) const
{
	static int lastFps = 0.0f;
	static float lastMSPF = 0.0f;

	static int frameCount = 0;
	static float lastTime = 0.0f;

	frameCount++;

	float frameTime = GameTime::GetTotalTime();

	if (frameTime - lastTime >= 1.0)
	{
		lastFps = frameCount;
		lastMSPF = 1000.0f / (float)lastFps;

		frameCount = 0;

		lastTime = frameTime;
	}

	fps = lastFps;
	mspf = lastMSPF;
	
}
