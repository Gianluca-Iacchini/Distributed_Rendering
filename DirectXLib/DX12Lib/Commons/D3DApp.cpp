#include "DX12Lib/pch.h"

#include "D3DApp.h"
#include <WindowsX.h>
#include <iostream>
#include "DX12Lib/Commons/DX12Window.h"
#include "DX12Lib/DXWrapper/Swapchain.h"


using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

D3DApp* D3DApp::m_App = nullptr;

D3DApp* D3DApp::GetApp()
{
	return m_App;
}

D3DApp::D3DApp(HINSTANCE hInstance)
	: m_hAppInst(hInstance)
{
	// Only one D3DApp can be constructed.
	assert(m_App == nullptr);
	m_App = this;
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
	return static_cast<float>(mClientWidth) / mClientHeight;
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
		OnResize();
	}
}

#pragma message ("TODO: Move render out of if-else")
int D3DApp::Run()
{
	MSG msg = { 0 };

	m_Time.Reset();

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

		float startFrame = m_Time.TotalTime();

		m_Time.Tick();

		CalculateFrameStats(m_Time);
		Update(m_Time);
		Draw(m_Time);

		float endFrame = m_Time.TotalTime();

		float elapsedTime = endFrame - startFrame;

		float sleepTime = (1.f / 200.f) - elapsedTime;
	}

	return static_cast<int>(msg.wParam);
}

bool D3DApp::Initialize()
{
	if (!InitConsole())
		return false;

	if (!InitMainWindow())
		return false;

	if (!InitDirect3D())
		return false;

	// Do the initial resize code.
	OnResize();

	s_commandQueueManager->GetGraphicsQueue().Flush();

	return true;
}

void D3DApp::OnResize()
{

	// Flush before changing any resources.
	FlushCommandQueue();



	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	Renderer::s_swapchain->Resize(mClientWidth, mClientHeight);

	Renderer::s_swapchain->CurrentBufferIndex = 0;


	Renderer::s_depthStencilBuffer->GetComPtr().Reset();
	Renderer::s_depthStencilBuffer->Create(mClientWidth, mClientHeight, m_depthStencilFormat);

	context->TransitionResource(*Renderer::s_depthStencilBuffer, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);

	
	context->Finish(true);

	mScreenViewport.TopLeftX = 0;
	mScreenViewport.TopLeftY = 0;
	mScreenViewport.Width = static_cast<float>(mClientWidth);
	mScreenViewport.Height = static_cast<float>(mClientHeight);
	mScreenViewport.MinDepth = 0.0f;
	mScreenViewport.MaxDepth = 1.0f;

	mScissorRect = { 0, 0, mClientWidth, mClientHeight };
}



bool D3DApp::InitMainWindow()
{
	m_dx12Window = std::make_unique<DX12Window>(m_hAppInst, mClientWidth, mClientHeight, mMainWndCaption);

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
	if (!Graphics::Initialize())
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

void D3DApp::CalculateFrameStats(GameTime& gt)
{
	static int frameCount = 0;
	static double lastTime = 0.0f;

	frameCount++;

	float frameTime = gt.TotalTime();

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

