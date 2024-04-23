#include "D3DApp.h"
#include <WindowsX.h>
#include <iostream>
#include "CIDXGIFactory.h"
#include "Device.h"
#include "Adapter.h"
#include "Fence.h"
#include "CommandQueue.h"
#include "CommandList.h"
#include "CommandAllocator.h"
#include "DX12Lib/DescriptorHeap.h"
#include "DX12Lib/GraphicsCore.h"
#include "DX12Lib/DX12Window.h"
#include "Swapchain.h"
#include "ColorBuffer.h"
#include "DepthBuffer.h"
#include "CommandContext.h"

using namespace Microsoft::WRL;
using namespace Graphics;

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
	FlushCommandQueue();
	
	Graphics::Shutdown();

	if (context != nullptr)
		delete context;
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
		CreateSwapChain();
		OnResize();
	}
}

#pragma message ("TODO: Move render out of if-else")
int D3DApp::Run()
{
	MSG msg = { 0 };

	m_Time.Reset();

	while (msg.message != WM_QUIT)
	{
		// If there are Window messages then process them.
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// Otherwise, do animation/game stuff.
		else
		{
			m_Time.Tick();

			if (!m_AppPaused)
			{
				CalculateFrameStats();
				Update(m_Time);
				Draw(m_Time);
			}
			else
			{
				Sleep(100);
			}
		}
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

	context = new CommandContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
	context->Initialize();

	// Do the initial resize code.
	OnResize();

	return true;
}

void D3DApp::OnResize()
{
	assert(m_swapchain);

	// Flush before changing any resources.
	FlushCommandQueue();





	m_swapchain->Resize(mClientWidth, mClientHeight);

	m_swapchain->CurrentBufferIndex = 0;




	m_depthStencilBuffer->GetComPtr().Reset();
	m_depthStencilBuffer->Create(mClientWidth, mClientHeight, mDepthStencilFormat);



	//auto resourceBarrier = CD3DX12_RESOURCE_BARRIER::Transition(mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
	context->m_commandList->TransitionResource(m_depthStencilBuffer->Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);

	
	context->Finish();
	FlushCommandQueue();

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

		return true;
	}

	return false;
}

bool D3DApp::InitDirect3D()
{
	if (!Graphics::Initialize())
		return false;

	CreateSwapChain();

	m_depthStencilBuffer = std::make_shared<DepthBuffer>();

	return true;
}


void D3DApp::CreateSwapChain()
{
	m_swapchain = std::make_unique<Swapchain>(*m_dx12Window, mBackBufferFormat);
	m_swapchain->Initialize(s_commandQueueManager->GetGraphicsQueue());
}

void D3DApp::FlushCommandQueue()
{	
	s_commandQueueManager->GetGraphicsQueue().Flush();
}

ComPtr<ID3D12Resource> D3DApp::CurrentBackBuffer() const
{
	return m_swapchain->GetCurrentBackBuffer().Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE D3DApp::CurrentBackBufferView() const
{
	return m_swapchain->GetCurrentBackBuffer().GetRTV();
}

D3D12_CPU_DESCRIPTOR_HANDLE D3DApp::DepthStencilView() const
{
	return m_depthStencilBuffer->GetDSV();
}

void D3DApp::CalculateFrameStats()
{
	static int frameCount = 0;
	static float timeElapsed = 0.0f;

	frameCount++;

	if (m_Time.TotalTime() - timeElapsed >= 1.0f)
	{
		float fps = (float)frameCount;
		float mspf = 1000.0f / fps;

		std::wstring fpsStr = std::to_wstring(fps);
		std::wstring mspfStr = std::to_wstring(mspf);

		std::wstring windowText = mMainWndCaption + L"		fps: " + fpsStr + L"	mspf: " + mspfStr;

		SetWindowTextW(m_dx12Window->GetWindowHandle(), windowText.c_str());

		frameCount = 0;

		timeElapsed += 1.0f;
	}
}
