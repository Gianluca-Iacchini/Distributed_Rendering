#include "D3DApp.h"
#include <WindowsX.h>
#include "CIDXGIFactory.h"
#include <iostream>
#include "Adapter.h"
#include "Device.h"
#include "Resource.h"

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
	if (m_d3dDevice != nullptr)
		mCommandQueue->Flush();
}

HINSTANCE D3DApp::AppInst() const
{
	return m_hAppInst;
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
		m_swapchain = std::make_unique<Swapchain>(*m_mainWindow, 3, mBackBufferFormat);
		m_swapchain->Finalize(*m_dxgiFactory, *m_d3dDevice, *mCommandQueue);
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
	if (!InitMainWindow())
		return false;

	if (!InitConsole())
		return false;

	if (!InitDirect3D())
		return false;

	// Do the initial resize code.
	OnResize();

	return true;
}

void D3DApp::CreateRtvAndDsvDescriptorHeaps()
{

	m_rtvHeap = std::make_unique<DescriptorHeap>(*m_d3dDevice, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, m_swapchain->BufferCount);
	m_dsvHeap = std::make_unique<DescriptorHeap>(*m_d3dDevice, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1);
}

void D3DApp::OnResize()
{
	assert(m_d3dDevice);
	assert(m_swapchain);

	// Flush before changing any resources.
	mCommandQueue->Flush();
	
	mCommandList->Reset(nullptr);

	if (m_depthStencilBuffer)
		m_depthStencilBuffer->ResetComPtr();

	m_swapchain->Resize(mClientWidth, mClientHeight);
	m_swapchain->CurrentBufferIndex = 0;

	for (unsigned int i = 0; i < m_swapchain->BufferCount; i++)
	{
		ResourceView rtView;
		rtView.descType = DescriptorType::RTV;
		rtView.view.RTV = nullptr;

		auto backBuffer = m_swapchain->GetBuffer(i);
		backBuffer->CreateView(rtView, *m_rtvHeap);
	}


	D3D12_RESOURCE_DESC depthStencilDesc;
	depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	depthStencilDesc.Alignment = 0;
	depthStencilDesc.Width = mClientWidth;
	depthStencilDesc.Height = mClientHeight;
	depthStencilDesc.DepthOrArraySize = 1;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	depthStencilDesc.SampleDesc.Count = 1;
	depthStencilDesc.SampleDesc.Quality = 0;
	depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
	
	D3D12_CLEAR_VALUE optClear;
	optClear.Format = mDepthStencilFormat;
	optClear.DepthStencil.Depth = 1.0f;
	optClear.DepthStencil.Stencil = 0;
	
	m_depthStencilBuffer = std::make_unique<Resource>(*m_d3dDevice, depthStencilDesc, &optClear);

	D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Format = mDepthStencilFormat;
	dsvDesc.Texture2D.MipSlice = 0;

	ResourceView dsvView;
	dsvView.descType = DescriptorType::DSV;
	dsvView.view.DSV = &dsvDesc;
	
	m_depthStencilBuffer->CreateView(dsvView, *m_dsvHeap);

	mCommandList->TransitionResource(m_depthStencilBuffer->Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);

	mCommandQueue->ExecuteCommandList(*mCommandList.get());



	mCommandQueue->Flush();

	mScreenViewport.TopLeftX = 0;
	mScreenViewport.TopLeftY = 0;
	mScreenViewport.Width = static_cast<float>(mClientWidth);
	mScreenViewport.Height = static_cast<float>(mClientHeight);
	mScreenViewport.MinDepth = 0.0f;
	mScreenViewport.MaxDepth = 1.0f;

	mScissorRect = {0, 0, mClientWidth, mClientHeight };
}

bool D3DApp::InitMainWindow()
{
	m_mainWindow = std::make_unique<DX12Window>(m_hAppInst, mClientWidth, mClientHeight, mMainWndCaption);
	
	return true;
}

bool D3DApp::InitDirect3D()
{
#if defined(DEBUG) || defined (_DEBUG)
	{
		Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
		ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
		debugController->EnableDebugLayer();
	}
#endif

	m_dxgiFactory = std::make_unique<CIDXGIFactory>();

	m_d3dDevice = std::make_unique<Device>();
	bool deviceCreated = m_d3dDevice->Initialize(nullptr);

	if (!deviceCreated)
	{
		Adapter warpAdapter = Adapter(*m_dxgiFactory, true);
		bool warpSucceded = m_d3dDevice->Initialize(&warpAdapter);

		if (!warpSucceded)
		{
			MessageBox(0, L"Direct3D Device Creation Failed.", 0, 0);
			return false;
		}
	}

#ifdef _DEBUG
	LogAdapters();
#endif // _DEBUG

	mCommandQueue = std::make_shared<CommandQueue>(*m_d3dDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);
	mCommandList = std::make_shared<CommandList>(*m_d3dDevice, 3, D3D12_COMMAND_LIST_TYPE_DIRECT, nullptr);

	mCommandList->GetComPtr()->Close();
	m_swapchain = std::make_unique<Swapchain>(*m_mainWindow, 3, mBackBufferFormat);
	m_swapchain->Finalize(*m_dxgiFactory, *m_d3dDevice, *mCommandQueue);

	CreateRtvAndDsvDescriptorHeaps();

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

		m_mainWindow->SetWindowTitle(windowText);
		
		frameCount = 0;

		timeElapsed += 1.0f;
	}
}

void D3DApp::LogAdapters()
{
	std::vector<Adapter> adapters = Adapter::GetAllAdapters(*m_dxgiFactory);

	for (auto a : adapters)
	{

		std::wstring text = L"***Adapter: ";
		text += a.GetDesc().Description;
		text += L"\n";

		std::wcout << text << std::endl;

		LogAdapterOutput(a);
	}
}

void D3DApp::LogAdapterOutput(Adapter m_adapter)
{
	UINT i = 0;
	ComPtr<IDXGIOutput> output = nullptr;
	while ((output = m_adapter.GetAdapterOutput(i)) != nullptr)
	{
		DXGI_OUTPUT_DESC desc;
		output->GetDesc(&desc);

		std::wstring text = L"***Output: ";
		text += desc.DeviceName;
		text += L"\n";
		
		std::wcout << text << std::endl;

		//LogOutputDisplayModes(output, mBackBufferFormat);
		i++;
	}
}

void D3DApp::LogOutputDisplayModes(ComPtr<IDXGIOutput> output, DXGI_FORMAT format)
{
	UINT count = 0;
	UINT flags = 0;

	output->GetDisplayModeList(format, flags, &count, nullptr);

	std::vector<DXGI_MODE_DESC> modeList(count);
	output->GetDisplayModeList(format, flags, &count, &modeList[0]);

	for (auto& x : modeList)
	{
		UINT n = x.RefreshRate.Numerator;
		UINT d = x.RefreshRate.Denominator;
		std::wstring text = L"Width = " + std::to_wstring(x.Width) + L" Height = " + std::to_wstring(x.Height) + L" Refresh = " + std::to_wstring(n) + L"/" + std::to_wstring(d) + L"\n";
		std::wcout << text << std::endl;
	}
}
