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

LRESULT CALLBACK
MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return D3DApp::GetApp()->MsgProc(hwnd, msg, wParam, lParam);
}

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
		FlushCommandQueue();
}

HINSTANCE D3DApp::AppInst() const
{
	return m_hAppInst;
}

HWND D3DApp::MainWnd() const
{
	return m_hMainWnd;
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

	// Do the initial resize code.
	OnResize();

	return true;
}

void D3DApp::OnResize()
{
	assert(m_d3dDevice);
	assert(mSwapChain);
	assert(m_appCommandAllocator);

	// Flush before changing any resources.
	FlushCommandQueue();

	//ThrowIfFailed(mCommandList->Reset(mCommandListAllocator.Get(), nullptr));

	m_commandList->Reset(*m_appCommandAllocator);

	for (int i = 0; i < SwapChainBufferCount; i++)
	{
		mSwapChainBuffer[i].Reset();
	}

	mDepthStencilBuffer.Reset();


	ThrowIfFailed(mSwapChain->ResizeBuffers(SwapChainBufferCount, mClientWidth, mClientHeight, mBackBufferFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));

	mCurrentBackBuffer = 0;


	//CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(mRtvHeap->GetCPUDescriptorHandleForHeapStart());

	for (unsigned int i = 0; i < SwapChainBufferCount; i++)
	{
		m_rtvHandles[i] = GraphicsCore::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_RTV); 
		ThrowIfFailed(mSwapChain->GetBuffer(i, IID_PPV_ARGS(&mSwapChainBuffer[i])));
		m_d3dDevice->CreateRenderTargetView(mSwapChainBuffer[i].Get(), nullptr, m_rtvHandles[i]);
	}

	D3D12_RESOURCE_DESC depthStencilDesc;
	depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	depthStencilDesc.Alignment = 0;
	depthStencilDesc.Width = mClientWidth;
	depthStencilDesc.Height = mClientHeight;
	depthStencilDesc.DepthOrArraySize = 1;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	depthStencilDesc.SampleDesc.Count = m_4xMsaaState ? 4 : 1;
	depthStencilDesc.SampleDesc.Quality = m_4xMsaaState ? (m_4xMsaaQuality - 1) : 0;
	depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

	D3D12_CLEAR_VALUE optClear;
	optClear.Format = mDepthStencilFormat;
	optClear.DepthStencil.Depth = 1.0f;
	optClear.DepthStencil.Stencil = 0;

	auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	ThrowIfFailed(m_d3dDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &depthStencilDesc, D3D12_RESOURCE_STATE_COMMON, &optClear, IID_PPV_ARGS(mDepthStencilBuffer.GetAddressOf())));

	m_dsvHandle = GraphicsCore::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

	D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Format = mDepthStencilFormat;
	dsvDesc.Texture2D.MipSlice = 0;
	m_d3dDevice->CreateDepthStencilView(mDepthStencilBuffer.Get(), &dsvDesc, m_dsvHandle);

	//auto resourceBarrier = CD3DX12_RESOURCE_BARRIER::Transition(mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
	m_commandList->TransitionResource(mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);


	m_commandList->Close();
	m_commandQueue->ExecuteCommandList(*m_commandList);

	FlushCommandQueue();

	mScreenViewport.TopLeftX = 0;
	mScreenViewport.TopLeftY = 0;
	mScreenViewport.Width = static_cast<float>(mClientWidth);
	mScreenViewport.Height = static_cast<float>(mClientHeight);
	mScreenViewport.MinDepth = 0.0f;
	mScreenViewport.MaxDepth = 1.0f;

	mScissorRect = { 0, 0, mClientWidth, mClientHeight };
}

LRESULT D3DApp::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
		// WM_ACTIVATE is sent when the window is activated or deactivated.  
		// We pause the game when the window is deactivated and unpause it 
		// when it becomes active.  
	case WM_ACTIVATE:
		if (LOWORD(wParam) == WA_INACTIVE)
		{
			m_AppPaused = true;
			m_Time.Stop();
		}
		else
		{
			m_AppPaused = false;
			m_Time.Start();
		}
		return 0;
	case WM_SIZE:
		mClientWidth = LOWORD(lParam);
		mClientHeight = HIWORD(lParam);
		if (m_d3dDevice)
		{
			if (wParam == SIZE_MINIMIZED)
			{
				m_AppPaused = true;
				m_Minimized = true;
				m_Maximized = false;
			}
			else if (wParam == SIZE_MAXIMIZED)
			{
				m_AppPaused = false;
				m_Minimized = false;
				m_Maximized = true;
				OnResize();
			}
			else if (wParam == SIZE_RESTORED)
			{
				if (m_Minimized)
				{
					m_AppPaused = false;
					m_Minimized = false;
					OnResize();
				}
				else if (m_Maximized)
				{
					m_AppPaused = false;
					m_Maximized = false;
					OnResize();
				}
				else if (m_Resizing)
				{
					// We do nothing when the user is dragging the window's frame. The resize will be done when the user stops resizing the window, which will send a
					// WM_EXITSIZEMOVE message.
				}
				else
				{
					OnResize();
				}
			}
		}
		return 0;
	case WM_ENTERSIZEMOVE:
		m_AppPaused = true;
		m_Resizing = true;
		m_Time.Stop();
		return 0;
	case WM_EXITSIZEMOVE:
		m_AppPaused = false;
		m_Resizing = false;
		m_Time.Start();
		OnResize();
		return 0;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_MENUCHAR:
		return MAKELRESULT(0, MNC_CLOSE);

		// Prevents the window from becoming too small.
	case WM_GETMINMAXINFO:
		((MINMAXINFO*)lParam)->ptMinTrackSize.x = 200;
		((MINMAXINFO*)lParam)->ptMinTrackSize.y = 200;
		return 0;

	case WM_LBUTTONDOWN:
	case WM_MBUTTONDOWN:
	case WM_RBUTTONDOWN:
		OnMouseDown(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_LBUTTONUP:
	case WM_MBUTTONUP:
	case WM_RBUTTONUP:
		OnMouseUp(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_MOUSEMOVE:
		OnMouseMove(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_KEYUP:
		if (wParam == VK_ESCAPE)
		{
			PostQuitMessage(0);
		}
		else if ((int)wParam == VK_F2)
		{
			Set4xMsaaState(!m_4xMsaaState);
		}
		return 0;

	}

	return DefWindowProc(hwnd, msg, wParam, lParam);
}

bool D3DApp::InitMainWindow()
{
	WNDCLASSEX wc = { 0 };
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = MainWndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = m_hAppInst;
	wc.hIcon = LoadIcon(0, IDI_APPLICATION);
	wc.hCursor = LoadCursor(0, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
	wc.lpszMenuName = 0;
	wc.lpszClassName = L"MainWindow";

	if (!RegisterClassEx(&wc))
	{
		MessageBox(0, L"RegisterClass Failed.", 0, 0);
		return false;
	}

	RECT R = { 0, 0, mClientWidth, mClientHeight };
	AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
	int width = R.right - R.left;
	int height = R.bottom - R.top;

	m_hMainWnd = CreateWindowEx(0, L"MainWindow", L"D3D12 Application", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, 0, 0, m_hAppInst, 0);

	if (!m_hMainWnd)
	{
		MessageBox(0, L"CreateWindow Failed.", 0, 0);
		return false;
	}

	ShowWindow(m_hMainWnd, SW_SHOW);
	UpdateWindow(m_hMainWnd);

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
#if defined(DEBUG) || defined (_DEBUG)
	{
		Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
		ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
		debugController->EnableDebugLayer();
	}
#endif
	
	m_idxgiFactory = std::make_unique<CIDXGIFactory>();

	m_device = std::make_unique<Device>();

	if (!m_device->Initialize(nullptr))
	{
		Adapter warpAdapter = Adapter(*m_idxgiFactory, true);

		if (!m_device->Initialize(&warpAdapter))
		{
			MessageBox(0, L"Direct3D initialization failed.", 0, 0);
			return false;
		}
	}

	GraphicsCore::Initialize(m_device.get());

	m_d3dDevice = m_device->GetComPtr();

	m_appFence = std::make_unique<Fence>(*m_device, 0);

	mRtvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	mDsvDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	mCbvSrvUavDescriptorSize = m_d3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);



#ifdef _DEBUG
	LogAdapters();
#endif // _DEBUG

	CreateCommandObjects();
	CreateSwapChain();

	return true;
}

void D3DApp::CreateCommandObjects()
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.NodeMask = 0;
	queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	m_commandQueue = std::make_unique<CommandQueue>(*m_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	m_appCommandAllocator = std::make_shared<CommandAllocator>(*m_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	m_commandList = std::make_shared<CommandList>(*m_device, *m_appCommandAllocator);

	//ThrowIfFailed(m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(mCommandQueue.GetAddressOf())));
	//ThrowIfFailed(m_d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandListAllocator)));
	//ThrowIfFailed(m_d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandListAllocator.Get(), nullptr, IID_PPV_ARGS(mCommandList.GetAddressOf())));

	m_commandList->Close();

	//mCommandList->Close();
}

void D3DApp::CreateSwapChain()
{
	mSwapChain.Reset();

	DXGI_SWAP_CHAIN_DESC swapChainDesc;
	swapChainDesc.BufferDesc.Width = mClientWidth;
	swapChainDesc.BufferDesc.Height = mClientHeight;
	swapChainDesc.BufferDesc.RefreshRate.Numerator = 144;
	swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	swapChainDesc.BufferDesc.Format = mBackBufferFormat;
	swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	swapChainDesc.SampleDesc.Count = m_4xMsaaState ? 4 : 1;
	swapChainDesc.SampleDesc.Quality = m_4xMsaaState ? (m_4xMsaaQuality - 1) : 0;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.BufferCount = SwapChainBufferCount;
	swapChainDesc.OutputWindow = m_hMainWnd;
	swapChainDesc.Windowed = true;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;



	ThrowIfFailed(m_idxgiFactory->GetComPtr()->CreateSwapChain(m_commandQueue->Get(), &swapChainDesc, mSwapChain.GetAddressOf()));
}

void D3DApp::FlushCommandQueue()
{	
	m_commandQueue->Flush(m_appFence.get());
}

ComPtr<ID3D12Resource> D3DApp::CurrentBackBuffer() const
{
	return mSwapChainBuffer[mCurrentBackBuffer];
}

D3D12_CPU_DESCRIPTOR_HANDLE D3DApp::CurrentBackBufferView() const
{
	return m_rtvHandles[mCurrentBackBuffer];
}

D3D12_CPU_DESCRIPTOR_HANDLE D3DApp::DepthStencilView() const
{
	return m_dsvHandle;
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

		SetWindowTextW(m_hMainWnd, windowText.c_str());

		frameCount = 0;

		timeElapsed += 1.0f;
	}
}

void D3DApp::LogAdapters()
{
	UINT i = 0;
	ComPtr<IDXGIAdapter> adapter = nullptr;
	std::vector<ComPtr<IDXGIAdapter>> adapterList;

	while (m_idxgiFactory->GetComPtr()->EnumAdapters(i, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_ADAPTER_DESC desc;
		adapter->GetDesc(&desc);

		std::wstring text = L"***Adapter: ";
		text += desc.Description;
		text += L"\n";

		OutputDebugStringW(text.c_str());

		adapterList.push_back(adapter);
		i++;
	}

	for (size_t i = 0; i < adapterList.size(); i++)
	{
		LogAdapterOutput(adapterList[i]);
	}
}

void D3DApp::LogAdapterOutput(ComPtr<IDXGIAdapter> adapter)
{
	UINT i = 0;
	ComPtr<IDXGIOutput> output = nullptr;
	while (adapter->EnumOutputs(i, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_OUTPUT_DESC desc;
		output->GetDesc(&desc);

		std::wstring text = L"***Output: ";
		text += desc.DeviceName;
		text += L"\n";
		OutputDebugStringW(text.c_str());

		LogOutputDisplayModes(output, mBackBufferFormat);
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
		OutputDebugStringW(text.c_str());
	}
}
