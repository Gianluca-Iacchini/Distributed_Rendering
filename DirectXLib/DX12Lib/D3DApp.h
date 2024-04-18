#pragma once
#include "Helpers.h"
#include "GameTime.h"

#include <wrl.h>

class CIDXGIFactory;
class Device;
class Fence;
class CommandQueue;
class CommandAllocator;
class CommandList;
class DescriptorAllocator;

using Microsoft::WRL::ComPtr;
class D3DApp
{
protected:
	D3DApp(HINSTANCE hInstance);
	D3DApp(const D3DApp& rhs) = delete;
	D3DApp& operator=(const D3DApp& rhs) = delete;
	virtual ~D3DApp();

public:

	static D3DApp* GetApp();

	HINSTANCE AppInst() const;
	HWND MainWnd() const;
	float AspectRatio() const;

	bool Get4xMsaaSate() const;
	void Set4xMsaaState(bool value);

	int Run();

	virtual bool Initialize();
	virtual LRESULT MsgProc(HWND hwn, UINT msg, WPARAM wParam, LPARAM lParam);

protected:
	virtual void OnResize();
	virtual void Update(const GameTime& gt) = 0;
	virtual void Draw(const GameTime& gt) = 0;

	// Handling mouse input
	virtual void OnMouseDown(WPARAM btnState, int x, int y) {}
	virtual void OnMouseUp(WPARAM btnState, int x, int y) {}
	virtual void OnMouseMove(WPARAM btnState, int x, int y) {}

protected:
	bool InitMainWindow();
	bool InitConsole();
	bool InitDirect3D();
	void CreateCommandObjects();
	void CreateSwapChain();
	void FlushCommandQueue();

	ComPtr<ID3D12Resource> CurrentBackBuffer() const;
	D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView() const;
	D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView() const;

	void CalculateFrameStats();

	void LogAdapters();
	void LogAdapterOutput(ComPtr<IDXGIAdapter> adapter);
	void LogOutputDisplayModes(ComPtr<IDXGIOutput> output, DXGI_FORMAT format);

protected:
	static D3DApp* m_App;

	HINSTANCE m_hAppInst = nullptr;
	HWND m_hMainWnd = nullptr;
	bool m_AppPaused = false;
	bool m_Minimized = false;
	bool m_Maximized = false;
	bool m_Resizing = false;
	bool m_FullscreenState = false;

	bool m_4xMsaaState = false;
	UINT m_4xMsaaQuality = 0;

	GameTime m_Time;

	std::unique_ptr<CIDXGIFactory> m_idxgiFactory;
	ComPtr<IDXGISwapChain> mSwapChain;
	std::unique_ptr<Device> m_device;

	Microsoft::WRL::ComPtr<ID3D12Device> m_d3dDevice;

	std::unique_ptr<Fence> m_appFence;
	UINT64 mCurrentFence = 0;

	std::unique_ptr<CommandQueue> m_commandQueue;
	std::shared_ptr<CommandAllocator> m_appCommandAllocator;
	std::shared_ptr<CommandList> m_commandList;

	static const int SwapChainBufferCount = 3;
	int mCurrentBackBuffer = 0;

	ComPtr<ID3D12Resource> mSwapChainBuffer[SwapChainBufferCount];
	ComPtr<ID3D12Resource> mDepthStencilBuffer;

	D3D12_VIEWPORT mScreenViewport;
	D3D12_RECT mScissorRect;

	UINT mRtvDescriptorSize = 0;
	UINT mDsvDescriptorSize = 0;
	UINT mCbvSrvUavDescriptorSize = 0;

	std::wstring mMainWndCaption = L"D3D12 Application";
	D3D_DRIVER_TYPE mD3DDriverType = D3D_DRIVER_TYPE_HARDWARE;
	DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT mDepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	int mClientWidth = 1920;
	int mClientHeight = 1080;

	D3D12_CPU_DESCRIPTOR_HANDLE m_rtvHandles[SwapChainBufferCount];
	D3D12_CPU_DESCRIPTOR_HANDLE m_dsvHandle;
};