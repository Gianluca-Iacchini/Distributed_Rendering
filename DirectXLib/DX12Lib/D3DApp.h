#pragma once
#include "Helpers.h"
#include "GameTime.h"
#include "CIDXGIFactory.h"
#include "Adapter.h"
#include "Device.h"
#include "CommandQueue.h"
#include "CommandAllocator.h"
#include "CommandList.h"
#include "DX12Window.h"
#include "Swapchain.h"
#include "DescriptorHeap.h"
#include "Resource.h"

#include <wrl.h>

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

	float AspectRatio() const;

	bool Get4xMsaaSate() const;
	void Set4xMsaaState(bool value);

	int Run();

	virtual bool Initialize();

protected:
	virtual void CreateRtvAndDsvDescriptorHeaps();
	virtual void OnResize();
	virtual void Update(const GameTime& gt) = 0;
	virtual void Draw(const GameTime& gt) = 0;

	// Handling mouse input
	virtual void OnMouseDown(WPARAM btnState, int x, int y) {}
	virtual void OnMouseUp(WPARAM btnState, int x, int y) {}
	virtual void OnMouseMove(WPARAM btnState, int x, int y) {}

protected:
	bool InitMainWindow();
	bool InitDirect3D();
	bool InitConsole();

	std::unique_ptr<DescriptorHeap> m_rtvHeap;
	std::unique_ptr<DescriptorHeap> m_dsvHeap;

	void CalculateFrameStats();

	void LogAdapters();
	void LogAdapterOutput(Adapter m_adapter);
	void LogOutputDisplayModes(ComPtr<IDXGIOutput> output, DXGI_FORMAT format);

protected:
	static D3DApp* m_App;

	std::unique_ptr<DX12Window> m_mainWindow;
	HINSTANCE m_hAppInst = nullptr;

	bool m_AppPaused = false;
	bool m_Minimized = false;
	bool m_Maximized = false;
	bool m_Resizing = false;
	bool m_FullscreenState = false;

	bool m_4xMsaaState = false;
	UINT m_4xMsaaQuality = 0;

	GameTime m_Time;

	std::unique_ptr<CIDXGIFactory> m_dxgiFactory;
	std::unique_ptr<Swapchain> m_swapchain;
	std::shared_ptr<Device> m_d3dDevice;

	ComPtr<ID3D12Fence> mFence;
	UINT64 mCurrentFence = 0;

	std::shared_ptr<CommandQueue> mCommandQueue;
	std::shared_ptr<CommandList> mCommandList;

	std::unique_ptr<Resource> m_depthStencilBuffer;

	D3D12_VIEWPORT mScreenViewport;
	D3D12_RECT mScissorRect;

	std::wstring mMainWndCaption = L"D3D12 Application";
	D3D_DRIVER_TYPE mD3DDriverType = D3D_DRIVER_TYPE_HARDWARE;
	DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT mDepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
	int mClientWidth = 1920;
	int mClientHeight = 1080;
};