#pragma once

#include <Windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <string>
#include "GameTime.h"

namespace DX12Lib {

	class ColorBuffer;
	class DepthBuffer;
	class DX12Window;
	class Device;
	class Fence;
	class CommandAllocator;
	class CommandQueue;
	class CommandList;
	class Swapchain;

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
		void CreateSwapChain();
		void FlushCommandQueue();

		ColorBuffer& CurrentBackBuffer() const;
		D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView() const;
		D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView() const;

		void CalculateFrameStats(GameTime& gt);

	protected:
		void SetMsAsTitle(float ms);

		static D3DApp* m_App;

		HINSTANCE m_hAppInst = nullptr;
		bool m_AppPaused = false;
		bool m_Minimized = false;
		bool m_Maximized = false;
		bool m_Resizing = false;
		bool m_FullscreenState = false;

		bool m_4xMsaaState = false;
		UINT m_4xMsaaQuality = 0;

		GameTime m_Time;


		std::unique_ptr<DX12Window> m_dx12Window;

		std::unique_ptr<Swapchain> m_swapchain;
		std::shared_ptr<DepthBuffer> m_depthStencilBuffer;

		//std::unique_ptr<Fence> m_appFence;
		UINT64 mCurrentFence = 0;


		D3D12_VIEWPORT mScreenViewport;
		D3D12_RECT mScissorRect;

		std::wstring mMainWndCaption = L"D3D12 Application";
		D3D_DRIVER_TYPE mD3DDriverType = D3D_DRIVER_TYPE_HARDWARE;
		DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		DXGI_FORMAT mDepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
		int mClientWidth = 1920;
		int mClientHeight = 1080;

		std::chrono::high_resolution_clock::time_point m_initalTime;
	};
}