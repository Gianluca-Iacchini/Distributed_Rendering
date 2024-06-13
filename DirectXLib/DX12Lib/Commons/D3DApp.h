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
	class Scene;

	using Microsoft::WRL::ComPtr;
	class D3DApp
	{
	protected:
		D3DApp(HINSTANCE hInstance, Scene* scene);
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

		virtual void OnResize(CommandContext& commandContext, int newWidth, int newHeigth);

		void SetScene(Scene* scene);

	protected:

		virtual void Update(CommandContext& commandContext);
		virtual void Draw(CommandContext& commandContext);
		virtual void OnClose(CommandContext& commandContext);

		// Handling mouse input
		virtual void OnMouseDown(WPARAM btnState, int x, int y) {}
		virtual void OnMouseUp(WPARAM btnState, int x, int y) {}
		virtual void OnMouseMove(WPARAM btnState, int x, int y) {}

	protected:
		bool InitMainWindow();
		bool InitConsole();
		bool InitDirect3D();
		void FlushCommandQueue();
		void CalculateFrameStats();

	private:
		void ResizeCallback(int newWidth, int newHeight);

	protected:

		static D3DApp* m_App;

		HINSTANCE m_hAppInst = nullptr;
		bool m_AppPaused = false;
		bool m_Minimized = false;
		bool m_Maximized = false;
		bool m_Resizing = false;
		bool m_FullscreenState = false;

		bool m_4xMsaaState = false;
		UINT m_4xMsaaQuality = 0;

		std::unique_ptr<Scene> m_Scene;

		std::unique_ptr<DX12Window> m_dx12Window;

		UINT64 mCurrentFence = 0;

		std::wstring mMainWndCaption = L"D3D12 Application";
		D3D_DRIVER_TYPE mD3DDriverType = D3D_DRIVER_TYPE_HARDWARE;
	};
}