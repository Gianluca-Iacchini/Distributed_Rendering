#pragma once

#include <Windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <string>
#include "GameTime.h"
#include "DX12Lib/Commons/CommandContext.h"


namespace DX12Lib {
	class DX12Window;
	class Scene;



	using Microsoft::WRL::ComPtr;
	class D3DApp
	{
	public:
		static D3DApp* GetApp();
		D3DApp(HINSTANCE hInstance, Scene* scene);
		D3DApp(const D3DApp& rhs) = delete;
		D3DApp& operator=(const D3DApp& rhs) = delete;
		virtual ~D3DApp();

		HINSTANCE AppInst() const;
		HWND MainWnd() const;
		float AspectRatio() const;

		bool Get4xMsaaSate() const;
		void Set4xMsaaState(bool value);

		int Run();

		bool InitializeApp();

		virtual void OnResize(GraphicsContext& commandContext, int newWidth, int newHeigth);

		void SetScene(Scene* scene);

	protected:

		virtual void Initialize(GraphicsContext& commandContext);
		virtual void Update(GraphicsContext& commandContext);
		virtual void Draw(GraphicsContext& commandContext);
		virtual void OnClose(GraphicsContext& commandContext);

		// Handling mouse input
		virtual void OnMouseDown(WPARAM btnState, int x, int y) {}
		virtual void OnMouseUp(WPARAM btnState, int x, int y) {}
		virtual void OnMouseMove(WPARAM btnState, int x, int y) {}

		void GetFrameStats(int& fps, float& mspf) const;

	protected:
		bool InitMainWindow();
		bool InitConsole();
		bool InitDirect3D();
		void FlushCommandQueue();

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


		std::wstring mMainWndCaption = L"D3D12 Application";
		D3D_DRIVER_TYPE mD3DDriverType = D3D_DRIVER_TYPE_HARDWARE;
	};
}