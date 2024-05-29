#include "DX12Lib/pch.h"

#include "DX12Window.h"
#include "Mouse.h"
#include "Keyboard.h"
#include "D3DApp.h"

using namespace DX12Lib;

LRESULT CALLBACK
MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
		// WM_ACTIVATE is sent when the window is activated or deactivated.  
		// We pause the game when the window is deactivated and unpause it 
		// when it becomes active.  
	case WM_ACTIVATE:
		//if (LOWORD(wParam) == WA_INACTIVE)
		//{
		//	m_AppPaused = true;
		//	m_Time.Stop();
		//}
		//else
		//{
		//	m_AppPaused = false;
		//	m_Time.Start();
		//}
	case WM_ACTIVATEAPP:
		DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
		DirectX::Mouse::ProcessMessage(msg, wParam, lParam);
		break;
	case WM_SIZE:
		//mClientWidth = LOWORD(lParam);
		//mClientHeight = HIWORD(lParam);
		//if (m_d3dDevice)
		//{
		//	if (wParam == SIZE_MINIMIZED)
		//	{
		//		m_AppPaused = true;
		//		m_Minimized = true;
		//		m_Maximized = false;
		//	}
		//	else if (wParam == SIZE_MAXIMIZED)
		//	{
		//		m_AppPaused = false;
		//		m_Minimized = false;
		//		m_Maximized = true;
		//		OnResize();
		//	}
		//	else if (wParam == SIZE_RESTORED)
		//	{
		//		if (m_Minimized)
		//		{
		//			m_AppPaused = false;
		//			m_Minimized = false;
		//			OnResize();
		//		}
		//		else if (m_Maximized)
		//		{
		//			m_AppPaused = false;
		//			m_Maximized = false;
		//			OnResize();
		//		}
		//		else if (m_Resizing)
		//		{
		//			// We do nothing when the user is dragging the window's frame. The resize will be done when the user stops resizing the window, which will send a
		//			// WM_EXITSIZEMOVE message.
		//		}
		//		else
		//		{
		//			OnResize();
		//		}
		//	}
		//}
		return 0;
	case WM_ENTERSIZEMOVE:
		//m_AppPaused = true;
		//m_Resizing = true;
		//m_Time.Stop();
		return 0;
	case WM_EXITSIZEMOVE:
		//m_AppPaused = false;
		//m_Resizing = false;
		//m_Time.Start();
		//OnResize();
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

	case WM_KEYDOWN:
	case WM_SYSKEYDOWN:
	case WM_KEYUP:
	case WM_SYSKEYUP:
		DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
		break;
		//if (wParam == VK_ESCAPE)
		//{
		//	PostQuitMessage(0);
		//}
		//else if ((int)wParam == VK_F2)
		//{
		//	//Set4xMsaaState(!m_4xMsaaState);
		//}
		//return 0;

	case WM_INPUT:
	case WM_MOUSEMOVE:
	case WM_LBUTTONDOWN:
	case WM_LBUTTONUP:
	case WM_RBUTTONDOWN:
	case WM_RBUTTONUP:
	case WM_MBUTTONDOWN:
	case WM_MBUTTONUP:
	case WM_MOUSEWHEEL:
	case WM_XBUTTONDOWN:
	case WM_XBUTTONUP:
	case WM_MOUSEHOVER:
		DirectX::Mouse::ProcessMessage(msg, wParam, lParam);
		break;
	}


	return DefWindowProc(hwnd, msg, wParam, lParam);
}

DX12Window::DX12Window(HINSTANCE hInstance, int width, int height, std::wstring windowTitle) : 
	m_hInstance(hInstance), m_width(width), m_height(height), m_windowTitle(windowTitle)
{
	m_windowClass = { 0 };
	m_windowClass.cbSize = sizeof(WNDCLASSEX);
	m_windowClass.style = CS_HREDRAW | CS_VREDRAW;
	m_windowClass.lpfnWndProc = MainWndProc;
	m_windowClass.cbClsExtra = 0;
	m_windowClass.cbWndExtra = 0;
	m_windowClass.hInstance = hInstance;
	m_windowClass.hIcon = LoadIcon(0, IDI_APPLICATION);
	m_windowClass.hCursor = LoadCursor(0, IDC_ARROW);
	m_windowClass.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
	m_windowClass.lpszMenuName = 0;
	m_windowClass.lpszClassName = L"MainWindow";


}

bool DX12Window::Create()
{
	if (!RegisterClassEx(&m_windowClass))
	{
		MessageBox(0, L"RegisterClass Failed.", 0, 0);
		return false;
	}

	RECT R = { 0, 0, m_width, m_height};
	AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
	int rWidth = R.right - R.left;
	int rHeight = R.bottom - R.top;

	m_hwnd = CreateWindowEx(0, L"MainWindow", L"D3D12 Application", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rWidth, rHeight, 0, 0, m_hInstance, 0);

	if (!m_hwnd)
	{
		MessageBox(0, L"CreateWindow Failed.", 0, 0);
		return false;
	}

	return true;
}

void DX12Window::Show()
{
	ShowWindow(m_hwnd, SW_SHOW);
	UpdateWindow(m_hwnd);
}

void DX12Window::SetWindowTitle(std::wstring title)
{
	m_windowTitle = title;
	SetWindowText(m_hwnd, title.c_str());
}

