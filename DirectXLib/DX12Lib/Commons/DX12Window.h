#pragma once

#include <Windows.h>
#include <string>

namespace DX12Lib {

	class DX12Window
	{
	public:
		DX12Window(HINSTANCE hInstance, int width, int height, std::wstring windowTitle);
		~DX12Window() {};

		bool Create();
		void Show();
		HWND GetWindowHandle() const { return m_hwnd; }
		int GetWidth() const { return m_width; }
		int GetHeight() const { return m_height; }
		std::wstring GetWindowTitle() const { return m_windowTitle; }
		void SetWindowTitle(std::wstring title);

	private:
		WNDCLASSEX m_windowClass;
		HINSTANCE m_hInstance;
		HWND m_hwnd;
		int m_width;
		int m_height;
		std::wstring m_windowTitle;
	};
}



