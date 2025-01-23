#pragma once
#include "windows.h"

#if CVGI_DX12
#define IMGUIWND HWND
#elif CVGI_GL
struct GLFWwindow;
#define IMGUIWND GLFWwindow*
#endif

namespace Commons
{
	class UIHelpers
	{
	public:
#if CVGI_DX12
		static void InitializeIMGUI(HWND imguiWnd);
#elif CVGI_GL
		static void InitializeIMGUI(GLFWwindow* imguiWnd);
#endif
		static void ShutdownIMGUI();

		static bool OddIntegerSlider(const char* label, int* value, int min, int max);

		static void StartFrame();

		static void EndFrame();

		static void ControlInfoBlock(bool isConnected = false);

		static void ConnectedClient(const char* peerAddr, UINT32 ping);

	private:

	};
}