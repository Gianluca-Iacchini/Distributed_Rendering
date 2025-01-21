#pragma once
#include "windows.h"

namespace Commons
{
	class UIHelpers
	{
	public:
		static void InitializeIMGUI(HWND windowHandle);
		static void ShutdownIMGUI();

		static bool OddIntegerSlider(const char* label, int* value, int min, int max);

		static void StartFrame();

	private:

	};
}