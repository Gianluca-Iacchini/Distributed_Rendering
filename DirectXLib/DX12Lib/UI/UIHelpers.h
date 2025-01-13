#pragma once

namespace DX12Lib
{
	class UIHelpers
	{
	public:
		static void InitializeIMGUI(DX12Lib::DX12Window* window);
		static void ShutdownIMGUI();

		static bool OddIntegerSlider(const char* label, int* value, int min, int max);

		static void StartFrame();

	private:

	};
}