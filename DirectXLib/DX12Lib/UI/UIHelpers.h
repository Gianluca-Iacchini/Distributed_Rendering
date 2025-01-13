#pragma once

namespace DX12Lib
{
	class UIHelpers
	{
	public:
		static void InitializeIMGUI(DX12Lib::DX12Window* window);
		static void ShutdownIMGUI();

		static void StartFrame();

	private:

	};
}