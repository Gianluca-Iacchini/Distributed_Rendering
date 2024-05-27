#pragma once

#include "ColorBuffer.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

namespace DX12Lib {

	class CIDXGIFactory;
	class DX12Window;
	class CommandQueue;

	class Swapchain
	{
	public:
		Swapchain(DX12Window& window, DXGI_FORMAT backBufferFormat);
		~Swapchain();

		void Initialize(CommandQueue& commandQueue);

		void Resize(UINT width, UINT height);

		ColorBuffer& GetBackBuffer(unsigned int index) { return m_backBuffers[index]; }
		ColorBuffer& GetCurrentBackBuffer() { return m_backBuffers[CurrentBufferIndex]; }

	public:
		static const unsigned int BufferCount = 3;
		unsigned int CurrentBufferIndex = 0;


	private:
		DXGI_SWAP_CHAIN_DESC1 m_swapchainDesc = {};

		bool m_isMsaaEnabled = false;
		unsigned int m_msaaQuality = 0;

		DX12Window& m_window;
		Microsoft::WRL::ComPtr<IDXGISwapChain1> m_swapchain;

		ColorBuffer m_backBuffers[BufferCount];

	public:

		Swapchain(Swapchain&& other) = default;
		Swapchain& operator=(Swapchain&& other) = default;

		Swapchain(Swapchain&) = delete;
		Swapchain& operator=(Swapchain&) = delete;

		Microsoft::WRL::ComPtr<IDXGISwapChain> GetComPointer() { return m_swapchain; }
		IDXGISwapChain1* Get() { return m_swapchain.Get(); }
		IDXGISwapChain1** GetAddressOf() { return m_swapchain.GetAddressOf(); }
	};

}



