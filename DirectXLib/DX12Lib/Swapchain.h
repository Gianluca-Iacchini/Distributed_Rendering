#include "Helpers.h"


#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

class CIDXGIFactory;
class DX12Window;
class Device;
class CommandQueue;
class Resource;

class Swapchain
{
public:
	Swapchain(DX12Window& window, int nBufferCount = 3, DXGI_FORMAT backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM);
	~Swapchain();

	void Finalize(CIDXGIFactory& factory, Device& device, CommandQueue& commandQueue);

	Resource* GetBuffer(UINT index) { return m_backBufferResources[index].get(); }
	Resource* GetCurrentBackBuffer() { return m_backBufferResources[m_currentBackBufferIndex].get(); }
	void Resize(UINT width, UINT height);



private:
	
	DXGI_SWAP_CHAIN_DESC1 m_swapchainDesc = {};

	unsigned int m_currentBackBufferIndex = 0;
	bool m_isMsaaEnabled = false;
	unsigned int m_msaaQuality = 0;
	
	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;


	std::vector<std::unique_ptr<Resource>> m_backBufferResources;

	DX12Window& m_window;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_swapchain;
	
public:
	const unsigned int BufferCount = 3;
	unsigned int CurrentBufferIndex = 0;

public:

	Swapchain(Swapchain&& other) = default;
	Swapchain& operator=(Swapchain&& other) = default;

	Swapchain(Swapchain&) = delete;
	Swapchain& operator=(Swapchain&) = delete;

	Microsoft::WRL::ComPtr<IDXGISwapChain> GetComPointer() { return m_swapchain; }
	IDXGISwapChain1* Get() { return m_swapchain.Get(); }
	IDXGISwapChain1** GetAddressOf() { return m_swapchain.GetAddressOf(); }
};

#endif // !SWAPCHAIN_H



