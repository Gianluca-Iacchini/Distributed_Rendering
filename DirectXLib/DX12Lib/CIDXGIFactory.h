#ifndef CIDXGI_FACTORY_H
#define CIDXGI_FACTORY_H

#include "Helpers.h"


class Swapchain;

class CIDXGIFactory
{
public:
	CIDXGIFactory();
	~CIDXGIFactory();


	Microsoft::WRL::ComPtr<IDXGIFactory6> GetComPointer() { return m_factory; }
	IDXGIFactory6* GetRawPointer() { return m_factory.Get(); }
	
	//Swapchain CreateSwapchain(DXGI_SWAP_CHAIN_DESC swapChainDesc);
	//Swapchain CreateSwapchain(HWND hwnd, DXGI_FORMAT backBufferFormat, int swapchainBufferCount = 3, bool windowed = true);

private:

	Microsoft::WRL::ComPtr<IDXGIFactory6> m_factory;

};
#endif // !CIDXGI_FACTORY_H



