#ifndef CIDXGI_FACTORY_H
#define CIDXGI_FACTORY_H

#include "Helpers.h"


class Swapchain;

class CIDXGIFactory
{
public:
	CIDXGIFactory();
	~CIDXGIFactory();



	
	//Swapchain CreateSwapchain(DXGI_SWAP_CHAIN_DESC swapChainDesc);
	//Swapchain CreateSwapchain(HWND hwnd, DXGI_FORMAT backBufferFormat, int swapchainBufferCount = 3, bool windowed = true);

private:

	Microsoft::WRL::ComPtr<IDXGIFactory6> m_factory;

public:

	Microsoft::WRL::ComPtr<IDXGIFactory6> GetComPtr() { return m_factory; }
	IDXGIFactory6* Get() { return m_factory.Get(); }
	IDXGIFactory6** GetAddressOf() { return m_factory.GetAddressOf(); }
	
	operator IDXGIFactory6* () const { return m_factory.Get(); }
	IDXGIFactory6* operator->() const { return m_factory.Get(); }

};
#endif // !CIDXGI_FACTORY_H



