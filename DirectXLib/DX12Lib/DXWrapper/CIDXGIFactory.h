#pragma once

#include <dxgi1_6.h>

namespace DX12Lib {
	class Swapchain;

	class CIDXGIFactory
	{
	public:
		CIDXGIFactory(UINT flags = 0);
		~CIDXGIFactory();

	private:

		Microsoft::WRL::ComPtr<IDXGIFactory6> m_factory;

	public:

		Microsoft::WRL::ComPtr<IDXGIFactory6> GetComPtr() { return m_factory; }
		IDXGIFactory6* Get() { return m_factory.Get(); }
		IDXGIFactory6** GetAddressOf() { return m_factory.GetAddressOf(); }

		operator IDXGIFactory6* () const { return m_factory.Get(); }
		IDXGIFactory6* operator->() const { return m_factory.Get(); }

	};
}



