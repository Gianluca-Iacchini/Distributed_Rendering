#ifndef ADAPTER_H
#define ADAPTER_H

#include "Helpers.h"

class CIDXGIFactory;

class Adapter
{
public:

	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetComPointer() const;
	IDXGIAdapter4* Get() const { return m_adapter.Get(); }
	IDXGIAdapter4** GetAddressOf() { return m_adapter.GetAddressOf(); }
	Adapter(Microsoft::WRL::ComPtr<IDXGIAdapter4> m_adapter);
	Adapter(CIDXGIFactory& factory, bool useWarp = false, DXGI_GPU_PREFERENCE gpuPreference = DXGI_GPU_PREFERENCE_UNSPECIFIED);
	static std::vector<Adapter> GetAllAdapters(CIDXGIFactory& factory);

	Microsoft::WRL::ComPtr<IDXGIOutput> GetAdapterOutput(UINT outputIndex);

	DXGI_ADAPTER_DESC GetDesc() const;
	~Adapter() {}
private:	

	DXGI_ADAPTER_DESC m_adapterDesc;
	Microsoft::WRL::ComPtr<IDXGIAdapter4> m_adapter;
};

#endif // !ADAPTER_H



