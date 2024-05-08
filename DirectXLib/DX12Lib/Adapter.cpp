#include "pch.h"
#include "Adapter.h"
#include "CIDXGIFactory.h"

using namespace DX12Lib;

using Microsoft::WRL::ComPtr;

Adapter::Adapter(ComPtr<IDXGIAdapter4> adapter) : m_adapter(adapter)
{
	m_adapter->GetDesc(&m_adapterDesc);
}

Adapter::Adapter(CIDXGIFactory& factory, bool useWarp, DXGI_GPU_PREFERENCE gpuPreference)
{
	if (useWarp)
	{
		ThrowIfFailed(factory.GetComPtr()->EnumWarpAdapter(IID_PPV_ARGS(m_adapter.GetAddressOf())));
	}
	else
	{
		ThrowIfFailed(factory.GetComPtr()->EnumAdapterByGpuPreference(0, gpuPreference, IID_PPV_ARGS(m_adapter.GetAddressOf())));
	}

	m_adapter->GetDesc(&m_adapterDesc);
}

std::vector<Adapter> Adapter::GetAllAdapters(CIDXGIFactory& factory)
{
	UINT i = 0;
	ComPtr<IDXGIAdapter1> adapter1 = nullptr;
	ComPtr<IDXGIAdapter4> adapter4 = nullptr;

	std::vector<Adapter> adapterList;

	while (factory.GetComPtr()->EnumAdapters1(i, adapter1.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
	{

		ThrowIfFailed(adapter1.As(&adapter4));
		adapterList.push_back(Adapter(adapter4));
		i++;
	}

	return adapterList;
}

Microsoft::WRL::ComPtr<IDXGIOutput> Adapter::GetAdapterOutput(UINT outputIndex)
{
	ComPtr<IDXGIOutput> output;

	if (m_adapter->EnumOutputs(outputIndex, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
	{
		return output;
	}

	return nullptr;
}

ComPtr<IDXGIAdapter4> Adapter::GetComPointer() const
{
	return m_adapter;
}

DXGI_ADAPTER_DESC Adapter::GetDesc() const
{
	return m_adapterDesc;
}