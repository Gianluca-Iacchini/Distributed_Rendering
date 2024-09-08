#include "DX12Lib/pch.h"

#include "Device.h"
#include "DX12Lib/DXWrapper/Adapter.h"

using namespace Microsoft::WRL;
using namespace DX12Lib;

Device::Device()
{
}

bool Device::InitializeApp(Adapter* adapter)
{
	IDXGIAdapter* dxgiAdapter = nullptr;

	if (adapter != nullptr)
	{
		dxgiAdapter = adapter->Get();
	}

	HRESULT hr = D3D12CreateDevice(dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(m_device.GetAddressOf()));

	RtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	DsvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	CbvSrvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	SamplerDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

	HRESULT hrDXR = m_device->QueryInterface(IID_PPV_ARGS(m_dxrDevice.GetAddressOf()));

	if (FAILED(hrDXR))
	{
		DXLIB_CORE_WARN("Failed to create ID3D12Device5, DXR will not be available.");
	}

	return SUCCEEDED(hr);
}

Device::~Device()
{
}
