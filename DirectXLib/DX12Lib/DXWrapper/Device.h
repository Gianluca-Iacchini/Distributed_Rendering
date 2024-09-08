#pragma once

#include <d3d12.h>

namespace DX12Lib {

	class Adapter;
	class CommandList;
	class CommandQueue;
	class CommandAllocator;

	class Device
	{
	public:

		Device();
		~Device();

		bool InitializeApp(Adapter* adapter = nullptr);

		UINT GetDescriptorSize(D3D12_DESCRIPTOR_HEAP_TYPE type) const { return m_device->GetDescriptorHandleIncrementSize(type); };



		UINT RtvDescriptorSize = 0;
		UINT DsvDescriptorSize = 0;
		UINT CbvSrvUavDescriptorSize = 0;
		UINT SamplerDescriptorSize = 0;

	private:

		Microsoft::WRL::ComPtr<ID3D12Device> m_device;
		Microsoft::WRL::ComPtr<ID3D12Device5> m_dxrDevice;

	public:
		operator ID3D12Device* () const { return m_device.Get(); }

		ID3D12Device* operator->() const { return m_device.Get(); }

		ID3D12Device* Get() const { return m_device.Get(); }
		ID3D12Device** GetAddressOf() { return m_device.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12Device> GetComPtr() const { return m_device; }

		ID3D12Device5* GetDXR() const { return m_dxrDevice.Get(); }
		ID3D12Device5** GetDXRAddressOf() { return m_dxrDevice.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12Device5> GetDXRComPtr() const { return m_dxrDevice; }
	};
}



