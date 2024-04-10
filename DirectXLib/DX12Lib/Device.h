#ifndef DEVICE_H
#define DEVICE_H

#include "Helpers.h"

class Adapter;
class CommandList;
class CommandQueue;
class CommandAllocator;

class Device
{
public:

	Device();
	~Device();

	bool Initialize(Adapter* adapter = nullptr);

	//CommandList CreateCommandList(D3D12_COMMAND_LIST_TYPE cmdlistType);


	ID3D12Device* Get() const { return m_device.Get(); }
	ID3D12Device** GetAddressOf() { return m_device.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12Device> GetComPtr() const { return m_device; }

	UINT RtvDescriptorSize = 0;
	UINT DsvDescriptorSize = 0;
	UINT CbvSrvUavDescriptorSize = 0;
	UINT SamplerDescriptorSize = 0;

private:

	Microsoft::WRL::ComPtr<ID3D12Device> m_device;
};

#endif // !DEVICE_H



