#include "Helpers.h"
#include "Device.h"

#ifndef DESCRIPTOR_HEAP_H
#define DESCRIPTOR_HEAP_H

enum class DescriptorType
{
	SRV,
	RTV,
	DSV,
	UAV,
	CBV
};

union ViewDesc
{
	const D3D12_SHADER_RESOURCE_VIEW_DESC* SRV;
	const D3D12_RENDER_TARGET_VIEW_DESC* RTV;
	const D3D12_DEPTH_STENCIL_VIEW_DESC* DSV;
	const D3D12_UNORDERED_ACCESS_VIEW_DESC* UAV;
	const D3D12_CONSTANT_BUFFER_VIEW_DESC* CBV;
};

struct ResourceView
{
public:
	DescriptorType descType;
	ViewDesc view;
	unsigned int DescriptorIndex = 0;
};

class DescriptorHeap
{
public:
	DescriptorHeap(Device& device, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT numDescriptors = 1, D3D12_DESCRIPTOR_HEAP_FLAGS flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	
	~DescriptorHeap() {};


	D3D12_CPU_DESCRIPTOR_HANDLE GetCPUDescriptorHandle(int offset = 0)
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_descriptorHeap->GetCPUDescriptorHandleForHeapStart(), offset, m_descriptorSize);
		return handle;
	}

	D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandle(int offset = 0)
	{
		CD3DX12_GPU_DESCRIPTOR_HANDLE handle(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), offset, m_descriptorSize);
		return handle;
	}

	UINT GetDescriptorCount() const { return m_descriptorCount; }

	void AddDescriptor(ID3D12Resource* resource, ResourceView& view);

private:

	Device& m_device;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
	UINT m_descriptorCount = 0;
	UINT m_descriptorSize = 0;
	UINT m_availableDescriptorIndex = 0;

	D3D12_DESCRIPTOR_HEAP_TYPE m_descriptorHeapType;

public:
	DescriptorHeap(DescriptorHeap&&) = default;
	DescriptorHeap& operator=(DescriptorHeap&&) = default;

	DescriptorHeap(DescriptorHeap&) = delete;
	DescriptorHeap& operator=(DescriptorHeap&) = delete;

	ID3D12DescriptorHeap* Get() const { return m_descriptorHeap.Get(); }
	ID3D12DescriptorHeap** GetAddressOf() { return m_descriptorHeap.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> GetComPtr() const { return m_descriptorHeap; }
};

#endif // !DESCRIPTOR_HEAP_H




