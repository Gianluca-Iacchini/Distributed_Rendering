#include "Helpers.h"
#include "Device.h"
#include <mutex>

#ifndef DESCRIPTOR_HEAP_H
#define DESCRIPTOR_HEAP_H

class DescriptorAllocator
{
public:
	DescriptorAllocator(D3D12_DESCRIPTOR_HEAP_TYPE type);

	D3D12_CPU_DESCRIPTOR_HANDLE Allocate(UINT count);

	static void DestroyAll(void);

protected:
	static const UINT sm_numDescriptorsPerHeap = 256;
	static std::mutex sm_allocationMutex;
	static std::vector<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>> sm_descriptorHeapPool;
	static ID3D12DescriptorHeap* RequestNewHeap(D3D12_DESCRIPTOR_HEAP_TYPE type);

protected:
	D3D12_DESCRIPTOR_HEAP_TYPE m_descriptorHeapType;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_currentHeap;
	CD3DX12_CPU_DESCRIPTOR_HANDLE m_currentHandle;
	UINT m_descriptorSize;
	UINT m_remainingFreeHandles;
};

class DescriptorHandle
{
public:
	DescriptorHandle()
	{
		m_cpuHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
		m_gpuHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
	}

	DescriptorHandle(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle)
		: m_cpuHandle(cpuHandle)
		, m_gpuHandle(gpuHandle)
	{
	}

	const D3D12_CPU_DESCRIPTOR_HANDLE* operator&() const { return &m_cpuHandle; }
	operator D3D12_CPU_DESCRIPTOR_HANDLE() const { return m_cpuHandle; }
	operator D3D12_GPU_DESCRIPTOR_HANDLE() const { return m_gpuHandle; }

	void operator += (INT offsetScaledByDescriptorSize)
	{
		if (m_cpuHandle.ptr != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			m_cpuHandle.ptr += offsetScaledByDescriptorSize;
		}
		if (m_gpuHandle.ptr != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			m_gpuHandle.ptr += offsetScaledByDescriptorSize;
		}
	}

	DescriptorHandle operator+ (INT offsetScaledByDescriptorSize) const
	{
		DescriptorHandle ret = *this;
		ret += offsetScaledByDescriptorSize;
		return ret;
	}

	size_t GetCPUPtr() const { return m_cpuHandle.ptr; }
	size_t GetGPUPtr() const { return m_gpuHandle.ptr; }
	bool IsNull() const { return m_cpuHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN; }
	bool IsShaderVisible() const { return m_gpuHandle.ptr != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN; }

private:

	D3D12_CPU_DESCRIPTOR_HANDLE m_cpuHandle = {};
	D3D12_GPU_DESCRIPTOR_HANDLE m_gpuHandle = {};
};

class DescriptorHeap
{
public:
	DescriptorHeap() {}
	~DescriptorHeap() { Destroy(); }
	void Create(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t maxCount);
	void Destroy() { m_heap = nullptr; }

	bool HasAvailableSpace(uint32_t count) const { return m_numFreeDescriptors >= count; }
	DescriptorHandle Alloc(uint32_t count = 1);
	
	DescriptorHandle operator[](uint32_t arrayIdx) const { return m_firstHandle + arrayIdx * m_descriptorSize; }

	uint32_t GetOffsetOfHandle(const DescriptorHandle& handle)
	{
		return static_cast<uint32_t>(handle.GetCPUPtr() - m_firstHandle.GetCPUPtr()) / m_descriptorSize;
	}

	bool ValidateHandle(const DescriptorHandle& handle) const;


	uint32_t GetDescriptorSize() const { return m_descriptorSize; }

private:
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_heap;
	D3D12_DESCRIPTOR_HEAP_DESC m_desc = {};
	uint32_t m_descriptorSize = 0;
	uint32_t m_numFreeDescriptors = 0;
	DescriptorHandle m_firstHandle = {};
	DescriptorHandle m_nextFreeHandle = {};

public:
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> GetComPtr() const { return m_heap; }
	ID3D12DescriptorHeap* Get() const { return m_heap.Get(); }
	ID3D12DescriptorHeap** GetAddressOf() { return m_heap.GetAddressOf(); }
};

#endif // !DESCRIPTOR_HEAP_H




