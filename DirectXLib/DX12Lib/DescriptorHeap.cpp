#include "DescriptorHeap.h"
#include "Device.h"
#include "GraphicsCore.h"

using namespace Microsoft::WRL;

DescriptorAllocator::DescriptorAllocator(D3D12_DESCRIPTOR_HEAP_TYPE type)
	: m_descriptorHeapType(type), m_currentHeap(nullptr), m_descriptorSize(0)
{	
	m_remainingFreeHandles = sm_numDescriptorsPerHeap;
	m_currentHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
}


std::mutex DescriptorAllocator::sm_allocationMutex;
std::vector<ComPtr<ID3D12DescriptorHeap>> DescriptorAllocator::sm_descriptorHeapPool;

void DescriptorAllocator::DestroyAll()
{
	sm_descriptorHeapPool.clear();
}

ID3D12DescriptorHeap* DescriptorAllocator::RequestNewHeap(D3D12_DESCRIPTOR_HEAP_TYPE type)
{
	std::lock_guard<std::mutex> LockGuard(sm_allocationMutex);

	D3D12_DESCRIPTOR_HEAP_DESC desc;
	desc.Type = type;
	desc.NumDescriptors = sm_numDescriptorsPerHeap;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	desc.NodeMask = 0;

	ComPtr<ID3D12DescriptorHeap> heap;

	ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf())));
	sm_descriptorHeapPool.emplace_back(heap);

	return heap.Get();
}

/* From Microsoft MiniEngine */
/// <summary>
/// Allocate a new descriptor heap if the current one can't allocate the requested number of descriptors
/// </summary>
/// <param name="count"></param>
/// <returns></returns>
D3D12_CPU_DESCRIPTOR_HANDLE DescriptorAllocator::Allocate(UINT count)
{
	if (m_currentHeap == nullptr || m_remainingFreeHandles < count)
	{
		m_currentHeap = RequestNewHeap(m_descriptorHeapType);
		m_currentHandle = m_currentHeap->GetCPUDescriptorHandleForHeapStart();
		m_remainingFreeHandles = m_currentHeap->GetDesc().NumDescriptors;

		if (m_descriptorSize == 0)
		{
			m_descriptorSize = Graphics::s_device->GetDescriptorSize(m_descriptorHeapType);
		}
	}

	D3D12_CPU_DESCRIPTOR_HANDLE handle = m_currentHandle;
	m_currentHandle.ptr += count * m_descriptorSize;
	m_remainingFreeHandles -= count;
	
	return handle;
}

void DescriptorHeap::Create(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t maxCount)
{
	m_desc.Type = type;
	m_desc.NumDescriptors = maxCount;
	m_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	m_desc.NodeMask = 0;

	ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateDescriptorHeap(&m_desc, IID_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

	m_descriptorSize = Graphics::s_device->GetDescriptorSize(type);
	m_numFreeDescriptors = m_desc.NumDescriptors;
	m_firstHandle = DescriptorHandle(m_heap->GetCPUDescriptorHandleForHeapStart(),
		m_heap->GetGPUDescriptorHandleForHeapStart());
	m_nextFreeHandle = m_firstHandle;
}

DescriptorHandle DescriptorHeap::Alloc(uint32_t count)
{
	assert(HasAvailableSpace(count) && "Descriptor Heap out of space");

	DescriptorHandle handle = m_nextFreeHandle;
	m_nextFreeHandle += count * m_descriptorSize;
	m_numFreeDescriptors -= count;

	return handle;
}

bool DescriptorHeap::ValidateHandle(const DescriptorHandle& handle) const
{
	if (handle.GetCPUPtr() < m_firstHandle.GetCPUPtr() ||
		handle.GetCPUPtr() >= m_firstHandle.GetCPUPtr() + m_desc.NumDescriptors * m_descriptorSize)
		return false;

	if (handle.GetGPUPtr() - m_firstHandle.GetGPUPtr() !=
		handle.GetCPUPtr() - m_firstHandle.GetCPUPtr())
		return false;

	return true;
}