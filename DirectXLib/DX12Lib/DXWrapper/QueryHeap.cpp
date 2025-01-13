#include "DX12Lib/pch.h"
#include "QueryHeap.h"

#define BITS_PER_WORD 64

using namespace DX12Lib;

void DX12Lib::QueryHeap::Create(D3D12_QUERY_HEAP_TYPE type, UINT16 maxElements)
{
	D3D12_QUERY_HEAP_DESC desc = {};
	desc.Count = maxElements;
	desc.Type = type;

	ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateQueryHeap(&desc, IID_PPV_ARGS(m_queryHeap.GetAddressOf())));

	m_type = type;
	m_maxElements = maxElements;

	m_occupancyBitmap.resize((size_t)ceilf(maxElements / 64.0f));
}

QueryHandle DX12Lib::QueryHeap::Alloc(UINT Count)
{
	int heapIndex = GetAllocIndex(Count);

	assert(heapIndex >= 0);

	return QueryHandle(heapIndex, Count);
}

void DX12Lib::QueryHeap::Free(QueryHandle handle)
{
	FreeRegion(handle.GetIndex(), handle.GetSize());
}

int DX12Lib::QueryHeap::GetAllocIndex(UINT allocSize)
{
    if (allocSize <= 0 || allocSize > m_maxElements) return -1; // Invalid request

    for (int wordIdx = 0; wordIdx < m_occupancyBitmap.size(); ++wordIdx) {
        uint64_t word = m_occupancyBitmap[wordIdx];

        // Check within this word for contiguous free bits
        for (int bit = 0; bit <= BITS_PER_WORD - allocSize; ++bit) {
            // 1 -> 00010000 -> 00001111
            uint64_t mask = ((1ULL << allocSize) - 1) << bit;
            if ((word & mask) == 0) { // Found a free block
                m_occupancyBitmap[wordIdx] |= mask;
                return wordIdx * BITS_PER_WORD + bit; 
            }
        }
    }

    return -1;
}

void DX12Lib::QueryHeap::FreeRegion(UINT index, UINT count)
{
    if ((index < 0) || ((index + count) > m_maxElements) || (count <= 0)) return; // Invalid input

    int wordIdx = index / BITS_PER_WORD;
    int bitOffset = index % BITS_PER_WORD;

    uint64_t mask = ((1ULL << count) - 1) << bitOffset; // Create mask for `size` bits
    m_occupancyBitmap[wordIdx] &= ~mask; // Clear the bits
}
