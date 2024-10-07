#include "DX12Lib/pch.h"
#include "UploadBuffer.h"

using namespace DX12Lib;

void DX12Lib::UploadBuffer::Create(size_t BufferSize)
{
    OnDestroy();

    m_BufferSize = BufferSize;

    // Create an upload buffer.  This is CPU-visible, but it's write combined memory, so
    // avoid reading back from it.
    D3D12_HEAP_PROPERTIES HeapProps;
    HeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    HeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    HeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    HeapProps.CreationNodeMask = 1;
    HeapProps.VisibleNodeMask = 1;

    // Upload buffers must be 1-dimensional
    D3D12_RESOURCE_DESC ResourceDesc = {};
    ResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ResourceDesc.Width = m_BufferSize;
    ResourceDesc.Height = 1;
    ResourceDesc.DepthOrArraySize = 1;
    ResourceDesc.MipLevels = 1;
    ResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    ResourceDesc.SampleDesc.Count = 1;
    ResourceDesc.SampleDesc.Quality = 0;
    ResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(Graphics::s_device->Get()->CreateCommittedResource(&HeapProps, D3D12_HEAP_FLAG_NONE, &ResourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_resource)));

    m_gpuVirtualAddress = m_resource->GetGPUVirtualAddress();
}

void* DX12Lib::UploadBuffer::Map(void)
{
    m_isMapped = true;

    auto range = CD3DX12_RANGE(0, m_BufferSize);

    void* memory;
    m_resource->Map(0, &range, &memory);
    return memory;
}

void DX12Lib::UploadBuffer::Unmap(size_t begin, size_t end)
{
    auto range = CD3DX12_RANGE(begin, std::min(end, m_BufferSize));
    m_resource->Unmap(0, &range);

    m_isMapped = false;
}
