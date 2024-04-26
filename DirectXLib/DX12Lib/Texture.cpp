#include "Texture.h"
#include "GraphicsCore.h"

using namespace Microsoft::WRL;
using namespace Graphics;

void Texture::Create2D(size_t rowPitchBytes, size_t Width, size_t Height, DXGI_FORMAT format, const void* initData)
{
    OnDestroy();

    m_currentState = D3D12_RESOURCE_STATE_COPY_DEST;

    m_width = (UINT)Width;
    m_height = (UINT)Height;
    m_depth = 1;

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = Width;
    texDesc.Height = (UINT)Height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = format;
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    ThrowIfFailed(s_device->GetComPtr()->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&texDesc,
		m_currentState,
		nullptr,
		IID_PPV_ARGS(m_resource.ReleaseAndGetAddressOf())));

    D3D12_SUBRESOURCE_DATA texResourceData;
    texResourceData.pData = initData;
    texResourceData.RowPitch = rowPitchBytes;
    texResourceData.SlicePitch = rowPitchBytes * Height;

    // Update subresources;

    if (m_hCpuDescriptorHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
    {
        m_hCpuDescriptorHandle = AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

    // Not passing a srv desc here, so the view will be created with the default values
    s_device->GetComPtr()->CreateShaderResourceView(m_resource.Get(), nullptr, m_hCpuDescriptorHandle);
}

void Texture::CreateFromTGAMemory(const void* _filePtr, bool sRGB)
{
    const uint8_t* filePtr = (const uint8_t*)_filePtr;

    // Skip first two bytes
    filePtr += 2;

    /*uint8_t imageTypeCode =*/ *filePtr++;

    // Ignore another 9 bytes
    filePtr += 9;

    uint16_t imageWidth = *(uint16_t*)filePtr;
    filePtr += sizeof(uint16_t);
    uint16_t imageHeight = *(uint16_t*)filePtr;
    filePtr += sizeof(uint16_t);
    uint8_t bitCount = *filePtr++;

    // Ignore another byte
    filePtr++;

    uint32_t* formattedData = new uint32_t[imageWidth * imageHeight];
    uint32_t* iter = formattedData;

    uint8_t numChannels = bitCount / 8;
    uint32_t numBytes = imageWidth * imageHeight * numChannels;

    switch (numChannels)
    {
    default:
        break;
    case 3:
        for (uint32_t byteIdx = 0; byteIdx < numBytes; byteIdx += 3)
        {
            *iter++ = 0xff000000 | filePtr[0] << 16 | filePtr[1] << 8 | filePtr[2];
            filePtr += 3;
        }
        break;
    case 4:
        for (uint32_t byteIdx = 0; byteIdx < numBytes; byteIdx += 4)
        {
            *iter++ = filePtr[3] << 24 | filePtr[0] << 16 | filePtr[1] << 8 | filePtr[2];
            filePtr += 4;
        }
        break;
    }

    Create2D(4 * imageWidth, imageWidth, imageHeight, sRGB ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM, formattedData);

    delete[] formattedData;
}
