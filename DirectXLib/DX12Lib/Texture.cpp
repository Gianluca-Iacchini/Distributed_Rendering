#include "Texture.h"
#include "GraphicsCore.h"
#include "CommandContext.h"
#include <iostream>

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

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

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

    CommandContext::InitializeTexture(*this, 1, &texResourceData);

    if (m_hCpuDescriptorHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
    {
        m_hCpuDescriptorHandle = AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

    // Not passing a srv desc here, so the view will be created with the default values
    s_device->GetComPtr()->CreateShaderResourceView(m_resource.Get(), nullptr, m_hCpuDescriptorHandle);
}

void Texture::CreateFromTGAFile(const std::wstring& filename, bool sRGB)
{
    DirectX::TexMetadata texMetaData;
    DirectX::ScratchImage scratchImage;

    ThrowIfFailed(DirectX::LoadFromTGAFile(filename.c_str(), &texMetaData, scratchImage));

    if (sRGB)
        texMetaData.format = DirectX::MakeSRGB(texMetaData.format);

    D3D12_RESOURCE_DESC texDesc = {};

    switch (texMetaData.dimension)
    {
    case DirectX::TEX_DIMENSION_TEXTURE1D:
        texDesc = CD3DX12_RESOURCE_DESC::Tex1D(texMetaData.format, texMetaData.width, (UINT16)texMetaData.arraySize);
    	break;
    case DirectX::TEX_DIMENSION_TEXTURE2D:
        texDesc = CD3DX12_RESOURCE_DESC::Tex2D(texMetaData.format, texMetaData.width, (UINT)texMetaData.height, (UINT16)texMetaData.arraySize);
		break;
    case DirectX::TEX_DIMENSION_TEXTURE3D:
		texDesc = CD3DX12_RESOURCE_DESC::Tex3D(texMetaData.format, texMetaData.width, (UINT)texMetaData.height, (UINT16)texMetaData.depth);
    default:
        throw std::exception("Invalid texture dimension");
        break;
    }

    Create2D(scratchImage.GetImage(0, 0, 0)->rowPitch, texMetaData.width, texMetaData.height, texMetaData.format, scratchImage.GetImage(0, 0, 0)->pixels);
}
