#include "pch.h"
#include "Texture.h"
#include "CommandContext.h"



using namespace DX12Lib;
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
		IID_PPV_ARGS(m_resource.GetAddressOf())));

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

    m_isLoaded = true;
}

void Texture::CreateFromFile(const std::wstring& filename, bool sRGB)
{
    DirectX::TexMetadata texMetaData;
    DirectX::ScratchImage scratchImage;

    if (filename.substr(filename.find_last_of(L".") + 1) == L"tga")
    {
        ThrowIfFailed(DirectX::LoadFromTGAFile(filename.c_str(), DirectX::TGA_FLAGS_NONE, &texMetaData, scratchImage));
    }
    else if (filename.substr(filename.find_last_of(L".") + 1) == L"dds")
    {
        ThrowIfFailed(DirectX::LoadFromDDSFile(filename.c_str(), DirectX::DDS_FLAGS_NONE, &texMetaData, scratchImage));
    }
    else if (filename.substr(filename.find_last_of(L".") + 1) == L"jpg" || filename.substr(filename.find_last_of(L".") + 1) == L"png")
    {
        ThrowIfFailed(DirectX::LoadFromWICFile(filename.c_str(), DirectX::WIC_FLAGS_NONE, &texMetaData, scratchImage));
    }
    else
        throw std::exception("Invalid texture file format");



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

TextureManager::TextureManager()
{
    UINT32 defaultTextureData[(UINT)DefaultTextures::NUM_DEFAULT_TEXTURES]
    {
        0xFFFF00FF, // Magenta
        0xFF000000, // Black Opaque
        0x00000000, // Black Transparent
        0xFFFFFFFF, // White Opaque
        0x00FFFFFF, // White Transparent
        0x80808080  // Normal Map
    };

    for (UINT i = 0; i < (UINT)TextureManager::DefaultTextures::NUM_DEFAULT_TEXTURES; ++i)
    {
        std::wstring defaultName = L"DefaultTexture" + std::to_wstring(i);
		this->DefaultTextures[i] = CreateTexture2D(4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, &defaultTextureData[i], defaultName);
	}

    CommandContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE_DIRECT);
}

SharedTexture TextureManager::LoadFromFile(const std::wstring& filename, bool sRGB)
{
    SharedTexture texture = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::wstring key = filename;

        if (sRGB)
            key += L"_sRGB";

        auto cacheTexture = m_textureCache.find(key);

        if (cacheTexture != m_textureCache.end())
        {
            texture = cacheTexture->second;
            texture->WaitForLoad();
            return texture;
        }

        texture = std::make_shared<Texture>();
        m_textureCache[key] = texture;
    }


    texture->CreateFromFile(filename, sRGB);

    return texture;
}

SharedTexture TextureManager::CreateTexture2D(size_t rowPitchBytes, size_t Width, size_t Height, DXGI_FORMAT format, const void* initData, const std::wstring& name)
{
    SharedTexture texture = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::wstring key = name;

        if (name == L"")
            key = L"Texture2D_" + std::to_wstring(rowPitchBytes) + L"rpb" + std::to_wstring(Width) + L"x" + std::to_wstring(Height) + L"_" + std::to_wstring(format);

        auto cacheTexture = m_textureCache.find(key);

        if (cacheTexture != m_textureCache.end())
        {
            texture = cacheTexture->second;
            texture->WaitForLoad();
            return texture;
        }

        texture = std::make_shared<Texture>();
        m_textureCache[key] = texture;
    }

    texture->Create2D(rowPitchBytes, Width, Height, format, initData);

    return texture;
}
