#include "DX12Lib/pch.h"
#include "ColorBuffer.h"


using namespace Microsoft::WRL;
using namespace DX12Lib;

void ColorBuffer::CreateFromSwapChain(ComPtr<ID3D12Resource> baseResource)
{
	AssociateWithResource(baseResource, D3D12_RESOURCE_STATE_PRESENT);

	m_RTVHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	Graphics::s_device->GetComPtr()->CreateRenderTargetView(m_resource.Get(), nullptr, m_RTVHandle);
}

void DX12Lib::ColorBuffer::Create2D(uint32_t width, uint32_t height, uint32_t numMips, DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr)
{
	D3D12_RESOURCE_FLAGS flags = CombineResourceFlags();

	D3D12_RESOURCE_DESC resourceDesc = DescribeTex2D(width, height, 1, numMips, format, flags);

	resourceDesc.SampleDesc.Count = m_fragmentCount;
	resourceDesc.SampleDesc.Quality = 0;

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = format;
	clearValue.Color[0] = m_clearColor.R();
	clearValue.Color[1] = m_clearColor.G();
	clearValue.Color[2] = m_clearColor.B();
	clearValue.Color[3] = m_clearColor.A();

	CreateTextureResource(resourceDesc, &clearValue);

	D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_TEXTURE2D;

	if (m_fragmentCount > 1)
		dim = D3D12_SRV_DIMENSION_TEXTURE2DMS;

	CreateDerivedViews(dim, format, 1, numMips);
}

void DX12Lib::ColorBuffer::Create3D(uint32_t width, uint32_t height, uint32_t depth, uint32_t numMips, DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr)
{
	D3D12_RESOURCE_FLAGS flags = CombineResourceFlags();

	D3D12_RESOURCE_DESC resourceDesc = DescribeTex3D(width, height, depth, numMips, format, flags);

	resourceDesc.SampleDesc.Count = m_fragmentCount;
	resourceDesc.SampleDesc.Quality = 0;

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = format;
	clearValue.Color[0] = m_clearColor.R();
	clearValue.Color[1] = m_clearColor.G();
	clearValue.Color[2] = m_clearColor.B();
	clearValue.Color[3] = m_clearColor.A();

	CreateTextureResource(resourceDesc, &clearValue);
	CreateDerivedViews(D3D12_SRV_DIMENSION_TEXTURE3D, format, 1, numMips);
}

void DX12Lib::ColorBuffer::CreateArray(uint32_t width, uint32_t height, uint32_t arrayCount, DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr)
{
	D3D12_RESOURCE_FLAGS flags = CombineResourceFlags();

	D3D12_RESOURCE_DESC resourceDesc = DescribeTex2D(width, height, arrayCount, 1, format, flags);

	resourceDesc.SampleDesc.Count = m_fragmentCount;
	resourceDesc.SampleDesc.Quality = 0;

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = format;
	clearValue.Color[0] = m_clearColor.R();
	clearValue.Color[1] = m_clearColor.G();
	clearValue.Color[2] = m_clearColor.B();
	clearValue.Color[3] = m_clearColor.A();

	CreateTextureResource(resourceDesc, &clearValue);
	CreateDerivedViews(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, format, arrayCount, 1);
}

void DX12Lib::ColorBuffer::CreateDerivedViews(D3D12_SRV_DIMENSION texDimension, DXGI_FORMAT format, uint32_t arraySize, uint32_t numMips)
{
	assert(arraySize == 1 || numMips == 1);

	m_numMipMaps = numMips - 1;

	D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};

	rtvDesc.Format = format;
	uavDesc.Format = GetUAVFormat(format);
	srvDesc.Format = format;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	if (texDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
	{
		rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
		rtvDesc.Texture2DArray.MipSlice = 0;
		rtvDesc.Texture2DArray.FirstArraySlice = 0;
		rtvDesc.Texture2DArray.ArraySize = (UINT)arraySize;

		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
		uavDesc.Texture2DArray.MipSlice = 0;
		uavDesc.Texture2DArray.FirstArraySlice = 0;
		uavDesc.Texture2DArray.ArraySize = (UINT)arraySize;

		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
		srvDesc.Texture2DArray.MipLevels = numMips;
		srvDesc.Texture2DArray.MostDetailedMip = 0;
		srvDesc.Texture2DArray.FirstArraySlice = 0;
		srvDesc.Texture2DArray.ArraySize = (UINT)arraySize;
	}
	else if (texDimension == D3D12_SRV_DIMENSION_TEXTURE2DMS)
	{
		rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
		rtvDesc.Texture2D.MipSlice = 0;

		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DMS;
		uavDesc.Texture2D.MipSlice = 0;

		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
		srvDesc.Texture2D.MipLevels = numMips;
		srvDesc.Texture2D.MostDetailedMip = 0;
	}
	else if (texDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
	{
		rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
		rtvDesc.Texture2D.MipSlice = 0;

		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		uavDesc.Texture2D.MipSlice = 0;

		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = numMips;
		srvDesc.Texture2D.MostDetailedMip = 0;
	}
	else if (texDimension == D3D12_SRV_DIMENSION_TEXTURE3D)
	{
		rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
		rtvDesc.Texture3D.MipSlice = 0;
		rtvDesc.Texture3D.FirstWSlice = 0;
		rtvDesc.Texture3D.WSize = -1;

		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
		uavDesc.Texture3D.MipSlice = 0;
		uavDesc.Texture3D.FirstWSlice = 0;
		uavDesc.Texture3D.WSize = -1;

		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
		srvDesc.Texture3D.MipLevels = numMips;
		srvDesc.Texture3D.MostDetailedMip = 0;
	}
	else
	{
		DXLIB_CORE_ERROR("Unsuppored texture dimension in CreateDerivedViews()");
	}

	if (m_SRVHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_RTVHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_SRVHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	ID3D12Resource* resource = m_resource.Get();

	Graphics::s_device->GetComPtr()->CreateRenderTargetView(resource, &rtvDesc, m_RTVHandle);
	Graphics::s_device->GetComPtr()->CreateShaderResourceView(resource, &srvDesc, m_SRVHandle);

	// UAV views are not created for multi-sampled resources
	if (m_fragmentCount > 1)
		return;


	// Create the UAVs for each mip level

	for (uint32_t i = 0; i < numMips; ++i)
	{
		if (m_UAVHandle[i].ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			m_UAVHandle[i] = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		Graphics::s_device->GetComPtr()->CreateUnorderedAccessView(resource, nullptr, &uavDesc, m_UAVHandle[i]);
		uavDesc.Texture2D.MipSlice += 1;
	}
	
}

