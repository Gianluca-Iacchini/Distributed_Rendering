#include "pch.h"

#include "DepthBuffer.h"


using namespace Graphics;
using namespace Microsoft::WRL;
using namespace DX12Lib;

void DepthBuffer::Create(uint32_t width, uint32_t height, DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr)
{
	Create(width, height, 1, format, vidMemPtr);
}

void DepthBuffer::Create(uint32_t width, uint32_t height, uint32_t numSamples, DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr)
{
	D3D12_RESOURCE_DESC depthBufferDesc = DescribeTex2D(width, height, 1, 1, format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
	depthBufferDesc.SampleDesc.Count = numSamples;

	D3D12_CLEAR_VALUE optimizedClearValue = {};
	optimizedClearValue.Format = format;
	optimizedClearValue.DepthStencil.Depth = m_clearDepth;
	optimizedClearValue.DepthStencil.Stencil = m_clearStencil;

	CreateTextureResource(*s_device, depthBufferDesc, optimizedClearValue);
	CreateDerviedViews(s_device->Get(), format);
}

void DepthBuffer::CreateDerviedViews(ID3D12Device* device, DXGI_FORMAT format)
{
	D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;

	dsvDesc.Format = GetDSVFormat(format);

	// Check for multisampling
	if (m_resource->GetDesc().SampleDesc.Count == 1)
	{
		dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		dsvDesc.Texture2D.MipSlice = 0;
	}
	else
	{
		dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
	}

	if (m_dsvHandle[0].ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_dsvHandle[0] = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		m_dsvHandle[1] = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	}

	dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
	device->CreateDepthStencilView(m_resource.Get(), &dsvDesc, m_dsvHandle[0]);

	dsvDesc.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
	device->CreateDepthStencilView(m_resource.Get(), &dsvDesc, m_dsvHandle[1]);

	DXGI_FORMAT stencilReadFormat = GetStencilFormat(format);
	if (stencilReadFormat != DXGI_FORMAT_UNKNOWN)
	{
		if (m_dsvHandle[2].ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			m_dsvHandle[2] = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
			m_dsvHandle[3] = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		}

		dsvDesc.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
		device->CreateDepthStencilView(m_resource.Get(), &dsvDesc, m_dsvHandle[2]);

		dsvDesc.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL;
		device->CreateDepthStencilView(m_resource.Get(), &dsvDesc, m_dsvHandle[3]);
	}
	else
	{
		m_dsvHandle[2] = m_dsvHandle[0];
		m_dsvHandle[3] = m_dsvHandle[1];
	}

	if (m_depthSRVHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_depthSRVHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = GetDepthFormat(format);

	if (dsvDesc.ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2D)
	{
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;
	}
	else
	{
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
	}

	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	device->CreateShaderResourceView(m_resource.Get(), &srvDesc, m_depthSRVHandle);

	if (stencilReadFormat != DXGI_FORMAT_UNKNOWN)
	{
		if (m_stencilSrvHandle.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			m_stencilSrvHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		srvDesc.Format = stencilReadFormat;
		srvDesc.Texture2D.PlaneSlice = 1;
	
		device->CreateShaderResourceView(m_resource.Get(), &srvDesc, m_stencilSrvHandle);
	}
}
