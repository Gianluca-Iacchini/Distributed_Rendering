#include "DX12Lib/pch.h"
#include "GPUBuffer.h"

using namespace DX12Lib;

GPUBuffer::GPUBuffer() : m_bufferSize(0), m_elementSize(0), m_elementCount(0)
{
	m_resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	m_uav.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
	m_srv.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
}

D3D12_RESOURCE_DESC DX12Lib::GPUBuffer::DescribeBuffer()
{
	assert(m_bufferSize != 0);

	D3D12_RESOURCE_DESC desc = {};
	desc.Alignment = 0;
	desc.DepthOrArraySize = 1;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Flags = m_resourceFlags;
	desc.Format = DXGI_FORMAT_UNKNOWN;
	desc.Height = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Width = (UINT64)m_bufferSize;

	return desc;
}

void GPUBuffer::Create(UINT32 numElements, UINT32 elementSize)
{
	OnDestroy();

	m_elementCount = numElements;
	m_elementSize = elementSize;
	m_bufferSize = numElements * elementSize;

	D3D12_RESOURCE_DESC desc = DescribeBuffer();

	m_currentState = D3D12_RESOURCE_STATE_COMMON;

	D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

	ThrowIfFailed(Graphics::s_device->Get()->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		m_currentState,
		nullptr,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));

	m_gpuVirtualAddress = m_resource->GetGPUVirtualAddress();

	CreateDerivedViews();
}

D3D12_VERTEX_BUFFER_VIEW DX12Lib::GPUBuffer::VertexBufferView(size_t Offset, uint32_t Size, uint32_t Stride) const
{
	D3D12_VERTEX_BUFFER_VIEW VBView;
	VBView.BufferLocation = m_gpuVirtualAddress + Offset;
	VBView.SizeInBytes = Size;
	VBView.StrideInBytes = Stride;
	return VBView;
}

D3D12_INDEX_BUFFER_VIEW DX12Lib::GPUBuffer::IndexBufferView(size_t Offset, uint32_t Size, bool b32Bit) const
{
	D3D12_INDEX_BUFFER_VIEW IBView;
	IBView.BufferLocation = m_gpuVirtualAddress + Offset;
	IBView.Format = b32Bit ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
	IBView.SizeInBytes = Size;
	return IBView;
}

void DX12Lib::ByteAddressBuffer::CreateDerivedViews()
{
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.NumElements = (UINT)m_elementCount;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;

	if (m_srv.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_srv = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateShaderResourceView(m_resource.Get(), &srvDesc, m_srv);


	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	uavDesc.Buffer.NumElements = m_elementCount;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

	if (m_uav.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_uav = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateUnorderedAccessView(m_resource.Get(), nullptr, &uavDesc, m_uav);
}

void DX12Lib::StructuredBuffer::CreateDerivedViews()
{
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.NumElements = m_elementCount;
	srvDesc.Buffer.StructureByteStride = m_elementSize;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	if (m_srv.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_srv = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateShaderResourceView(m_resource.Get(), &srvDesc, m_srv);


	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.Buffer.NumElements = m_elementCount;
	uavDesc.Buffer.StructureByteStride = m_elementSize;
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	if (m_uav.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_uav = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateUnorderedAccessView(m_resource.Get(), nullptr, &uavDesc, m_uav);

}

void DX12Lib::TypedBuffer::CreateDerivedViews()
{
	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
	SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	SRVDesc.Format = m_dataFormat;
	SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	SRVDesc.Buffer.NumElements = m_elementCount;
	SRVDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	if (m_srv.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		m_srv = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CreateShaderResourceView(m_resource.Get(), &SRVDesc, m_srv);

	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	UAVDesc.Format = m_dataFormat;
	UAVDesc.Buffer.NumElements = m_elementCount;
	UAVDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	if (m_uav.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		m_uav = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CreateUnorderedAccessView(m_resource.Get(), nullptr, &UAVDesc, m_uav);
}

void DX12Lib::ReadBackBuffer::Create(UINT32 numElements, UINT32 elementSize)
{
	m_elementCount = numElements;
	m_elementSize = elementSize;
	m_bufferSize = numElements * elementSize;

	this->m_resourceFlags &= ~D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	this->m_resourceFlags &= ~D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;


	D3D12_RESOURCE_DESC desc = DescribeBuffer();

	m_currentState = D3D12_RESOURCE_STATE_COPY_DEST;

	D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

	ThrowIfFailed(Graphics::s_device->Get()->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		m_currentState,
		nullptr,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));

	m_gpuVirtualAddress = m_resource->GetGPUVirtualAddress();
}

void DX12Lib::ReadBackBuffer::Create(StructuredBuffer& buffer)
{
	this->Create(buffer.m_elementCount, m_elementSize);
}

void* DX12Lib::ReadBackBuffer::ReadBack(UINT32 byteSize)
{
	void* pData;
	CD3DX12_RANGE readRange(0, byteSize);
	
	ThrowIfFailed(m_resource->Map(0, &readRange, &pData));

	return pData;
}
