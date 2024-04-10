#include "Resource.h"
#include "Device.h"
#include "DescriptorHeap.h"

using namespace Microsoft::WRL;

Resource::Resource(ComPtr<ID3D12Resource> resource) : m_resource(resource)
{
}

Resource::Resource(Device& device, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clearValue)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&desc,
		D3D12_RESOURCE_STATE_COMMON,
		clearValue,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));
}

Resource::Resource(Device& device, int width, int height, int depth, DXGI_FORMAT format,
	D3D12_RESOURCE_STATES initState, D3D12_RESOURCE_FLAGS flags, const D3D12_CLEAR_VALUE* clearValue)
{
	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Alignment = 0;
	desc.Width = width;
	desc.Height = height;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.Format = format;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	desc.Flags = flags;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

	ThrowIfFailed(device.GetComPtr()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&desc,
		initState,
		clearValue,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));
}

Resource::~Resource()
{
}


void Resource::CreateView(struct ResourceView& view, DescriptorHeap& descriptorHeap)
{
	descriptorHeap.AddDescriptor(m_resource.Get(), view);
	m_views.push_back(view);
}