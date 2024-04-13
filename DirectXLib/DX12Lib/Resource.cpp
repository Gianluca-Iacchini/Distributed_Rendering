#include "Resource.h"
#include "Device.h"
#include "DescriptorHeap.h"

using namespace Microsoft::WRL;

Resource::Resource(std::shared_ptr<Device> device, ComPtr<ID3D12Resource> resource) : m_device(device), m_resource(resource)
{
	m_desc = m_resource->GetDesc();

	Recreate();
}

Resource::Resource(std::shared_ptr<Device> device, const D3D12_RESOURCE_DESC& desc, D3D12_CLEAR_VALUE* clearValue) : m_device(device), m_desc(desc)
{
	if (clearValue != nullptr)
	{
		m_clearValue = new D3D12_CLEAR_VALUE();
		*m_clearValue = *clearValue;
	}

	Recreate();
}

Resource::Resource(std::shared_ptr<Device> device, int width, int height, int depth, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags)
	: m_device(device)
{
	m_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	m_desc.Alignment = 0;
	m_desc.Width = width;
	m_desc.Height = height;
	m_desc.DepthOrArraySize = 1;
	m_desc.MipLevels = 1;
	m_desc.Format = format;
	m_desc.SampleDesc.Count = 1;
	m_desc.SampleDesc.Quality = 0;
	m_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	m_desc.Flags = flags;

	Recreate();
}

Resource::~Resource()
{
	if (m_clearValue != nullptr)
	{
		delete m_clearValue;
		m_clearValue = nullptr;
	}
}


void Resource::CreateView(struct ResourceView& view, DescriptorHeap& descriptorHeap)
{
	descriptorHeap.AddDescriptor(m_resource.Get(), view);
	m_views.push_back(view);
}

void Resource::SetClearValue(D3D12_CLEAR_VALUE& clearValue)
{
	
	if (m_clearValue != nullptr)
	{
		delete m_clearValue;
		m_clearValue = nullptr;
	}

	m_clearValue = new D3D12_CLEAR_VALUE();
	*m_clearValue = clearValue;
	
}

void Resource::Recreate()
{
	auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

	ThrowIfFailed(m_device->GetComPtr()->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&m_desc,
		D3D12_RESOURCE_STATE_COMMON,
		m_clearValue,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));
}