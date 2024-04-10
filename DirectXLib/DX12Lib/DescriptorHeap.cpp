#include "DescriptorHeap.h"

using namespace Microsoft::WRL;

DescriptorHeap::DescriptorHeap(Device& device, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags) : m_device(device), m_descriptorHeapType(type), m_descriptorCount(numDescriptors)
{

	switch (type)
	{
	case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
		m_descriptorSize = m_device.CbvSrvUavDescriptorSize;
		break;
	case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
		m_descriptorSize = m_device.SamplerDescriptorSize;
		break;
	case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
		m_descriptorSize = m_device.RtvDescriptorSize;
		break;
	case D3D12_DESCRIPTOR_HEAP_TYPE_DSV:
		m_descriptorSize = m_device.DsvDescriptorSize;
		break;
	default:
		break;
	}

	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = numDescriptors;
	desc.Type = type;
	desc.Flags = flags;

	ThrowIfFailed(m_device.GetComPtr()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_descriptorHeap.GetAddressOf())));
}

void DescriptorHeap::AddDescriptor(ID3D12Resource* resource, ResourceView& view)
{
	assert((m_availableDescriptorIndex + 1 <= m_descriptorCount) && "Descriptor is not big enough");

	switch (view.descType)
	{
	case DescriptorType::SRV:
		assert(m_descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		m_device.GetComPtr()->CreateShaderResourceView(resource, view.view.SRV, GetCPUDescriptorHandle(m_availableDescriptorIndex));
		break;
	case DescriptorType::CBV:
		assert(m_descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		m_device.GetComPtr()->CreateConstantBufferView(view.view.CBV, GetCPUDescriptorHandle(m_availableDescriptorIndex));
		break;
	case DescriptorType::UAV:
		assert(m_descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		m_device.GetComPtr()->CreateUnorderedAccessView(resource, nullptr, view.view.UAV, GetCPUDescriptorHandle(m_availableDescriptorIndex));
		break;
	case DescriptorType::RTV:
		assert(m_descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_device.GetComPtr()->CreateRenderTargetView(resource, view.view.RTV, GetCPUDescriptorHandle(m_availableDescriptorIndex));
		break;
	case DescriptorType::DSV:
		assert(m_descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		m_device.GetComPtr()->CreateDepthStencilView(resource, view.view.DSV, GetCPUDescriptorHandle(m_availableDescriptorIndex));
		break;
	default:
		break;
	}

	view.DescriptorIndex = m_availableDescriptorIndex;
	m_availableDescriptorIndex++;
}
