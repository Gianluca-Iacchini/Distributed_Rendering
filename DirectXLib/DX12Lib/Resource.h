#include "Helpers.h"
#include "DescriptorHeap.h"

#ifndef RESOURCE_H
#define RESOURCE_H

class Device;



class Resource
{
public:

	Resource(Device& device, Microsoft::WRL::ComPtr<ID3D12Resource> resource);
	Resource(Device& device, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clearValue = nullptr);
	Resource(Device& device, int width, int height, int depth = 1, DXGI_FORMAT format = DXGI_FORMAT_R32G32B32_FLOAT,
		D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, const D3D12_CLEAR_VALUE* clearValue = nullptr);
	

	void SetSize(int width, int height, int depth = 1) { m_desc.Width = width, m_desc.Height = height, m_desc.DepthOrArraySize = depth; }
	void SetDimension(D3D12_RESOURCE_DIMENSION dimension) { m_desc.Dimension = dimension; }
	void SetFormat(DXGI_FORMAT format) { m_desc.Format = format; }
	void SetAlignment(UINT64 alignment) { m_desc.Alignment = alignment; }
	void SetFlags(D3D12_RESOURCE_FLAGS flags) { m_desc.Flags = flags; }
	void SetSampleDesc(UINT count, UINT quality) { m_desc.SampleDesc.Count = count, m_desc.SampleDesc.Quality = quality; }
	void SetLayout(D3D12_TEXTURE_LAYOUT layout) { m_desc.Layout = layout; }

	void SetResourceDesc(const D3D12_RESOURCE_DESC& desc) { m_desc = desc; }
	void SetClearValue(const D3D12_CLEAR_VALUE* clearValue) {
		if (clearValue != nullptr)
		{
			m_clearValue = new D3D12_CLEAR_VALUE();
			*m_clearValue = *clearValue;
		}
	}

	void RecreateResource();

	void CreateView(struct ResourceView& view, DescriptorHeap& descriptorHeap);

	~Resource();

	D3D12_RESOURCE_DESC GetDesc() const { return m_resource->GetDesc(); }
protected:
	Device& m_device;
	D3D12_RESOURCE_DESC m_desc = {};
	D3D12_CLEAR_VALUE* m_clearValue = nullptr;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
	std::vector<ResourceView> m_views;

public:
	ID3D12Resource* Get() const { return m_resource.Get(); }
	ID3D12Resource** GetAddressOf() { return m_resource.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12Resource> GetComPtr() const { return m_resource; }
	void ResetComPtr() { m_resource.Reset(); }

	Resource(Resource&&) = default;
	Resource& operator=(Resource&&) = default;

	Resource(Resource&) = delete;
	Resource& operator=(Resource&) = delete;
};



#endif // !RESOURCE_H


