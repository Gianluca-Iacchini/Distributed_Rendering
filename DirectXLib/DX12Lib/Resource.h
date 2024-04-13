#include "Helpers.h"
#include "DescriptorHeap.h"

#ifndef RESOURCE_H
#define RESOURCE_H

class Device;



class Resource
{
public:

	Resource(std::shared_ptr<Device> device, Microsoft::WRL::ComPtr<ID3D12Resource> resource);
	Resource(std::shared_ptr<Device> device, const D3D12_RESOURCE_DESC& desc, D3D12_CLEAR_VALUE* clearValue = nullptr);
	Resource(std::shared_ptr<Device> device, int width, int height, int depth = 1, DXGI_FORMAT format = DXGI_FORMAT_R32G32B32_FLOAT, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
	
	void SetClearValue(D3D12_CLEAR_VALUE& clearValue);
	void SetDimension(D3D12_RESOURCE_DIMENSION dimension) { m_desc.Dimension = dimension; }
	void SetSize(int width, int height, int depth = 1) { m_desc.Width = width; m_desc.Height = height; m_desc.DepthOrArraySize = depth; }
	void SetFormat(DXGI_FORMAT format) { m_desc.Format = format; }
	void SetSampleDesc(UINT count, UINT quality) { m_desc.SampleDesc.Count = count; m_desc.SampleDesc.Quality = quality; }
	void SetMipLevels(UINT mipLevels) { m_desc.MipLevels = mipLevels; }
	void SetLayout(D3D12_TEXTURE_LAYOUT layout) { m_desc.Layout = layout; }

	void Recreate();

	void CreateView(struct ResourceView& view, DescriptorHeap& descriptorHeap);

	~Resource();

	D3D12_RESOURCE_DESC GetDesc() const { return m_resource->GetDesc(); }
protected:
	Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
	D3D12_RESOURCE_DESC m_desc;
	D3D12_CLEAR_VALUE* m_clearValue = nullptr;

	std::shared_ptr<Device> m_device;

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


