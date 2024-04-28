#pragma once
#include "Helpers.h"
#include "Resource.h"

class Texture : public Resource
{
public:
	Texture()
		: m_width(0), m_height(0), m_depth(0)
	{ m_hCpuDescriptorHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN; }
	Texture(D3D12_CPU_DESCRIPTOR_HANDLE hCpuDescriptorHandle) 
		: m_hCpuDescriptorHandle(hCpuDescriptorHandle), m_width(0), m_height(0), m_depth(0)
	{}

	virtual void OnDestroy() override
	{
		Resource::OnDestroy();
		m_hCpuDescriptorHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
	}

	void Create2D(size_t rowPitchBytes, size_t Width, size_t Height, DXGI_FORMAT format, const void* initData);
	void Create3D(size_t rowPitchBytes, size_t Width, size_t Height, size_t Depth, DXGI_FORMAT format, const void* initData);

	void CreateFromTGAFile(const std::wstring& filename, bool sRGB);
	void CreateFromDDSFile(const std::wstring filename, bool sRGB) {};

	D3D12_CPU_DESCRIPTOR_HANDLE GetSRV() const { return m_hCpuDescriptorHandle; }

private:
	UINT m_width;
	UINT m_height;
	UINT m_depth;

	D3D12_CPU_DESCRIPTOR_HANDLE m_hCpuDescriptorHandle;
};

