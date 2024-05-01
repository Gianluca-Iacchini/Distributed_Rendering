#pragma once
#include "Helpers.h"
#include "Resource.h"
#include <unordered_map>
#include <mutex>

class Texture : public Resource
{
	friend class TextureManager;

public:
	D3D12_CPU_DESCRIPTOR_HANDLE GetSRV() const { return m_hCpuDescriptorHandle; }
	Texture()
		: m_width(0), m_height(0), m_depth(0), m_isLoaded(false)
	{
		m_hCpuDescriptorHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
	}
private:

	Texture(D3D12_CPU_DESCRIPTOR_HANDLE hCpuDescriptorHandle)
		: m_hCpuDescriptorHandle(hCpuDescriptorHandle), m_width(0), m_height(0), m_depth(0), m_isLoaded(false)
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

	void WaitForLoad() const { while ((volatile bool&)m_isLoaded == false) { std::this_thread::yield(); } }

private:
	UINT m_width;
	UINT m_height;
	UINT m_depth;
	bool m_isLoaded = false;

	D3D12_CPU_DESCRIPTOR_HANDLE m_hCpuDescriptorHandle;
};

using SharedTexture = std::shared_ptr<Texture>;

class TextureManager
{
public:
	TextureManager() = default;
	~TextureManager() = default;

	SharedTexture LoadFromFile(const std::wstring& filename, bool sRGB);

private:
	std::unordered_map<std::wstring, std::shared_ptr<Texture>> m_textureCache;
	std::mutex m_mutex;
};
