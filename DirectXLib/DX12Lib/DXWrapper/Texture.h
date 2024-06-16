#pragma once
#include "DX12Lib/DXWrapper/Resource.h"
#include <unordered_map>
#include <mutex>

namespace DX12Lib {

	class Texture : public Resource
	{
		friend class TextureManager;

	public:
		D3D12_CPU_DESCRIPTOR_HANDLE GetSRV() const { return m_hCpuDescriptorHandle; }
		Texture();
	private:

		Texture(D3D12_CPU_DESCRIPTOR_HANDLE hCpuDescriptorHandle)
			: m_hCpuDescriptorHandle(hCpuDescriptorHandle), m_width(0), m_height(0), m_depth(0), m_isLoaded(false)
		{}

		virtual void OnDestroy() override;

		void Create2D(size_t rowPitchBytes, size_t Width, size_t Height, DXGI_FORMAT format, const void* initData);
		void Create3D(size_t rowPitchBytes, size_t Width, size_t Height, size_t Depth, DXGI_FORMAT format, const void* initData);

		void CreateFromFile(const std::wstring& filename, bool sRGB);

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
		enum class DefaultTextures
		{
			MAGENTA = 0,
			BLACK_OPAQUE,
			BLACK_TRANSPARENT,
			WHITE_OPAQUE,
			WHITE_TRANSPARENT,
			RED_OPAQUE,
			NORMAL_MAP,
			NUM_DEFAULT_TEXTURES
		};

	public:
		TextureManager();
		~TextureManager() = default;

		SharedTexture LoadFromFile(const std::wstring& filename, bool sRGB);
		SharedTexture CreateTexture2D(size_t rowPitchBytes, size_t Width, size_t Height, DXGI_FORMAT format, const void* initData, const std::wstring& texName = L"");

		SharedTexture DefaultTextures[(UINT)DefaultTextures::NUM_DEFAULT_TEXTURES];

	private:
		std::unordered_map<std::wstring, std::shared_ptr<Texture>> m_textureCache;
		std::mutex m_mutex;
	};

}