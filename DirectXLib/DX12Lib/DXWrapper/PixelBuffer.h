#pragma once

#include "Resource.h"

namespace DX12Lib {

	class Device;

	class PixelBuffer : public Resource
	{
	public:

		PixelBuffer() : m_width(0), m_height(0), m_arraySize(1), m_format(DXGI_FORMAT_UNKNOWN) {}

		UINT GetWidth() const { return m_width; }
		UINT GetHeight() const { return m_height; }
		UINT GetArraySize() const { return m_arraySize; }
		const DXGI_FORMAT& GetFormat() const { return m_format; }
		void CreateTextureResource(UINT width, UINT height, UINT arraySize=1, UINT numMips=1, DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
		void CreateTextureResource(const D3D12_RESOURCE_DESC& resourceDesc, D3D12_CLEAR_VALUE* clearValue = nullptr, D3D12_HEAP_PROPERTIES* heapProps = nullptr);
	protected:

		D3D12_RESOURCE_DESC DescribeTex2D(UINT width, UINT height, UINT arraySize, UINT numMips, DXGI_FORMAT format, UINT flags);
		D3D12_RESOURCE_DESC DescribeTex3D(UINT width, UINT height, UINT depth, UINT numMips, DXGI_FORMAT format, UINT flags);

		void AssociateWithResource(Microsoft::WRL::ComPtr<ID3D12Resource> resource, D3D12_RESOURCE_STATES currentState);


		static DXGI_FORMAT GetBaseFormat(DXGI_FORMAT Format);
		static DXGI_FORMAT GetUAVFormat(DXGI_FORMAT Format);
		static DXGI_FORMAT GetDSVFormat(DXGI_FORMAT Format);
		static DXGI_FORMAT GetDepthFormat(DXGI_FORMAT Format);
		static DXGI_FORMAT GetStencilFormat(DXGI_FORMAT Format);
		static size_t BytesPerPixel(DXGI_FORMAT Format);

	protected:
		UINT m_width;
		UINT m_height;
		UINT m_arraySize;
		DXGI_FORMAT m_format;
	};
}

