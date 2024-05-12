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

	protected:
		D3D12_RESOURCE_DESC DescribeTex2D(UINT width, UINT height, UINT arraySize, UINT numMips, DXGI_FORMAT format, UINT flags);

		void AssociateWithResource(Microsoft::WRL::ComPtr<ID3D12Resource> resource, D3D12_RESOURCE_STATES currentState);
		void CreateTextureResource(Device& device, const D3D12_RESOURCE_DESC& resourceDesc, D3D12_CLEAR_VALUE clearValue);

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

