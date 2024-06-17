#pragma once

#include "PixelBuffer.h"
#include "Color.h"

namespace DX12Lib {

	class ColorBuffer : public PixelBuffer
	{
	public:
		ColorBuffer(Color clearColor = Color(0.0f, 0.0f, 0.0f, 0.0f))
			: m_clearColor(clearColor), m_numMipMaps(0), m_fragmentCount(1), m_sampleCount(1)
		{
			m_RTVHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_SRVHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;

			for (int i = 0; i < _countof(m_UAVHandle); ++i)
			{
				m_UAVHandle[i].ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			}
		}

		~ColorBuffer() = default;

		void CreateFromSwapChain(Microsoft::WRL::ComPtr<ID3D12Resource> baseResource);

		void Create2D(uint32_t width, uint32_t height, uint32_t numMips,
			DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN);

		void Create3D(uint32_t width, uint32_t height, uint32_t depth, uint32_t numMips,
			DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN);

		void CreateArray(uint32_t width, uint32_t height, uint32_t arrayCount,
			DXGI_FORMAT format, D3D12_GPU_VIRTUAL_ADDRESS vidMemPtr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN);


		const D3D12_CPU_DESCRIPTOR_HANDLE& GetSRV() const { return m_SRVHandle; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetRTV() const { return m_RTVHandle; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetUAV(uint32_t mipSlice) const { return m_UAVHandle[mipSlice]; }

		Color GetClearColor() const { return m_clearColor; }
		void SetClearColor(Color clearColor) { m_clearColor = clearColor; }

		void SetMsaaMode(uint32_t numColorSamples, uint32_t numCoverageSamples)
		{
			assert(numCoverageSamples >= numColorSamples);

			m_fragmentCount = numCoverageSamples;
			m_sampleCount = numColorSamples;
		}

		//void GenerateMipMaps();

	protected:

		void CreateDerivedViews(D3D12_SRV_DIMENSION texDimension, DXGI_FORMAT format, uint32_t arraySize, uint32_t numMips = 1);

		D3D12_RESOURCE_FLAGS CombineResourceFlags() const
		{
			D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;

			if (flags == D3D12_RESOURCE_FLAG_NONE && m_fragmentCount == 1)
			{
				flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
			}

			return D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | flags;
		}

		static inline uint32_t ComputeNumMips(uint32_t width, uint32_t height)
		{
			uint32_t highBit;
			_BitScanReverse((unsigned long*)&highBit, width | height);
			return highBit + 1;
		}

	protected:

		Color m_clearColor;
		D3D12_CPU_DESCRIPTOR_HANDLE m_SRVHandle;
		D3D12_CPU_DESCRIPTOR_HANDLE m_RTVHandle;
		D3D12_CPU_DESCRIPTOR_HANDLE m_UAVHandle[12];

		uint32_t m_numMipMaps;
		uint32_t m_fragmentCount;
		uint32_t m_sampleCount;
	};
}




