#pragma once

#include "PixelBuffer.h"

namespace DX12Lib {

	class DepthBuffer : public PixelBuffer
	{
	public:
		DepthBuffer(float clearDepth = 1.0f, uint8_t clearStencil = 0)
			: m_clearDepth(clearDepth), m_clearStencil(clearStencil)
		{
			m_dsvHandle[0].ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_dsvHandle[1].ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_dsvHandle[2].ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_dsvHandle[3].ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_depthSRVHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_stencilSrvHandle.ptr = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
		}

		void Create(uint32_t width, uint32_t height, DXGI_FORMAT format);

		void Create(uint32_t width, uint32_t height, uint32_t numSamples, DXGI_FORMAT format);

		const D3D12_CPU_DESCRIPTOR_HANDLE& GetDSV() const { return m_dsvHandle[0]; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetDSV_Depth() const { return m_dsvHandle[1]; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetDSV_Stencil() const { return m_dsvHandle[2]; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetDSV_ReadOnly() const { return m_dsvHandle[3]; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetDepthSRV() const { return m_depthSRVHandle; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetStencilSRV() const { return m_stencilSrvHandle; }

		float GetClearDepth() const { return m_clearDepth; }
		uint8_t GetClearStencil() const { return m_clearStencil; }

	protected:

		void CreateDerivedViews(ID3D12Device* device, DXGI_FORMAT format);

		float m_clearDepth = 0.0f;
		uint8_t m_clearStencil = 0;
		D3D12_CPU_DESCRIPTOR_HANDLE m_dsvHandle[4];
		D3D12_CPU_DESCRIPTOR_HANDLE m_depthSRVHandle;
		D3D12_CPU_DESCRIPTOR_HANDLE m_stencilSrvHandle;
	};
}
