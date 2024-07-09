#pragma once

#include "Resource.h"

namespace DX12Lib
{
	class GPUBuffer : public Resource
	{
	public:
		virtual ~GPUBuffer() = default;
		
		void Create(UINT32 numElements, UINT32 elementSize);
		
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetSRV() const { return m_srv; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetUAV() const { return m_uav; }

	protected:
		GPUBuffer();
		D3D12_RESOURCE_DESC DescribeBuffer();
		virtual void CreateDerivedViews() = 0;

	protected:
		size_t m_bufferSize;
		UINT32 m_elementCount;
		UINT32 m_elementSize;

		D3D12_RESOURCE_FLAGS m_resourceFlags;

		D3D12_CPU_DESCRIPTOR_HANDLE m_srv;
		D3D12_CPU_DESCRIPTOR_HANDLE m_uav;
	};

	class ByteAddressBuffer : public GPUBuffer
	{
	public:
		virtual ~ByteAddressBuffer() = default;

	protected:
		virtual void CreateDerivedViews() override;
	};

	class StructuredBuffer : public GPUBuffer
	{
	public:
		virtual ~StructuredBuffer() = default;

		virtual void CreateDerivedViews() override;

		ByteAddressBuffer& GetCounterBuffer() { return m_counterBuffer; }

		const D3D12_CPU_DESCRIPTOR_HANDLE& GetCounterSRV(DX12Lib::CommandContext& context);
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetCounterUAV(DX12Lib::CommandContext& context);

	private:
		ByteAddressBuffer m_counterBuffer;
	};
}