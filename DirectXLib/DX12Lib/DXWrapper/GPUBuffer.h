#pragma once

#include "Resource.h"

namespace DX12Lib
{
	class CommandContext;
	class ReadBackBuffer;

	class GPUBuffer : public Resource
	{
	public:
		virtual ~GPUBuffer() = default;
		
		virtual void Create(UINT32 numElements, UINT32 elementSize);
		
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetSRV() const { return m_srv; }
		const D3D12_CPU_DESCRIPTOR_HANDLE& GetUAV() const { return m_uav; }

		D3D12_VERTEX_BUFFER_VIEW VertexBufferView(size_t Offset, uint32_t Size, uint32_t Stride) const;
		D3D12_VERTEX_BUFFER_VIEW VertexBufferView(size_t BaseVertexIndex = 0) const
		{
			size_t Offset = BaseVertexIndex * m_elementSize;
			return VertexBufferView(Offset, (uint32_t)(m_bufferSize - Offset), m_elementSize);
		}

		D3D12_INDEX_BUFFER_VIEW IndexBufferView(size_t Offset, uint32_t Size, bool b32Bit = false) const;
		D3D12_INDEX_BUFFER_VIEW IndexBufferView(size_t StartIndex = 0) const
		{
			size_t Offset = StartIndex * m_elementSize;
			return IndexBufferView(Offset, (uint32_t)(m_bufferSize - Offset), m_elementSize == 4);
		}

		size_t GetBufferSize() const { return m_bufferSize; }
		UINT32 GetElementCount() const { return m_elementCount; }
		UINT32 GetElementSize() const { return m_elementSize; }

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
		friend class ReadBackBuffer;

	public:
		virtual ~StructuredBuffer() = default;

		virtual void CreateDerivedViews() override;

	};

	class TypedBuffer : public GPUBuffer
	{
	public:
		TypedBuffer(DXGI_FORMAT Format) : m_dataFormat(Format) {}
		virtual void CreateDerivedViews() override;

	protected:
		DXGI_FORMAT m_dataFormat;
	};

	class ReadBackBuffer : public GPUBuffer
	{
	public:
		virtual ~ReadBackBuffer() = default;
		virtual void Create(UINT32 numElements, UINT32 elementSize) override;
		void Create(StructuredBuffer& buffer);

		void* ReadBack(GPUBuffer& srcBuffer);

		virtual void CreateDerivedViews() override {}
	};
}