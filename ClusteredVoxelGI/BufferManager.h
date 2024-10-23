#pragma once

#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/DXWrapper/ColorBuffer.h"
#include "assert.h"
#include "DirectXMath.h"
#include "DX12Lib/Commons/CommandContext.h"

namespace CVGI
{


	class BufferManager
	{
	public:
		BufferManager() {}
		~BufferManager() {}

		UINT AddStructuredBuffer(UINT32 elementCount, size_t elementSize);
		UINT AddByteAddressBuffer(UINT32 elementCount = 1);
		UINT Add2DTextureBuffer(UINT32 width, UINT32 height, DXGI_FORMAT format);
		UINT Add2DTextureBuffer(DirectX::XMUINT2 size, DXGI_FORMAT format);

		UINT Add3DTextureBuffer(UINT32 width, UINT32 height, UINT32 depth, DXGI_FORMAT format);
		UINT Add3DTextureBuffer(DirectX::XMUINT3 size, DXGI_FORMAT format);
		
		void RemoveBuffer(int index);
		void ResizeBuffer(int index, UINT32 elementCount);
		void ReplaceBuffer(UINT index, UINT32 elementCount, size_t elementSize);

		void AllocateBuffers();

		void ZeroBuffer(DX12Lib::CommandContext& context, UINT index);

		void TransitionAll(DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers = false);
		void TransitionBuffer(UINT buffIndex, DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers = false);

		DX12Lib::GPUBuffer& GetBuffer(UINT index);

		DX12Lib::DescriptorHandle& GetUAVHandle() { return m_uavHandle; }
		DX12Lib::DescriptorHandle& GetSRVHandle() { return m_srvHandle; }

		UINT GetBufferCount() { return m_buffers.size(); }

		template<typename T>
		inline T ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex, UINT32 bufferSize = 0);

		void MoveDataTo(BufferManager& other);

	private:
		std::vector<std::shared_ptr<DX12Lib::Resource>> m_buffers;

		DX12Lib::DescriptorHandle m_uavHandle;
		DX12Lib::DescriptorHandle m_srvHandle;
	};

	template<typename T>
	inline T BufferManager::ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex, UINT32 bufferSize)
	{
		assert(bufferIndex < m_buffers.size() && bufferIndex >= 0);

		DX12Lib::GPUBuffer& buffer = BufferManager::GetBuffer(bufferIndex);

		UINT elementCount = 1;
		UINT64 elementSize = bufferSize;

		if (bufferSize == 0)
		{
			elementCount = buffer.GetElementCount();
			elementSize = buffer.GetElementSize();
		}

		DX12Lib::ReadBackBuffer readBuffer;
		readBuffer.Create(elementCount, elementSize);

		if (bufferSize == 0)
		{
			context.CopyBuffer(readBuffer, buffer);
		}
		else
		{
			context.CopyBufferRegion(readBuffer, 0, buffer, 0, elementCount * elementSize);
		}

		context.Flush(true);

		void* data = readBuffer.ReadBack(elementCount * elementSize);
		return reinterpret_cast<T>(data);
	}
}


