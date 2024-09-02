#pragma once

#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "assert.h"

namespace CVGI
{


	class BufferManager
	{
	public:
		BufferManager() {}
		~BufferManager() {}

		UINT AddStructuredBuffer(UINT32 elementCount, size_t elementSize);
		UINT AddByteAddressBuffer();
		
		void RemoveBuffer(int index);
		void ResizeBuffer(int index, UINT32 elementCount);

		void AllocateBuffers();

		void ZeroBuffer(DX12Lib::CommandContext& context, UINT index);

		void TransitionAll(DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers = false);

		DX12Lib::GPUBuffer& GetBuffer(UINT index) { return *m_buffers[index]; }

		DX12Lib::DescriptorHandle& GetUAVHandle() { return m_uavHandle; }
		DX12Lib::DescriptorHandle& GetSRVHandle() { return m_srvHandle; }

		template<typename T>
		inline T ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex);

	private:
		std::vector<std::unique_ptr<DX12Lib::GPUBuffer>> m_buffers;

		DX12Lib::DescriptorHandle m_uavHandle;
		DX12Lib::DescriptorHandle m_srvHandle;
	};

	template<typename T>
	inline T BufferManager::ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex)
	{
		assert(bufferIndex < m_buffers.size() && bufferIndex >= 0);

		GPUBuffer* buffer = m_buffers[bufferIndex].get();

		ReadBackBuffer readBuffer;
		readBuffer.Create(buffer->GetElementCount(), buffer->GetElementSize());



		context.CopyBuffer(readBuffer, *buffer);

		context.Flush(true);

		void* data = readBuffer.ReadBack(*buffer);
		return reinterpret_cast<T>(data);
	}
}


