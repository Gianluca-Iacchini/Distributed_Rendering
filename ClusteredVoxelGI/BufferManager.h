#pragma once

#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/DXWrapper/ColorBuffer.h"
#include "assert.h"
#include "DirectXMath.h"

namespace CVGI
{


	class BufferManager
	{
	public:
		BufferManager() {}
		~BufferManager() {}

		UINT AddStructuredBuffer(UINT32 elementCount, size_t elementSize);
		UINT AddByteAddressBuffer();
		UINT Add3DTextureBuffer(UINT32 width, UINT32 height, UINT32 depth, DXGI_FORMAT format);
		UINT Add3DTextureBuffer(DirectX::XMUINT3, DXGI_FORMAT format);
		
		void RemoveBuffer(int index);
		void ResizeBuffer(int index, UINT32 elementCount);
		void ReplaceBuffer(UINT index, UINT32 elementCount, size_t elementSize);

		void AllocateBuffers();

		void ZeroBuffer(DX12Lib::CommandContext& context, UINT index);

		void TransitionAll(DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers = false);

		DX12Lib::GPUBuffer& GetBuffer(UINT index);

		DX12Lib::DescriptorHandle& GetUAVHandle() { return m_uavHandle; }
		DX12Lib::DescriptorHandle& GetSRVHandle() { return m_srvHandle; }

		template<typename T>
		inline T ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex);

		void MoveDataTo(BufferManager& other);

	private:
		std::vector<std::shared_ptr<DX12Lib::Resource>> m_buffers;

		DX12Lib::DescriptorHandle m_uavHandle;
		DX12Lib::DescriptorHandle m_srvHandle;
	};

	template<typename T>
	inline T BufferManager::ReadFromBuffer(DX12Lib::CommandContext& context, UINT bufferIndex)
	{
		assert(bufferIndex < m_buffers.size() && bufferIndex >= 0);

		GPUBuffer& buffer = GetBuffer(bufferIndex);

		ReadBackBuffer readBuffer;
		readBuffer.Create(buffer.GetElementCount(), buffer.GetElementSize());



		context.CopyBuffer(readBuffer, buffer);

		context.Flush(true);

		void* data = readBuffer.ReadBack(buffer);
		return reinterpret_cast<T>(data);
	}
}


