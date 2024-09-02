#include "BufferManager.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

using namespace CVGI;
using namespace DX12Lib;

UINT BufferManager::AddStructuredBuffer(UINT32 elementCount, size_t elementSize)
{
	std::unique_ptr<StructuredBuffer> buffer = std::make_unique<StructuredBuffer>();
	buffer->Create(elementCount, elementSize);

	m_buffers.push_back(std::move(buffer));

	return m_buffers.size() - 1;
}

UINT CVGI::BufferManager::AddByteAddressBuffer()
{
	std::unique_ptr<ByteAddressBuffer> buffer = std::make_unique<ByteAddressBuffer>();
	buffer->Create(1, sizeof(UINT32));

	m_buffers.push_back(std::move(buffer));

	return m_buffers.size() - 1;
}

void CVGI::BufferManager::RemoveBuffer(int index)
{
	if (index >= m_buffers.size())
		return;

	m_buffers.erase(m_buffers.begin() + index);

	for (UINT i = index; i < m_buffers.size(); i++)
	{
		auto descriptorStart = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * i;

		if (m_uavHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			Graphics::s_device->Get()->CopyDescriptorsSimple(
				1,
				m_uavHandle + descriptorStart,
				m_buffers[i]->GetUAV(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
			);
		}

		if (m_srvHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			Graphics::s_device->Get()->CopyDescriptorsSimple(
				1,
				m_srvHandle + descriptorStart,
				m_buffers[i]->GetSRV(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
			);
		}
	}
}

void CVGI::BufferManager::ResizeBuffer(int index, UINT32 elementCount)
{
	if (index >= m_buffers.size())
		return;

	m_buffers[index]->OnDestroy();
	m_buffers[index]->Create(elementCount, m_buffers[index]->GetElementSize());

	if (m_uavHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_uavHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * index,
			m_buffers[index]->GetUAV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}


	if (m_srvHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_srvHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * index,
			m_buffers[index]->GetSRV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}

}

void CVGI::BufferManager::AllocateBuffers()
{
	m_uavHandle =  Graphics::Renderer::s_textureHeap->Alloc(m_buffers.size());
	m_srvHandle = Graphics::Renderer::s_textureHeap->Alloc(m_buffers.size());

	for (UINT i = 0; i < m_buffers.size(); i++)
	{
		auto descriptorStart = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * i;

		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1, 
			m_uavHandle + descriptorStart, 
			m_buffers[i]->GetUAV(), 
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);

		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1, 
			m_srvHandle + descriptorStart, 
			m_buffers[i]->GetSRV(), 
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}
}

void CVGI::BufferManager::ZeroBuffer(DX12Lib::CommandContext& context, UINT index)
{
	assert(index < m_buffers.size());

	UploadBuffer uploadBuffer;
	uploadBuffer.Create(m_buffers[index]->GetElementCount() * m_buffers[index]->GetElementSize());

	void* mappedData = uploadBuffer.Map();

	memset(mappedData, 0, m_buffers[index]->GetElementCount() * m_buffers[index]->GetElementSize());



	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(*m_buffers[index], D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(m_buffers[index]->Get(), uploadBuffer.Get());

	context.Flush(true);

	uploadBuffer.Unmap();
}

void CVGI::BufferManager::TransitionAll(DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers)
{
	for (auto& buffer : m_buffers)
	{
		context.TransitionResource(*buffer, newState);
	}

	if (flusBarriers)
	{
		context.FlushResourceBarriers();
	}
}


