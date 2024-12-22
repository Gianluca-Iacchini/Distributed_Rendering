#include "BufferManager.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

using namespace VOX;
using namespace DX12Lib;

UINT BufferManager::AddStructuredBuffer(UINT32 elementCount, size_t elementSize, std::wstring bufferName)
{
	std::shared_ptr<StructuredBuffer> buffer = std::make_shared<StructuredBuffer>();
	buffer->Create(elementCount, elementSize);
	buffer->Get()->SetName(bufferName.c_str());

	m_buffers.push_back(buffer);

	return m_buffers.size() - 1;
}

UINT BufferManager::AddByteAddressBuffer(UINT32 elementCount, std::wstring bufferName)
{
	std::shared_ptr<ByteAddressBuffer> buffer = std::make_shared<ByteAddressBuffer>();
	buffer->Create(elementCount, sizeof(UINT32));
	buffer->Get()->SetName(bufferName.c_str());

	m_buffers.push_back(buffer);

	return m_buffers.size() - 1;
}

UINT BufferManager::Add2DTextureBuffer(UINT32 width, UINT32 height, DXGI_FORMAT format, std::wstring bufferName)
{
	std::shared_ptr<ColorBuffer> buffer = std::make_shared<ColorBuffer>();
	buffer->Create3D(width, height, 1, 1, format);
	buffer->Get()->SetName(bufferName.c_str());

	m_buffers.push_back(buffer);

	return m_buffers.size() - 1;
}

UINT BufferManager::Add2DTextureBuffer(DirectX::XMUINT2 size, DXGI_FORMAT format, std::wstring bufferName)
{
	return Add2DTextureBuffer(size.x, size.y, format, bufferName);
}

UINT BufferManager::Add3DTextureBuffer(UINT32 width, UINT32 height, UINT32 depth, DXGI_FORMAT format, std::wstring bufferName)
{
	std::shared_ptr<ColorBuffer> buffer = std::make_shared<ColorBuffer>();
	buffer->Create3D(width, height, depth, 1, format);
	buffer->Get()->SetName(bufferName.c_str());

	m_buffers.push_back(buffer);

	return m_buffers.size() - 1;
}

UINT BufferManager::Add3DTextureBuffer(DirectX::XMUINT3 textureSize, DXGI_FORMAT format, std::wstring bufferName)
{
	return Add3DTextureBuffer(textureSize.x, textureSize.y, textureSize.z, format, bufferName);
}

void BufferManager::RemoveBuffer(int index)
{
	if (index >= m_buffers.size())
		return;

	m_buffers.erase(m_buffers.begin() + index);

	for (UINT i = index; i < m_buffers.size(); i++)
	{
		auto descriptorStart = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * i;

		Resource* resource = m_buffers[i].get();

		if (resource == nullptr)
			continue;

		bool isGpuBuffer = dynamic_cast<GPUBuffer*>(resource) != nullptr;

		const D3D12_CPU_DESCRIPTOR_HANDLE& uavHandle = isGpuBuffer ? static_cast<GPUBuffer*>(resource)->GetUAV() : static_cast<ColorBuffer*>(resource)->GetUAV(0);
		const D3D12_CPU_DESCRIPTOR_HANDLE& srvHandle = isGpuBuffer ? static_cast<GPUBuffer*>(resource)->GetSRV() : static_cast<ColorBuffer*>(resource)->GetSRV();

		if (m_uavHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			Graphics::s_device->Get()->CopyDescriptorsSimple(
				1,
				m_uavHandle + descriptorStart,
				uavHandle,
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
			);
		}

		if (m_srvHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
		{
			Graphics::s_device->Get()->CopyDescriptorsSimple(
				1,
				m_srvHandle + descriptorStart,
				srvHandle,
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
			);
		}
	}
}

void BufferManager::ResizeBuffer(int index, UINT32 elementCount)
{
	if (index >= m_buffers.size())
		return;

	GPUBuffer* buffer = dynamic_cast<GPUBuffer*>(m_buffers[index].get());

	if (buffer == nullptr)
		return;

	buffer->OnDestroy();
	buffer->Create(elementCount, buffer->GetElementSize());

	if (m_uavHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_uavHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * index,
			buffer->GetUAV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}


	if (m_srvHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_srvHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * index,
			buffer->GetSRV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}

}

void BufferManager::ReplaceBuffer(UINT index, UINT32 elementCount, size_t elementSize)
{
	UINT actualIndex = index;
	if (index >= m_buffers.size())
	{
		actualIndex = m_buffers.size();
		m_buffers.resize(actualIndex + 1);
	}

	std::shared_ptr<StructuredBuffer> buffer = std::make_shared<StructuredBuffer>();
	buffer->Create(elementCount, elementSize);


	if (m_srvHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_srvHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * actualIndex,
			buffer->GetSRV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}

	if (m_uavHandle.GetGPUPtr() != D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1,
			m_uavHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * actualIndex,
			buffer->GetUAV(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}

	m_buffers[actualIndex] = buffer;
}

void BufferManager::AllocateBuffers()
{
	m_uavHandle =  Graphics::Renderer::s_textureHeap->Alloc(m_buffers.size());
	m_srvHandle = Graphics::Renderer::s_textureHeap->Alloc(m_buffers.size());

	for (UINT i = 0; i < m_buffers.size(); i++)
	{
		auto descriptorStart = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * i;
		Resource* resource = m_buffers[i].get();

		if (resource == nullptr)
			continue;

		bool isGpuBuffer = dynamic_cast<GPUBuffer*>(resource) != nullptr;

		const D3D12_CPU_DESCRIPTOR_HANDLE& uavHandle = isGpuBuffer ? static_cast<GPUBuffer*>(resource)->GetUAV() : static_cast<ColorBuffer*>(resource)->GetUAV(0);
		const D3D12_CPU_DESCRIPTOR_HANDLE& srvHandle = isGpuBuffer ? static_cast<GPUBuffer*>(resource)->GetSRV() : static_cast<ColorBuffer*>(resource)->GetSRV();


		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1, 
			m_uavHandle + descriptorStart, 
			uavHandle, 
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);

		Graphics::s_device->Get()->CopyDescriptorsSimple(
			1, 
			m_srvHandle + descriptorStart, 
			srvHandle, 
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
		);
	}
}

void BufferManager::SwapBuffers(UINT buffIndex0, UINT buffIndex1)
{
	assert(buffIndex0 < m_buffers.size());
	assert(buffIndex1 < m_buffers.size());

	auto descriptorStart0 = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * buffIndex0;
	auto descriptorStart1 = Graphics::Renderer::s_textureHeap->GetDescriptorSize() * buffIndex1;
	
	Resource* resource0 = m_buffers[buffIndex0].get();
	Resource* resource1 = m_buffers[buffIndex1].get();

	if (resource0 == nullptr || resource1 == nullptr)
		return;

	bool isGpuBuffer = dynamic_cast<GPUBuffer*>(resource0) != nullptr;

	const D3D12_CPU_DESCRIPTOR_HANDLE& uavHandle0 = isGpuBuffer ? static_cast<GPUBuffer*>(resource0)->GetUAV() : static_cast<ColorBuffer*>(resource0)->GetUAV(0);
	const D3D12_CPU_DESCRIPTOR_HANDLE& srvHandle0 = isGpuBuffer ? static_cast<GPUBuffer*>(resource0)->GetSRV() : static_cast<ColorBuffer*>(resource0)->GetSRV();

	isGpuBuffer = dynamic_cast<GPUBuffer*>(resource1) != nullptr;

	const D3D12_CPU_DESCRIPTOR_HANDLE& uavHandle1 = isGpuBuffer ? static_cast<GPUBuffer*>(resource1)->GetUAV() : static_cast<ColorBuffer*>(resource1)->GetUAV(0);
	const D3D12_CPU_DESCRIPTOR_HANDLE& srvHandle1 = isGpuBuffer ? static_cast<GPUBuffer*>(resource1)->GetSRV() : static_cast<ColorBuffer*>(resource1)->GetSRV();

	Graphics::s_device->Get()->CopyDescriptorsSimple(
		1,
		m_uavHandle + descriptorStart0,
		uavHandle1,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
	);

	Graphics::s_device->Get()->CopyDescriptorsSimple(
		1,
		m_srvHandle + descriptorStart0,
		srvHandle1,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
	);

	Graphics::s_device->Get()->CopyDescriptorsSimple(
		1,
		m_uavHandle + descriptorStart1,
		uavHandle0,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
	);

	Graphics::s_device->Get()->CopyDescriptorsSimple(
		1,
		m_srvHandle + descriptorStart1,
		srvHandle0,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
	);
	
}

void BufferManager::ZeroBuffer(DX12Lib::CommandContext& context, UINT index)
{
	assert(index < m_buffers.size());

	GPUBuffer* buffer = dynamic_cast<GPUBuffer*>(m_buffers[index].get());

	if (buffer == nullptr)
		return;

	UploadBuffer uploadBuffer;
	uploadBuffer.Create(buffer->GetElementCount() * buffer->GetElementSize());

	void* mappedData = uploadBuffer.Map();

	memset(mappedData, 0, buffer->GetElementCount() * buffer->GetElementSize());



	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(*m_buffers[index], D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(m_buffers[index]->Get(), uploadBuffer.Get());

	context.Flush(true);

	uploadBuffer.Unmap();
}

void BufferManager::TransitionAll(DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flusBarriers)
{
	for (auto buffer : m_buffers)
	{
		context.TransitionResource(*buffer, newState);
	}

	if (flusBarriers)
	{
		context.FlushResourceBarriers();
	}
}

void BufferManager::TransitionBuffer(UINT buffIndex, DX12Lib::CommandContext& context, D3D12_RESOURCE_STATES newState, bool flushBarriers)
{
	if (buffIndex >= m_buffers.size())
	{
		__debugbreak();
	}
	//assert(buffIndex < m_buffers.size());

	context.TransitionResource(*m_buffers[buffIndex], newState, flushBarriers);
}

DX12Lib::GPUBuffer& BufferManager::GetBuffer(UINT index)
{
	assert(index < m_buffers.size());

	GPUBuffer* buffer = dynamic_cast<GPUBuffer*>(m_buffers[index].get());

	assert(buffer != nullptr);

	return *buffer;
}

void BufferManager::MoveDataTo(BufferManager& other)
{
	other.m_buffers.clear();

	for (UINT i = 0; i < m_buffers.size(); i++)
	{
		other.m_buffers.push_back(m_buffers[i]);
	}

	other.m_uavHandle = this->m_uavHandle;
	other.m_srvHandle = this->m_srvHandle;
}


