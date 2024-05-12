#include "DX12Lib/pch.h"
#include "UploadBuffer.h"

using namespace DX12Lib;

template<typename T>
inline UploadBuffer<T>::UploadBuffer(Device& device, UINT elementCount, bool isConstantBuffer) 
	: m_device(device), m_isConstantBuffer(isConstantBuffer), m_elementCount(elementCount)
{
	m_elementByteSize = sizeof(T);

	if (isConstantBuffer)
	{
		m_elementByteSize = Utils::AlignAtBytes(m_elementByteSize, CONSTANT_BUFFER_SIZE);
	}
	
	Recreate();
}

template<typename T>
void UploadBuffer<T>::Recreate()
{
	ThrowIfFailed(m_device.GetComPtr()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(m_elementByteSize * m_elementCount),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(m_resource.GetAddressOf())));

	ThrowIfFailed(m_resource->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedData)));
}

template<typename T>
UploadBuffer<T>::~UploadBuffer()
{
	if (m_resource != nullptr)
	{
		m_resource->Unmap(0, nullptr);
	}
	m_mappedData = nullptr;
	
	m_resource = nullptr;
}

template<typename T>
void UploadBuffer<T>::CopyData(int elementIndex, const T& data)
{
	memcpy(&m_mappedData[elementIndex * m_elementByteSize], &data, sizeof(T));
}
