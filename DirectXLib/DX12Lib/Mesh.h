#pragma once
#include "Helpers.h"
#include "GraphicsMemory.h"
#include "VertexTypes.h"

class Mesh
{
	friend class Model;

public:
	Mesh() {};
	~Mesh() {};

	D3D12_VERTEX_BUFFER_VIEW VertexBufferView() const
	{
		D3D12_VERTEX_BUFFER_VIEW vbv;
		vbv.BufferLocation = m_vertexBufferResource->GetGPUVirtualAddress();
		vbv.StrideInBytes = m_vertexBufferStride;
		vbv.SizeInBytes = m_vertexBufferByteSize;

		return vbv;
	}

	D3D12_INDEX_BUFFER_VIEW IndexBufferView() const
	{
		D3D12_INDEX_BUFFER_VIEW ibv;
		ibv.BufferLocation = m_indexBufferResource->GetGPUVirtualAddress();
		ibv.Format = m_indexBufferFormat;
		ibv.SizeInBytes = m_indexBufferByteSize;

		return ibv;
	}

	D3D_PRIMITIVE_TOPOLOGY m_primitiveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	UINT m_numIndices = 0;

private:
	UINT m_vertexBufferStride;
	UINT m_vertexBufferByteSize;
	UINT m_vertexStart;
	DXGI_FORMAT m_indexBufferFormat;
	UINT m_indexBufferByteSize;
	UINT m_indexStart;

	DirectX::SharedGraphicsResource m_vertexBuffer;
	DirectX::SharedGraphicsResource m_indexBuffer;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;
	std::vector<D3D12_INPUT_ELEMENT_DESC> m_inputLayout;
};

