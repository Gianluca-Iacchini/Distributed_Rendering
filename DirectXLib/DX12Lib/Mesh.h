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


	D3D_PRIMITIVE_TOPOLOGY m_primitiveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	UINT m_numIndices = 0;


	UINT m_vertexBufferStride = 0;
	UINT m_vertexBufferByteSize = 0;
	UINT m_vertexStart = 0;
	DXGI_FORMAT m_indexBufferFormat = DXGI_FORMAT_R16_UINT;
	UINT m_indexBufferByteSize = 0;
	UINT m_indexStart = 0;;

private:
	DirectX::SharedGraphicsResource m_vertexBuffer;
	DirectX::SharedGraphicsResource m_indexBuffer;
	//std::vector<D3D12_INPUT_ELEMENT_DESC> m_inputLayout;
};

