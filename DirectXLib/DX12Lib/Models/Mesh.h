#pragma once

#include "GraphicsMemory.h"
#include "VertexTypes.h"

#include "d3d12.h"

namespace DX12Lib {

	class Material;

	class Mesh
	{
		friend class Model;

	public:
		Mesh() {};
		~Mesh() {};

		void Draw(ID3D12GraphicsCommandList* cmdList);

		D3D_PRIMITIVE_TOPOLOGY m_primitiveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		UINT m_numIndices = 0;
		UINT m_vertexStart = 0;
		UINT m_indexStart = 0;;


		Material* MeshMaterial = nullptr;

	};
}
