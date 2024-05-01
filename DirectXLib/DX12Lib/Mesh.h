#pragma once
#include "Helpers.h"
#include "GraphicsMemory.h"
#include "VertexTypes.h"
#include "Texture.h"
#include "DX12Lib/DescriptorHeap.h"

class Mesh
{
	friend class Model;

public:
	Mesh() {};
	~Mesh() {};

	void AddTexture(SharedTexture texture, DescriptorHandle textureHandle) { m_meshTextures.push_back(texture); m_textureHandle.push_back(textureHandle); }

	D3D_PRIMITIVE_TOPOLOGY m_primitiveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	UINT m_numIndices = 0;
	UINT m_vertexStart = 0;
	UINT m_indexStart = 0;;

	UINT m_materialIndex = 0;

private:
	std::vector<SharedTexture> m_meshTextures;
	std::vector<DescriptorHandle> m_textureHandle;
};

