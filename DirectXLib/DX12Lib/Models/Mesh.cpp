#include "DX12Lib/pch.h"

#include "Mesh.h"

using namespace DX12Lib;

void Mesh::Draw(ID3D12GraphicsCommandList* cmdList)
{
	assert(cmdList != nullptr && "CommandList is null");

	cmdList->IASetPrimitiveTopology(m_primitiveTopology);
	cmdList->DrawIndexedInstanced(m_numIndices, 1, m_indexStart, m_vertexStart, 0);
}
