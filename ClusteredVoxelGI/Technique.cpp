#include "Technique.h"
#include "DX12Lib/Commons/Renderer.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

CVGI::BufferManager& CVGI::TechniqueData::GetBufferManager(std::wstring name)
{
	auto it = m_bufferManagers.find(name);
	assert(it != m_bufferManagers.end() && "Buffer Manager not found");
	return *(it->second);
}

void CVGI::TechniqueData::SetVoxelGridSize(DirectX::XMUINT3 size)
{
	m_cbVoxelCommons.voxelTextureDimensions = size;
	m_cbVoxelCommons.invVoxelTextureDimensions = DirectX::XMFLOAT3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z);
	VoxelGridSize = size;
}

void CVGI::TechniqueData::SetVoxelCellSize(DirectX::XMFLOAT3 size)
{
	m_cbVoxelCommons.voxelCellSize = size;
	m_cbVoxelCommons.invVoxelCellSize = DirectX::XMFLOAT3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z);
	VoxelCellSize = size;
}

void CVGI::TechniqueData::SetSceneAABB(DX12Lib::AABB aabb)
{
	m_cbVoxelCommons.SceneAABBMin = aabb.Min;
	m_cbVoxelCommons.SceneAABBMax = aabb.Max;
	SceneAABB = aabb;
}

DirectX::GraphicsResource& CVGI::TechniqueData::GetVoxelCommonsResource()
{
	if (m_cbVoxelCommonsResource.Size() == 0)
	{
		BuildMatrices();
		m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);
	}


	return m_cbVoxelCommonsResource;
}

void CVGI::TechniqueData::BuildMatrices()
{
	DirectX:XMMATRIX voxelToWorld = BuildVoxelToWorldMatrix();
	DirectX::XMStoreFloat4x4(&m_cbVoxelCommons.VoxelToWorld, DirectX::XMMatrixTranspose(voxelToWorld));
	DirectX::XMMATRIX invMatrix = DirectX::XMMatrixInverse(nullptr, voxelToWorld);
	DirectX::XMStoreFloat4x4(&m_cbVoxelCommons.WorldToVoxel, DirectX::XMMatrixTranspose(invMatrix));
}

DirectX::XMMATRIX CVGI::TechniqueData::BuildVoxelToWorldMatrix()
{
	DirectX::XMMATRIX normalizeMatrix = DirectX::XMMatrixScaling(1.0f / VoxelGridSize.x,
																 1.0f / VoxelGridSize.y, 
																 1.0f / VoxelGridSize.z);

	DirectX::XMFLOAT3 extents = DirectX::XMFLOAT3(SceneAABB.Max.x - SceneAABB.Min.x,
												  SceneAABB.Max.y - SceneAABB.Min.y,
												  SceneAABB.Max.z - SceneAABB.Min.z);

	DirectX::XMMATRIX scaleMatrix = DirectX::XMMatrixScaling(extents.x,
															 extents.y,
															 extents.z);

	DirectX::XMMATRIX translateMatrix = DirectX::XMMatrixTranslation(extents.x / VoxelGridSize.x * 0.5f,
																	 extents.y / VoxelGridSize.y * 0.5f,
																	 extents.z / VoxelGridSize.z * 0.5f);

	DirectX::XMMATRIX translateToOrigin = DirectX::XMMatrixTranslation(SceneAABB.Min.x,
		SceneAABB.Min.y,
		SceneAABB.Min.z);

	DirectX::XMMATRIX voxelToWorld = normalizeMatrix * scaleMatrix * translateMatrix * translateToOrigin;

	return voxelToWorld;
}
