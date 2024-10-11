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
		m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);

	return m_cbVoxelCommonsResource;
}
