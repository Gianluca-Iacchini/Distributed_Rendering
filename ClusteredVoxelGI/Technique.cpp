#include "Technique.h"
#include "DX12Lib/Commons/Renderer.h"
#include "DX12Lib/Commons/ShadowMap.h"
#include "DX12Lib/Scene/SceneCamera.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

CVGI::BufferManager& CVGI::TechniqueData::GetBufferManager(std::wstring name)
{
	auto it = m_bufferManagers.find(name);
	assert(it != m_bufferManagers.end() && "Buffer Manager not found");
	return *(it->second);
}

void CVGI::TechniqueData::SetVoxelCount(UINT32 count)
{
	m_cbVoxelCommons.VoxelCount = count;
	m_voxelCount = count;

	this->m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);
}

void CVGI::TechniqueData::SetClusterCount(UINT32 count)
{
	m_cbVoxelCommons.ClusterCount = count;
	m_clusterCount = count;

	this->m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);
}

void CVGI::TechniqueData::SetCamera(DX12Lib::SceneCamera* camera)
{
	m_sceneCamera = camera;
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
	DirectX:XMMATRIX worldToVoxel = BuildWorldToVoxelMatrix();
	DirectX::XMStoreFloat4x4(&m_cbVoxelCommons.WorldToVoxel, DirectX::XMMatrixTranspose(worldToVoxel));
	DirectX::XMMATRIX invMatrix = DirectX::XMMatrixInverse(nullptr, worldToVoxel);
	DirectX::XMStoreFloat4x4(&m_cbVoxelCommons.VoxelToWorld, DirectX::XMMatrixTranspose(invMatrix));
}

void CVGI::TechniqueData::SetDepthCameraResource(ConstantBufferLight cameraCB)
{
	m_depthCameraResource = Renderer::s_graphicsMemory->AllocateConstant(cameraCB);
}

DirectX::GraphicsResource& CVGI::TechniqueData::GetDepthCameraResource()
{
	return m_depthCameraResource;
}

void CVGI::TechniqueData::SetLightCameraResource(ConstantBufferLight cameraCB)
{
	m_lightCameraResource = Renderer::s_graphicsMemory->AllocateConstant(cameraCB);
}

DirectX::GraphicsResource& CVGI::TechniqueData::GetLightCameraResource()
{
	return m_lightCameraResource;
}


DirectX::XMMATRIX CVGI::TechniqueData::BuildWorldToVoxelMatrix()
{
	DirectX::XMFLOAT3 sceneExtents = DirectX::XMFLOAT3(SceneAABB.Max.x - SceneAABB.Min.x,
													   SceneAABB.Max.y - SceneAABB.Min.y,
													   SceneAABB.Max.z - SceneAABB.Min.z);


	DirectX::XMMATRIX transformOrigin = DirectX::XMMatrixTranslation(sceneExtents.x / 2.0f,
																	 sceneExtents.y / 2.0f,
																	 sceneExtents.z / 2.0f);

	DirectX::XMMATRIX scaleMatrix = DirectX::XMMatrixScaling(
													1.0f / VoxelCellSize.x , 
													1.0f / VoxelCellSize.y , 
													1.0f / VoxelCellSize.z );

	// Technically we would need to apply another transformation first which reverses the transformations caused by the orthographic
	// Camera during the voxelization step. However since the orthographic camera uses the same extents as the scene and the camera
	// Is placed at the center of the scene the resulting transformation is the identity matrix.
	return transformOrigin * scaleMatrix;
}
