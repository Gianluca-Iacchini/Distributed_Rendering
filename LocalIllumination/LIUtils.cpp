#include "LIUtils.h"

using namespace LI;
using namespace DX12Lib;

DX12Lib::ConstantBufferVoxelTransform LI::LIUtils::BuildVoxelCommons(DX12Lib::AABB SceneAABB, DirectX::XMUINT3 VoxelGridSize)
{
	DirectX::XMFLOAT3 sceneExtents = DirectX::XMFLOAT3(SceneAABB.Max.x - SceneAABB.Min.x,
		SceneAABB.Max.y - SceneAABB.Min.y,
		SceneAABB.Max.z - SceneAABB.Min.z);


	DirectX::XMMATRIX transformOrigin = DirectX::XMMatrixTranslation(sceneExtents.x / 2.0f,
		sceneExtents.y / 2.0f,
		sceneExtents.z / 2.0f);

	DirectX::XMFLOAT3 VoxelCellSize = DirectX::XMFLOAT3(sceneExtents.x / VoxelGridSize.x,
		sceneExtents.y / VoxelGridSize.y,
		sceneExtents.z / VoxelGridSize.z);

	DirectX::XMMATRIX scaleMatrix = DirectX::XMMatrixScaling(
		1.0f / VoxelCellSize.x,
		1.0f / VoxelCellSize.y,
		1.0f / VoxelCellSize.z);

	ConstantBufferVoxelTransform voxelTransform;

	// Technically we would need to apply another transformation first which reverses the transformations caused by the orthographic
	// Camera during the voxelization step. However since the orthographic camera uses the same extents as the scene and the camera
	// Is placed at the center of the scene the resulting transformation is the identity matrix.
	DirectX::XMMATRIX worldToVoxel = transformOrigin * scaleMatrix;
	DirectX::XMStoreFloat4x4(&voxelTransform.worldToVoxel, DirectX::XMMatrixTranspose(worldToVoxel));
	DirectX::XMMATRIX invMatrix = DirectX::XMMatrixInverse(nullptr, worldToVoxel);
	DirectX::XMStoreFloat4x4(&voxelTransform.voxelToWorld, DirectX::XMMatrixTranspose(invMatrix));

	voxelTransform.voxelGridSize = VoxelGridSize;
	//voxelTransform.voxelCellSize = VoxelCellSize;

	return voxelTransform;
}
