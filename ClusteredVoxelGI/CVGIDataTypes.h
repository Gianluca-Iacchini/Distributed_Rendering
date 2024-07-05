#pragma once

#include "DirectXMath.h"

enum class VoxelizeSceneComputeRootParameterSlot
{
	VoxelCommonCBV = 0,
	ObjectCBV = 1,
	VerticesSRV = 2,
	IndicesSRV = 3,
	MaterialSRV,
	MaterialTextureSRV,
	VoxelTextureUAV,
	Count
};

enum class VoxelizeSceneRootParameterSlot
{
	VoxelCommonCBV = 0,
	VoxelCameraCBV = 1,
	ObjectCBV = 2,
	MaterialSRV,
	MaterialTextureSRV,
	VoxelTextureUAV,
	Count
};

enum class DisplayVoxelRootParameterSlot
{
	VoxelCommonCBV = 0,
	CameraCBV = 1,
	VoxelTextureUAV = 2,
	Count
};

__declspec(align(16)) struct ConstantBufferVoxelCommons
{

	DirectX::XMFLOAT3 voxelTextureDimensions = DirectX::XMFLOAT3(128.0f, 128.0f, 128.0f);
	float totalTime = 0.0f;

	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
	float deltaTime = 0.0f;

	DirectX::XMFLOAT3 invVoxelTextureDimensions = DirectX::XMFLOAT3(1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f);
	float pad0 = 0.0f;

	DirectX::XMFLOAT3 invVoxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
	float pad1 = 0.0f;

};