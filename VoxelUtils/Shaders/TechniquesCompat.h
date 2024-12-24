#ifndef TECHNIQUEHLSLCOMPAT_H
#define TECHNIQUEHLSLCOMPAT_H

#ifdef HLSL
#include "HlslCompat.h"
#define XMFLOAT4X4 float4x4
#define XMUINT2 uint2
#define XMUINT3 uint3 /* I don't know why, but this won't work if i define it in the HlslCompat.h file */

// AABB is already defined in MathHelper.h so we don't need to redefine it for the c++ side;
// However we need to define it for the hlsl files.
struct AABB
{
	float3 Min;
	float3 Max;
};

#else

#include <directx/d3d12.h>
#include <DirectXMath.h>
using namespace DirectX;

#endif // HLSL

struct ConstantBufferVoxelCommons
{
	XMUINT3 voxelTextureDimensions;
	float totalTime;

	XMFLOAT3 voxelCellSize;
	float deltaTime;

	XMFLOAT3 invVoxelTextureDimensions;
	UINT StoreData;

	XMFLOAT3 invVoxelCellSize;
	UINT VoxelCount;

	XMFLOAT3 SceneAABBMin;
	UINT ClusterCount;

	XMFLOAT3 SceneAABBMax;
	float pad3;

	XMFLOAT4X4 VoxelToWorld;
	XMFLOAT4X4 WorldToVoxel;
};

struct ConstantBufferCompactBuffer
{
	UINT CurrentPhase;
	UINT CurrentStep;
	UINT CompactBufferSize;
	UINT ElementsPerThread;

	UINT NumElementsSweepDown;
	UINT NumElementsBase;
	UINT NumElementsLevel0;
	UINT NumElementsLevel1;

	UINT NumElementsLevel2;
	UINT NumElementsLevel3;
	float pad0;
	float pad1;

	XMUINT3 VoxelGridSize;
	float pad2;
};

struct ConstantBufferClusterizeBuffer
{
	UINT CurrentPhase;
	UINT VoxelCount;
	UINT K;
	float m;

	XMUINT3 VoxelTextureDimensions;
	UINT S;

	XMUINT3 TileGridDimension;
	UINT FirstClusterSet;

	XMUINT3 CurrentTileUpdate;
	UINT UnassignedOnlyPass;
};

struct ConstantBufferRTShadows
{
	XMFLOAT3 LightDirection;
	UINT FaceCount;

	UINT CurrentStep;
	UINT VoxelCount;
	UINT ClusterCount;
	UINT FrameCount;
};

struct RTSceneVisibility
{
	float pad0;
	float pad1;
	float pad2;
	UINT FaceCount;

	XMUINT3 GridDimension;
	UINT CurrentPhase;

	UINT BlasCount;
	UINT GeometryCount;
	float Rand1;
	float Rand2;

};

struct ConstantBufferFaceCount
{
	XMUINT3 GridDimension;
	UINT CurrentPhase;

	UINT VoxelCount;
	XMUINT3 pad1;
};

struct ConstantBufferAABBGeneration
{
	XMUINT3 GridDimension;
	UINT ClusterCount;
};

struct ConstantBufferFacePenalty
{
	UINT FaceCount;
	UINT VoxelCount;
	float pad0;
	float pad1;
};

struct ConstantBufferFrustumCulling
{
	XMFLOAT4 FrustumPlanes[6];

	XMFLOAT3 CameraPosition;
	UINT AABBGroupCount;

	UINT CurrentStep;
	UINT VoxelCount;
	UINT FaceCount;
	UINT ResetRadianceBuffers;
};

struct ConstantBufferIndirectLightTransport
{
	XMFLOAT3 LightDirection;
	UINT VoxelCount;

	XMFLOAT3 LightPosition;
	float LightIntensity;

	XMFLOAT3 EyePosition;
	UINT DispatchNumber;
};

struct ConstantBufferClearBuffers
{
	UINT ValueCount0;
	UINT ValueCount1;
	UINT ValueCount2;
	UINT ValueCount3;
};

struct ConstantBufferLitVoxels
{

};

struct ClusterAABBInfo
{
	XMFLOAT3 Min;
	UINT ClusterStartIndex;
	XMFLOAT3 Max;
	UINT ClusterElementCount;
};

struct ConstantBufferLerpRadiance
{
	UINT CurrentPhase;
	float accumulatedTime;
	float maxTime;
	UINT FaceCount;
};

struct ClusterData
{
	XMUINT3 Center;
	UINT VoxelCount;

	XMFLOAT3 Normal;
	UINT FirstDataIndex;

	XMFLOAT3 Color;
	UINT FragmentCount;

	XMFLOAT3 MinAABB;
	UINT NeighbourCount;

	XMFLOAT3 MaxAABB;
	float pad0;

	UINT ClusterNeighbours[64];
};

struct ConstantBufferComputeNeighbour
{
	UINT ClusterCount;
	UINT ElementsPerThread;
	UINT TotalComputationCount;
	float pad1;
};

struct ConstantBufferGaussianFilter
{
	XMFLOAT3 EyePosition;
	UINT BlockNum;

	UINT CurrentPhase;
	UINT KernelSize;
	UINT VoxelCount;
	UINT FaceCount;
};

struct ConstantBufferRadianceFromNetwork
{
	UINT ShouldReset;
	UINT FaceCount;
	UINT ReceivedFaceCount;
	UINT pad0;
};

#endif // TECHNIQUEHLSLCOMPAT_H