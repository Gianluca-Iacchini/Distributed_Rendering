#ifndef TECHNIQUEHLSLCOMPAT_H
#define TECHNIQUEHLSLCOMPAT_H

#ifdef HLSL
#include "HlslCompat.h"
#define XMUINT2 uint2
#define XMUINT3 uint3 /* I don't know why, but this won't work if i define it in the HlslCompat.h file */
#else

#include <directx/d3d12.h>
#include <DirectXMath.h>
using namespace DirectX;

#endif // HLSL

struct ConstantBufferRTShadows
{
	XMUINT3 GridDimension;
	float pad0;

	XMUINT2 ShadowTexDimensions;
	float pad1;
	float pad2;
};

struct RTSceneVisibility
{
	XMUINT3 DispatchSize;
	UINT NumberOfFaces;

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

struct AABBInfo
{
	UINT ClusterStartIndex;
	UINT ClusterElementCount;
	float pad0;
	float pad1;
};

struct AABB
{
	XMFLOAT3 Min;
	XMFLOAT3 Max;
};



#endif // TECHNIQUEHLSLCOMPAT_H