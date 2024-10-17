#ifndef VOXEL_UTILS
#define VOXEL_UTILS

#include "..\..\DirectXLib\DX12Lib\DXWrapper\Shaders\Common.hlsli"
#define HLSL
#include "TechniquesCompat.h"

static const unsigned int UINT_MAX = 0xffffffff;
static const float cos30 = 0.81915204428f;
static const float cos25 = 0.9f;

struct VertexOutVoxel
{
    float4 PosH : SV_POSITION;
    float3 PosW : POSITION0;
    uint ProjAxis : AXIS;
    float3 NormalW : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VoxelCommons
{
    uint3 gridDimension;
    float totalTime;
    
    float3 cellSize;
    float deltaTime;
    
    float3 inverseGridDimension;
    uint storeData;
    
    float3 inverseCellSize;
    float pad1;
};


struct VoxelCamera
{
    float4x4 xAxisView;
    float4x4 yAxisView;
    float4x4 zAxisView;
    float4x4 orthoProj;
    
    float4x4 xAxisViewProj;
    float4x4 yAxisViewProj;
    float4x4 zAxisViewProj;
    
    float nearPlane;
    float farPlane;
    float pad0;
    float pad1;
};

struct FragmentData
{
    float3 position;
    float pad0;

    float4 color;
	
    float3 normal;
    uint voxelLinearCoord;
};

struct ClusterData
{
    uint3 Center;
    uint VoxelCount;
    
    float3 Normal;
    uint FirstDataIndex;
};

uint3 GetVoxelPosition(uint voxelLinearCoord, uint3 gridDimension)
{
    uint3 voxelPosition;
    voxelPosition.x = voxelLinearCoord % gridDimension.x;
    voxelPosition.y = (voxelLinearCoord / gridDimension.x) % gridDimension.y;
    voxelPosition.z = voxelLinearCoord / (gridDimension.x * gridDimension.y); 
    return voxelPosition;
}

uint GetLinearCoord(uint3 coord3, uint3 gridDimension)
{
    return  coord3.x +
            coord3.y * gridDimension.x +
            coord3.z * gridDimension.x * gridDimension.y;
}

// Returns true if the voxel coordinates are withing the bounds after adding the offset vector
bool IsWithinBounds(uint3 coord, int3 offset, uint3 gridDimension)
{
    // Check for underflow
    if (any(coord < uint3(-min(int3(0, 0, 0), offset))))
        return false;
    
    uint3 result = coord + max(int3(0, 0, 0), offset);

    return all(result < gridDimension);
}

bool IsVoxelPresent(uint voxelLinearCoord, ByteAddressBuffer voxelPresenceBuffer)
{
    uint index = voxelLinearCoord >> 5u;
    uint bit = voxelLinearCoord & 31u;
    
    // ByteAddressBuffer operations wants multiple of 4 bytes
    uint value = voxelPresenceBuffer.Load(index * 4);
    
    return (value & (1u << bit)) != 0;
}

bool IsVoxelPresent(uint voxelLinearCoord, RWByteAddressBuffer voxelPresenceBuffer)
{
    uint index = voxelLinearCoord >> 5u;
    uint bit = voxelLinearCoord & 31u;
    
    // ByteAddressBuffer operations wants multiple of 4 bytes
    uint value = voxelPresenceBuffer.Load(index * 4);
    
    return (value & (1u << bit)) != 0;
}

bool IsVoxelPresent(uint3 voxelCoord, uint3 gridDimension, ByteAddressBuffer voxelPresenceBuffer)
{
    uint voxelLinearCoord = GetLinearCoord(voxelCoord, gridDimension);
    return IsVoxelPresent(voxelLinearCoord, voxelPresenceBuffer);
}

bool SetVoxelPresence(uint voxelLinearCoord, RWByteAddressBuffer voxelPresenceBuffer)
{
    uint index = voxelLinearCoord >> 5u;
    uint bit = voxelLinearCoord & 31u;
    
    index = index * 4;
    
    uint outVal = 0;
    
    voxelPresenceBuffer.InterlockedOr(index, (1u << bit), outVal);
    
    return (outVal & (1u << bit)) != 0;
}


bool SetVoxelPresence(uint3 voxelCoord, uint3 gridDimension, RWByteAddressBuffer voxelPresenceBuffer)
{
    uint voxelLinearCoord = GetLinearCoord(voxelCoord, gridDimension);
    return SetVoxelPresence(voxelLinearCoord, voxelPresenceBuffer);
}


#endif // VOXEL_UTILS