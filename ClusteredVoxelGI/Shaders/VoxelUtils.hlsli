#include "..\..\DirectXLib\DX12Lib\DXWrapper\Shaders\Common.hlsli"

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
    float3 gridDimension;
    float totalTimew;
    
    float3 cellSize;
    float deltaTime;
    
    float3 inverseGridDimension;
    float pad0;
    
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

struct VoxelData
{
    float3 position;
    uint numberOfFragments;

    uint4 color;
	
    float3 normal;
    uint3 voxelCoord;
};


