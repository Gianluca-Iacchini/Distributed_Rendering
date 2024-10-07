#include "..\..\DirectXLib\DX12Lib\DXWrapper\Shaders\Common.hlsli"


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

