#include "VoxelUtils.hlsli"

struct AABBInfo
{
    float3 Min;
    uint StartIndex;
    
    float3 Max;
    uint Count;
};

struct VoxelAABB
{
    float3 Min;
    float3 Max;

};

cbuffer cbAABBCommons : register(b0)
{
    uint3 GridDimension;
    uint ClusterCount;
}


StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space1);
StructuredBuffer<uint> gNextVoxelInClusterBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelAssignmentBuffer : register(t2, space1);

RWStructuredBuffer<VoxelAABB> gVoxelAABBBuffer : register(u0, space0);
RWStructuredBuffer<AABBInfo> gClusterAABBInfoBuffer : register(u1, space0);
RWStructuredBuffer<uint> gAABBCounter : register(u2, space0);


[numthreads(256, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{    
    // Number of assigned voxels
    if (DTid.x >= ClusterCount)
        return;
        
    uint voxelCount = gClusterDataBuffer[DTid.x].VoxelCount;
        
    uint originalValue = 0;
    InterlockedAdd(gAABBCounter[0], voxelCount, originalValue);
    
    AABBInfo clusterInfo;
    clusterInfo.StartIndex = originalValue;
    clusterInfo.Count = voxelCount;
        
    uint voxelIndex = gClusterDataBuffer[DTid.x].FirstDataIndex;
        
    float3 offset = float3(0.5f, 0.5f, 0.5f);
    
    float3 minCluster = float3(UINT_MAX, UINT_MAX, UINT_MAX);
    float3 maxCluster = float3(-512.0f, -512.0f, -512.0f);
    
    uint nVoxel = 0;
    while (voxelIndex != UINT_MAX)
    {
        float3 voxelCoord = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], GridDimension));
        
        VoxelAABB voxelAABB;
        voxelAABB.Min = voxelCoord - offset;
        voxelAABB.Max = voxelCoord + offset;
        
        gVoxelAABBBuffer[originalValue + nVoxel] = voxelAABB;
            
        minCluster = min(minCluster, voxelAABB.Min);
        maxCluster = max(maxCluster, voxelAABB.Max);
        
        nVoxel += 1;
        voxelIndex = gNextVoxelInClusterBuffer[voxelIndex];
    }
    
    clusterInfo.Min = minCluster;
    clusterInfo.Max = maxCluster;
    
    gClusterAABBInfoBuffer[DTid.x] = clusterInfo;

}