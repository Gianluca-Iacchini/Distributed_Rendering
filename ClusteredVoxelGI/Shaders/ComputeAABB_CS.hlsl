#define HLSL 1
#include "VoxelUtils.hlsli"
#include "TechniquesCompat.h"


ConstantBuffer<ConstantBufferAABBGeneration> cbAABB : register(b0);


StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space1);
StructuredBuffer<uint> gNextVoxelInClusterBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelAssignmentBuffer : register(t2, space1);

RWStructuredBuffer<AABB> gVoxelAABBBuffer : register(u0, space0);
RWStructuredBuffer<AABBInfo> gClusterAABBInfoBuffer : register(u1, space0);
// Map from aabbVoxelIndices to gVoxelIndicesCompactBuffer
RWStructuredBuffer<uint> gAABBVoxelIndices : register(u2, space0);
RWStructuredBuffer<uint> gAABBCounter : register(u3, space0);
RWStructuredBuffer<uint> gAABBDebug : register(u4, space0);

groupshared uint voxelCount[2] = { 0, 0 };

groupshared uint aabbStart = 0;


uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (any(coord >= cbAABB.GridDimension))
        return result;
    
    if (rank == 0)
        return result;
    
    uint tempHashed;
    uint startIndex = index;
    uint endIndex = index + rank;
    uint currentIndex = (startIndex + endIndex) / 2;

    for (int i = 0; i < int(12); ++i)
    {
        tempHashed = gVoxelHashedCompactBuffer[currentIndex];

        if (tempHashed == hashedPosition)
        {
            return uint2(currentIndex, 1);
        }

        if (tempHashed < hashedPosition)
        {
            startIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
        else
        {
            endIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
    }

    return result;
}


[numthreads(8, 8, 8)]
void CS( uint3 DTid : SV_DispatchThreadID, uint GroupThreadIndex : SV_GroupIndex, uint3 groupIdx : SV_GroupID)
{        
    
    uint2 result = FindHashedCompactedPositionIndex(DTid, cbAABB.GridDimension);
    
    InterlockedAdd(voxelCount[0], result.y);

    GroupMemoryBarrierWithGroupSync();
    
    if (GroupThreadIndex == 0)
    {
        InterlockedAdd(gAABBCounter[0], voxelCount[0], aabbStart);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint originalValue = 0;
    InterlockedAdd(voxelCount[1], result.y, originalValue);
    
    if (result.y  == 1)
    {
        uint3 voxelCoord = DTid;
        voxelCoord.y = voxelCoord.y;
    
        AABB voxelAABB;
        voxelAABB.Min = voxelCoord - float3(0.5f, 0.5f, 0.5f);
        voxelAABB.Max = voxelCoord + float3(0.5f, 0.5f, 0.5f);
        gVoxelAABBBuffer[aabbStart + originalValue] = voxelAABB;
        gAABBVoxelIndices[aabbStart + originalValue] = result.x;
    }
    
    if (GroupThreadIndex == 0)
    {
        if (voxelCount[0] > 0)
        {
            uint aabbGroupIdx = 0;
            InterlockedAdd(gAABBDebug[0], 1, aabbGroupIdx);
        
            AABBInfo info = { aabbStart, voxelCount[0], 0.0f, 0.0f };
            gClusterAABBInfoBuffer[aabbGroupIdx] = info;
        }
    }
}