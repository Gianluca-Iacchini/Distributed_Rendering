
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
RWStructuredBuffer<ClusterAABBInfo> gClusterAABBInfoBuffer : register(u1, space0);
// Map from aabbVoxelIndices to gVoxelIndicesCompactBuffer.
RWStructuredBuffer<uint> gAABBVoxelIndices : register(u2, space0);
// Buffer used to create offsets for aabb indices.
RWStructuredBuffer<uint> gAABBCounter : register(u3, space0);
// Buffer used to return the number of aabb grids.
RWStructuredBuffer<uint> gAABBGridCount : register(u4, space0);

// Groupshared memory used to store the aabb at the correct indices. We only care that aabbs from the same
// group are stored contiguously so we can build the acceleration structures later.

// We use two counters per thread group to avoid performing two passes.
// Counter at position 0 is used to find the offset of this aabb group.
// Counter at position 1 is used to find the offset of the single aabb withing the group.
groupshared uint voxelCount[2] = { 0, 0 };

// Store the start index of the aabb group.
groupshared uint aabbStart = 0;

groupshared uint3 minAABB;
groupshared uint3 maxAABB;

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (all(coord < cbAABB.GridDimension) && rank > 0)
    {
        
    

    
        uint tempHashed;
        uint startIndex = index;
        uint endIndex = index + rank;
        uint currentIndex = (startIndex + endIndex) / 2;

        for (int i = 0; i < int(12); ++i)
        {
            tempHashed = gVoxelHashedCompactBuffer[currentIndex];

            if (tempHashed == hashedPosition)
            {
                result = uint2(currentIndex, 1);
                break;
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
        minAABB = uint3(cbAABB.GridDimension) + 2;
        maxAABB = uint3(0, 0, 0);
        InterlockedAdd(gAABBCounter[0], voxelCount[0], aabbStart);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint originalValue = 0;
    InterlockedAdd(voxelCount[1], result.y, originalValue);
    
    if (result.y  == 1)
    {
        uint3 voxelCoord = DTid;

        InterlockedMin(minAABB.x, voxelCoord.x);
        InterlockedMin(minAABB.y, voxelCoord.y);
        InterlockedMin(minAABB.z, voxelCoord.z);
        
        InterlockedMax(maxAABB.x, voxelCoord.x);
        InterlockedMax(maxAABB.y, voxelCoord.y);
        InterlockedMax(maxAABB.z, voxelCoord.z);
        
        AABB voxelAABB;
        voxelAABB.Min = voxelCoord - float3(0.5f, 0.5f, 0.5f);
        voxelAABB.Max = voxelCoord + float3(0.5f, 0.5f, 0.5f);
        gVoxelAABBBuffer[aabbStart + originalValue] = voxelAABB;
        gAABBVoxelIndices[aabbStart + originalValue] = result.x;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (GroupThreadIndex == 0)
    {
        if (voxelCount[0] > 0)
        {
            uint aabbGroupIdx = 0;
            InterlockedAdd(gAABBGridCount[0], 1, aabbGroupIdx);
        
            ClusterAABBInfo info;
            info.Min = float3(minAABB) - float3(0.5f, 0.5f, 0.5f);
            info.ClusterStartIndex = aabbStart;
            info.Max = float3(maxAABB) + float3(0.5f, 0.5f, 0.5f);
            info.ClusterElementCount = voxelCount[0];
            
            
            gClusterAABBInfoBuffer[aabbGroupIdx] = info;
        }
    }
}