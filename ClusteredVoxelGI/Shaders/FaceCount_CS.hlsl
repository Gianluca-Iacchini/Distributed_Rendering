#include "VoxelUtils.hlsli"

cbuffer cbFaceCount : register(b0)
{   
    uint3 GridDimension;
    uint CurrentPhase;
    
    uint VoxelCount;
    uint3 pad0;
}

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);


RWStructuredBuffer<uint2> gVoxelFaceDataBuffer : register(u0, space0);
RWStructuredBuffer<uint> gVoxelFaceCountBuffer : register(u1, space0);

uint2 FindHash(uint index, uint rank, uint hashedPosition)
{
    uint2 result = uint2(0, 0);
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

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank > 0 && any(coord < GridDimension))
    {
        return FindHash(index, rank, hashedPosition);
    }
    // This unnecessary else statement is here because the HLSL compiler doesn't like early returns. This is a workaround to avoid
    // Getting compiler warnings
    else
    {
        return result;
    }
}

[numthreads(256, 1, 1)]
void CS( uint3 ThreadId : SV_DispatchThreadID )
{
    // Using 1D Groups with 1D Threads
    if (ThreadId.x >= VoxelCount)
        return;
    
    uint voxelIndex = gVoxelHashedCompactBuffer[ThreadId.x];
    

    uint3 voxelCoord = GetVoxelPosition(voxelIndex, GridDimension);
    uint2 result = FindHashedCompactedPositionIndex(voxelCoord, GridDimension);
        
    int3 directions[6] = {
        int3(0, 0, -1),
        int3(0, 0, 1),
        int3(-1, 0, 0),
        int3(1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),


    };
        
    for (uint i = 0; i < 6; i++)
    {
        int3 neighborCoord = int3(voxelCoord) + directions[i];
            
        if (i == 4)
        {
            neighborCoord = int3(voxelCoord) + directions[5];
        }
        else if (i == 5)
        {
            neighborCoord = int3(voxelCoord) + directions[4];
        }
        
        if (any(neighborCoord < 0) || any(neighborCoord >= int3(GridDimension)))
            continue;
            
        uint2 neighborHash = FindHashedCompactedPositionIndex(uint3(neighborCoord), GridDimension);
            
        uint originalValue = 0;
        InterlockedAdd(gVoxelFaceCountBuffer[0], 1 - neighborHash.y, originalValue);
        
        if (CurrentPhase == 1)
        {
            if (neighborHash.y == 0)
            {
                uint2 faceData = uint2(ThreadId.x, i);
                gVoxelFaceDataBuffer[originalValue] = faceData;
            }
        }
        
    }

    
}