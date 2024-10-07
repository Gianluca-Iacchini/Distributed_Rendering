#define HLSL 1
#include "VoxelUtils.hlsli"
#include "TechniquesCompat.h"

ConstantBuffer<ConstantBufferFaceCount> cbFaceCount : register(b0);
StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

// For each element i: 
// x is the voxel index of the ith face, 
// y is the normal direction of the ith face (0: -Z, 1: +Z, 2: -X, 3: +X, 4: -Y, 5: +Y)
// All the faces of the same voxel are stored in sequence
RWStructuredBuffer<uint2> gVoxelFaceDataBuffer : register(u0, space0);
// The element i contains the start index in gVoxelFaceDataBuffer and the number of the faces for the voxel with index i
RWStructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(u1, space0);
// Temp buffer used to store 
RWStructuredBuffer<uint> gVoxelFaceCountBuffer : register(u2, space0);


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
    
    if (rank > 0 && any(coord < cbFaceCount.GridDimension))
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
    if (ThreadId.x >= cbFaceCount.VoxelCount)
        return;
    
    uint voxelIndex = gVoxelHashedCompactBuffer[ThreadId.x];
    

    uint3 voxelCoord = GetVoxelPosition(voxelIndex, cbFaceCount.GridDimension);
    uint2 result = FindHashedCompactedPositionIndex(voxelCoord, cbFaceCount.GridDimension);
        
    int3 directions[6] = {
        int3(0, 0, -1),
        int3(0, 0, 1),
        int3(-1, 0, 0),
        int3(1, 0, 0),
        int3(0, -1, 0),
        int3(0, 1, 0),
    };
    
    uint foundDirectionsBitmap = 0;
    uint foundDirectionsCount = 0;
    
    for (uint i = 0; i < 6; i++)
    {
        int3 neighborCoord = int3(voxelCoord) + directions[i];
        
        if (any(neighborCoord < 0) || any(neighborCoord >= int3(cbFaceCount.GridDimension)))
            continue;
            
        uint neighbourPresent = 1 - FindHashedCompactedPositionIndex(uint3(neighborCoord), cbFaceCount.GridDimension).y;
            
        foundDirectionsBitmap |= (neighbourPresent << i);
        foundDirectionsCount += neighbourPresent;
    }

    uint startValue = 0;
    InterlockedAdd(gVoxelFaceCountBuffer[0], foundDirectionsCount, startValue);
    
    if (cbFaceCount.CurrentPhase == 1)
    {

        gVoxelFaceStartCountBuffer[ThreadId.x] = uint2(startValue, foundDirectionsCount);
        uint offset = 0;
        
        for (i = 0; i < 6; i++)
        {
            if ((foundDirectionsBitmap & (1 << i)) != 0)
            {
                uint2 faceData = uint2(ThreadId.x, i);
                gVoxelFaceDataBuffer[startValue + offset] = faceData;
                offset++;
            }
        }
    }
}