#include "VoxelUtils.hlsli"
#include "TechniquesCompat.h"

ConstantBuffer<ConstantBufferFaceCount> cbFaceCount : register(b0);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);

// For each element i: 
// x is the voxel index of the ith face, 
// y is the normal direction of the ith face (0: -Z, 1: +Z, 2: -X, 3: +X, 4: -Y, 5: +Y)
// All the faces of the same voxel are stored in sequence
RWStructuredBuffer<uint2> gVoxelFaceDataBuffer : register(u0, space0);
// The element i contains the start index in gVoxelFaceDataBuffer and the number of the faces for the voxel with index i
RWStructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(u1, space0);
// Temp buffer used to store 
RWStructuredBuffer<uint> gVoxelFaceCountBuffer : register(u2, space0);


[numthreads(256, 1, 1)]
void CS( uint3 ThreadId : SV_DispatchThreadID )
{
    // Using 1D Groups with 1D Threads
    if (ThreadId.x >= cbFaceCount.VoxelCount)
        return;
    
    uint voxelIndex = gVoxelHashedCompactBuffer[ThreadId.x];
    

    uint3 voxelCoord = GetVoxelPosition(voxelIndex, cbFaceCount.GridDimension);

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
    uint faceIdx = 0;
    
    [unroll(6)]
    for (uint i = 0; i < 6; i++, faceIdx++)
    {
        if (!IsWithinBounds(voxelCoord, directions[i], cbFaceCount.GridDimension))
            continue;
        
        int3 neighborCoord = int3(voxelCoord) + directions[i];
        
        uint neighbourNotPresent = 1 - (uint) IsVoxelPresent(uint3(neighborCoord), cbFaceCount.GridDimension, gVoxelOccupiedBuffer);
            
        foundDirectionsBitmap |= (neighbourNotPresent << i);
        foundDirectionsCount += neighbourNotPresent;
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