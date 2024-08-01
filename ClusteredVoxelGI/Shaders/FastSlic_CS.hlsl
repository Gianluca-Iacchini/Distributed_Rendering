#include "VoxelUtils.hlsli"

cbuffer SLICCommon : register(b0)
{
    uint CurrentPhase;
    uint VoxelCount;
    uint K; // Number of clusters
    uint m; // Compactness factor
    
    uint3 GridDimension;
    uint S; // Cluster area
    
    uint3 TileDimension;
    float pad0;
}

RWStructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(u0, space0);
RWStructuredBuffer<uint> gVoxelHashedCompactBuffer : register(u1, space0);
RWStructuredBuffer<uint> gNextIndexBuffer : register(u2, space0);
RWStructuredBuffer<FragmentData> gFragmentBuffer : register(u3, space0);


RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space1);
RWStructuredBuffer<uint> gClusterAssignmentBuffer : register(u1, space1);
RWStructuredBuffer<uint> gClusterDistanceBuffer : register(u2, space1);
RWStructuredBuffer<uint> gTileBuffer : register(u3, space1);

[numthreads(128, 1, 1)]
void CS(uint3 GroupID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex)
{

    // Phase 0 is dispatched with 1D thread groups of size gVoxelHashedCompactBuffer.Count / 128 
    if (CurrentPhase == 0)
    {        
        uint linearIndex = GroupID.x * 128 + GroupThreadIndex;
        
        if (linearIndex <= VoxelCount)
        {
            gClusterDistanceBuffer[linearIndex] = UINT_MAX;
            gClusterAssignmentBuffer[linearIndex] = UINT_MAX;
        }
        
        if (linearIndex >= TileDimension.x * TileDimension.y * TileDimension.z)
        {
            return;
        }
        
        uint voxelIndex = gVoxelHashedCompactBuffer[linearIndex];
        uint3 voxelPosition = GetVoxelPosition(voxelIndex, GridDimension);
        
        uint3 tilePosition = uint3(voxelPosition.x / (2 * S), voxelPosition.y / (2 * S), voxelPosition.z / (2 * S));
        uint tileLinearIndex = GetLinearCoord(tilePosition, TileDimension);
        uint outValue = 0;
        
        InterlockedExchange(gTileBuffer[tileLinearIndex], UINT_MAX, outValue);
        
        if (outValue == 0)
        {
            ClusterData data;
            data.VoxelCount = 0;
            data.Normal = float3(0, 0, 0);
            data.pad0 = 0;
            
            float3 tileCenter = float3(tilePosition.x * (2 * S) + S, tilePosition.y * (2 * S) + S, tilePosition.z * (2 * S) + S);
            
            uint clusterOffset = 0;
            
            for (int i = -1; i < 2; i += 2)
            {
                for (int j = -1; j < 2; j += 2)
                {
                    for (int k = -1; k < 2; k += 2)
                    {
                        if (tileLinearIndex + clusterOffset >= K)
                            continue;
                        
                        data.Center = tileCenter + float3(i * S/2, j * S/2, k * S/2);
                        gClusterDataBuffer[tileLinearIndex + clusterOffset] = data;
                        clusterOffset++;
                    }
                }
            }
        }
    }
    else if (CurrentPhase == 1)
    {
        
    }
}