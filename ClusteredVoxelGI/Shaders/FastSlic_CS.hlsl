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

// Using groupshared memory to store number of voxels and clusters in the current tile
// Each thread group represent a tile
groupshared uint smVoxelInTileCount;
groupshared uint smClusterInTileCount;

StructuredBuffer<FragmentData> gFragmentBuffer : register(t0, space0);
StructuredBuffer<uint> gNextIndexBuffer : register(t1, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space0);
RWStructuredBuffer<uint> gNextVoxelBuffer: register(u1, space0);
RWStructuredBuffer<uint> gClusterAssignmentBuffer : register(u2, space0);
RWStructuredBuffer<uint> gClusterDistanceBuffer : register(u3, space0);
RWTexture3D<uint> gTileBuffer : register(u4, space0);
RWStructuredBuffer<uint> gNextCluster : register(u5, space0);
RWStructuredBuffer<uint> gClusterCounterBuffer : register(u6, space0);

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank > 0)
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
    // This unnecessary else statement is here because the HLSL compiler doesn't like early returns. This is a workaround to avoid
    // Getting compiler warnings
    else
    {
        return result;
    }
}

[numthreads(8, 8, 8)]
void CS(uint3 TileID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{
  
    uint threadLinearIndex = TileID.z * (TileDimension.x * TileDimension.y) + TileID.y * TileDimension.x + TileID.x;
    threadLinearIndex = threadLinearIndex * 8 * 8 * 8 + GroupThreadIndex;
   
    // Each group represents a tile, but each group has 8x8x8 threads.
    // A tile has a dimension of 2S X 2S X 2S, therefore each thread is responsible for (2S / 8) X (2S / 8) X (2S / 8) voxels
    uint ThreadTilePiece = ceil(S / 4);
    
    // Coord of the first voxel of this tile
    uint3 FirstVoxelInTile = TileID * (2 * S);
    // Coord of the first voxel of the thread for this tile
    uint3 TileStart = FirstVoxelInTile + (GroupThreadID * ThreadTilePiece);
    
    uint3 TileEnd = min(TileStart + ThreadTilePiece, FirstVoxelInTile + uint3(2 * S, 2 * S, 2 * S));
    
    if (CurrentPhase == 0)
    {
        for (uint i = TileStart.x; i < TileEnd.x; i++)
        {
            for (uint j = TileStart.y; j < TileEnd.y; j++)
            {
                for (uint k = TileStart.z; k < TileEnd.z; k++)
                {
                    uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(i, j, k), GridDimension);
                    InterlockedAdd(smVoxelInTileCount, hashedPositionIndex.y);
                }
            }
        }
        
        
        GroupMemoryBarrierWithGroupSync();
        
        if (GroupThreadIndex > 0)
        {
            return;
        }
        
        uint numberOfClusterInTile = ceil(K * ((float) smVoxelInTileCount / VoxelCount));
        float clusterOffset = (2.0f * S) / numberOfClusterInTile;
        
        uint tileClusterGrid = ceil(pow(numberOfClusterInTile, 1 / 3.0f));
        
        ClusterData data;
        data.Normal = float3(0.0f, 0.0f, 0.0f);
        data.VoxelCount = 0;
        data.pad0 = 0;
        
        uint originalValue = 0;
        
        for (uint nCluster = 0; nCluster < numberOfClusterInTile; nCluster++)
        {
            float3 pos = GetVoxelPosition(nCluster, uint3(tileClusterGrid, tileClusterGrid, tileClusterGrid));

            pos = FirstVoxelInTile + (pos + 0.5f) * clusterOffset;
            
            data.Center = pos;
            
            InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
            
            if (originalValue >= K)
                return;
            
            gClusterDataBuffer[originalValue] = data;
        }
        
    }
    else if (CurrentPhase == 1)
    {
        uint voxelsPerThread = ceil(VoxelCount / (TileDimension.x * TileDimension.y * TileDimension.z * 8 * 8 * 8));
        uint clusterPerThread = ceil(K / (TileDimension.x * TileDimension.y * TileDimension.z * 8 * 8 * 8));
       
        uint initialindex = threadLinearIndex * voxelsPerThread;
        uint finalindex = min(initialindex + voxelsPerThread, VoxelCount);
        
        for (uint i = initialindex; i < finalindex; i++)
        {
            gClusterAssignmentBuffer[i] = UINT_MAX;
            gClusterDistanceBuffer[i] = UINT_MAX;
        }
    }
}