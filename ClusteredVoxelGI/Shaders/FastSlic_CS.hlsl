#include "VoxelUtils.hlsli"

cbuffer SLICCommon : register(b0)
{
    uint CurrentPhase;
    uint VoxelCount;
    uint K; // Number of clusters
    float m; // Compactness factor
    
    uint3 GridDimension;
    uint S; // Cluster area
    
    uint3 TileDimension;
    uint FirstClusterSet;
    
    uint3 TilesToUpdate;
    float pad1;
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
RWStructuredBuffer<float> gClusterDistanceBuffer : register(u3, space0);
RWTexture3D<uint> gTileBuffer : register(u4, space0);
RWStructuredBuffer<uint> gNextCluster : register(u5, space0);
RWStructuredBuffer<uint> gClusterCounterBuffer : register(u6, space0);

static const float cos30 = 0.81915204428f;

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank > 0 || any(coord >= GridDimension))
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

uint GetEmptyVoxelCount(uint3 voxelCoord, int3 direction)
{
    int3 otherDirection = 2 * abs(direction) - 1;
    
    int3 offset = direction;
    offset.x = offset.x == 0 ? 1 : 0;
    offset.y = offset.y == 0 ? 1 : 0;
    offset.z = offset.z == 0 ? 1 : 0;
    
    uint emptyVoxels = 0;
    
    for (int x = otherDirection.x; x < 2; x++)
    {
        for (int y = otherDirection.y; y < 2; y++)
        {
            for (int z = otherDirection.z; z < 2; z++)
            {
                // Careful
                int3 adjacentCoord = int3(voxelCoord) + int3(x, y, z) * offset;
                if (any(adjacentCoord >= int3(GridDimension)) || any(adjacentCoord < 0))
                {
                    emptyVoxels += 1;
                    continue;
                }

                uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(adjacentCoord), GridDimension);
                emptyVoxels += 1 - hashedPositionIndex.y;
            }
        }
    }
    
    return emptyVoxels;
}



[numthreads(8, 8, 8)]
void CS(uint3 TileID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{
  
    uint threadLinearIndex = TileID.z * (TileDimension.x * TileDimension.y) + TileID.y * TileDimension.x + TileID.x;
    threadLinearIndex = threadLinearIndex * 8 * 8 * 8 + GroupThreadIndex;
   
    // Each group represents a tile, but each group has 8x8x8 threads.
    // A tile has a dimension of 2S X 2S X 2S, therefore each thread is responsible for (2S / 8) X (2S / 8) X (2S / 8) voxels
    uint ThreadTilePiece = ceil(S / 4.0f);
    
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
        
        
        gTileBuffer[TileID] = UINT_MAX;
        uint numberOfClusterInTile = 9; //ceil(K * ((float) smVoxelInTileCount / VoxelCount));

        
        uint tileClusterGrid = ceil(pow(numberOfClusterInTile, 1 / 3.0f));
        
        float clusterOffset = (2.0f * S) / tileClusterGrid;
        
        ClusterData data;
        data.VoxelCount = 1;
        data.pad0 = 0.0f;
        data.pad1 = 0.0f;
        data.pad2 = 0.0f;
        data.Normal = float3(1.0f, 0.0f, 0.0f);
        data.NormalAccum = float3(1.0f, 0.0f, 0.0f);
        
        
        uint originalValue = 0;
          
        for (uint nCluster = 0; nCluster < numberOfClusterInTile; nCluster++)
        {
            float3 pos = GetVoxelPosition(nCluster, uint3(tileClusterGrid, tileClusterGrid, tileClusterGrid));

            pos = FirstVoxelInTile + (pos) * clusterOffset + (clusterOffset / 2.0f);
            
            
            
            data.Center = pos;
            data.CenterAccum = pos;
            
            InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
            
            if (originalValue >= K)
                return;
            
            gClusterDataBuffer[originalValue] = data;
            gNextCluster[originalValue] = UINT_MAX;
            
        }
        
    }
    else if (CurrentPhase == 1)
    {
        uint NumberOfThreads = TileDimension.x * TileDimension.y * TileDimension.z * 8 * 8 * 8;
        uint voxelsPerThread = ceil((float) VoxelCount / NumberOfThreads);
        uint clusterPerThread = ceil((float) K / (NumberOfThreads));
       
        uint initialindex = threadLinearIndex * voxelsPerThread;
        uint finalindex = min(initialindex + voxelsPerThread, VoxelCount);
        
        for (uint i = initialindex; i < finalindex; i++)
        {
            gClusterAssignmentBuffer[i] = UINT_MAX;
            gClusterDistanceBuffer[i] = UINT_MAX;
        }
        
        initialindex = threadLinearIndex * clusterPerThread;
        finalindex = min(initialindex + clusterPerThread, K);
        
        for (uint j = initialindex; j < finalindex; j++)
        {
            ClusterData cData = gClusterDataBuffer[j];
            
            cData.Center = cData.CenterAccum / max(cData.VoxelCount, 1);
            cData.Normal = normalize(cData.NormalAccum / max(cData.VoxelCount, 1));
            
            cData.CenterAccum = float3(0.0f, 0.0f, 0.0f);
            cData.NormalAccum = float3(0.0f, 0.0f, 0.0f);
            cData.VoxelCount = 0;
            
            uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
            
            uint newVal = j;
            uint prev = UINT_MAX;
        
            uint currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, newVal, currentValue);
        
            [allow_uav_condition]
            while (currentValue != prev)
            {
                prev = currentValue;
                gNextCluster[j] = currentValue;
                InterlockedCompareExchange(gTileBuffer[tileCoord], prev, newVal, currentValue);
            }
            
            gClusterDataBuffer[j] = cData;
        }

    }
    else if (CurrentPhase == 2)
    {
        // Only tiles which are not adjacent are updated (corner tiles are considered adjacent)
        // So we check if the tile is even or odd for each axis
        
        
        if (any(TileID % 2 != TilesToUpdate))
            return;
        
        uint clusterIndex = gTileBuffer[TileID];
        
        while (clusterIndex != UINT_MAX)
        {
            ClusterData cData = gClusterDataBuffer[clusterIndex];
            
            for (uint i = TileStart.x; i < TileEnd.x; i++)
            {
                for (uint j = TileStart.y; j < TileEnd.y; j++)
                {
                    for (uint k = TileStart.z; k < TileEnd.z; k++)
                    {
                        uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(i, j, k), GridDimension);
                        
                        if (hashedPositionIndex.y == 0)
                            continue;
                        
                        uint3 voxelPos = GetVoxelPosition(gVoxelHashedCompactBuffer[hashedPositionIndex.x], GridDimension);
                        
                        float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
                        

                        
                        // Position distance
                            float d = (m / S) * (
                            abs(cData.Center.x - voxelPos.x) +
                            abs(cData.Center.y - voxelPos.y) +
                            abs(cData.Center.z - voxelPos.z));
                        
                        
                        if (d < gClusterDistanceBuffer[hashedPositionIndex.x])
                        {
                      
                            gClusterDistanceBuffer[hashedPositionIndex.x] = d;
                            gClusterAssignmentBuffer[hashedPositionIndex.x] = clusterIndex;
                            
                            cData.CenterAccum += float3(voxelPos);
                            cData.NormalAccum += avgNormal;
                            cData.VoxelCount += 1;
                            
                            gClusterDataBuffer[clusterIndex] = cData;
                        }
                    }
                }
            }
            
            clusterIndex = gNextCluster[clusterIndex];
        }
    }
    
    else if (CurrentPhase == 3)
    {
        if (GroupThreadIndex == 0)
        {
            uint clusterIndex = gTileBuffer[TileID];
            
            while (clusterIndex != UINT_MAX)
            {
                uint nextIndex = gNextCluster[clusterIndex];
                gNextCluster[clusterIndex] = UINT_MAX;
                clusterIndex = nextIndex;
            }
            
            gTileBuffer[TileID] = UINT_MAX;
        }
    }
}